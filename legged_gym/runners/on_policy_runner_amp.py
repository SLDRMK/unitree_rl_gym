# SPDX-FileNotice: Extends NVIDIA / ETH rsl OnPolicyRunner for AMP discriminator reward training.
import os
import time
from collections import deque

import torch
import torch.nn.functional as F

from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from legged_gym.runners.discriminator_amp import AMPDiscriminator
from legged_gym.utils.helpers import extract_iteration_from_checkpoint_path


class OnPolicyRunnerAMP(OnPolicyRunner):
    """PPO with discriminator shaping: ``r_amp = -log(clamp(1 - sigmoid(D(x)), eps))``.

    Requires ``cfg.amp`` on env and flattened ``train_cfg['amp']`` (from ``class_to_dict``),
    motion reference bank on env, ``get_amp_observations()``, ``get_amp_flat_dim()``.
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        amp_dict = train_cfg.get("amp", None)
        if amp_dict is None:
            raise ValueError("train_cfg missing 'amp' section for AMP runner")

        arc = train_cfg["runner"].get("algo_runner_class", "OnPolicyRunner")
        if arc != "OnPolicyRunnerAMP":
            raise ValueError(f"algo_runner_class must be OnPolicyRunnerAMP, got {arc!r}")
        self.amp_cfg = amp_dict

        super().__init__(env, train_cfg, log_dir, device)

        self.flat_amp_dim = int(self.env.get_amp_flat_dim())

        hid = list(self.amp_cfg.get("hidden_dims", [512, 256]))
        act = str(self.amp_cfg.get("activation", "elu"))
        self.disc = AMPDiscriminator(self.flat_amp_dim, hid, act).to(self.device)
        dlr = float(self.amp_cfg.get("disc_learning_rate", 3e-4))
        wd = float(self.amp_cfg.get("disc_weight_decay", 0.0))
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=dlr, weight_decay=wd)
        self.disc.eval()

        self.amp_obs_rollout = torch.zeros(
            self.num_steps_per_env,
            env.num_envs,
            self.flat_amp_dim,
            device=self.device,
            dtype=torch.float32,
        )
        ls = float(self.amp_cfg.get("label_smoothing", 0.1))
        self._label_smooth_lo = ls
        self._label_smooth_hi = 1.0 - ls

        self._last_amp_stats = {}

        self._amp_curriculum_enabled = bool(self.amp_cfg.get("curriculum_enabled", False))
        self._amp_interp_milestones = bool(self.amp_cfg.get("curriculum_interp_between_milestones", False))
        self._amp_reward_scale_fallback = float(self.amp_cfg.get("reward_scale", 1.0))
        self._amp_schedule_pairs = self._normalize_amp_lambda_schedule(self.amp_cfg.get("reward_scale_schedule_iters"))
        self.min_scale_for_amp_disc = float(self.amp_cfg.get("min_scale_for_amp_disc", 0.0))
        self.reward_scale_amp = (
            float(self._lambda_amp_at_iter(int(self.current_learning_iteration)))
            if self._amp_curriculum_enabled and self._amp_schedule_pairs
            else self._amp_reward_scale_fallback
        )
        self._last_logged_lambda_amp = float("nan")

        # 历史策略 AMP 特征池（FIFO，仅存假样本）；容量由 cfg fake_amp_pool_capacity_rows 解析
        self._fk_pool_mix = float(self.amp_cfg.get("fake_pool_mix_fraction", 0.5))
        self.disc_acc_stop_above = float(self.amp_cfg.get("disc_stop_train_accuracy_above", 0.85))
        self.train_mask_prob = float(self.amp_cfg.get("train_feature_mask_prob", 0.0))

        cap_raw = int(self.amp_cfg.get("fake_amp_pool_capacity_rows", -1))
        rollout_rows = int(self.num_steps_per_env * self.env.num_envs)
        if cap_raw == 0:
            self._fk_pool_cap = 0
        elif cap_raw < 0:
            self._fk_pool_cap = max(8192, 8 * rollout_rows)
        else:
            self._fk_pool_cap = cap_raw
        self._fk_pool_buf = None
        self._fk_pool_overflow_resample = bool(self.amp_cfg.get("fake_pool_overflow_resample", True))

    def _masked_amp_for_disc_train(self, t: torch.Tensor) -> torch.Tensor:
        """按元素Bernoulli mask 置 0；仅用于判别器训练前向，增广鲁棒性。"""
        p = float(self.train_mask_prob)
        if p <= 0 or t.numel() == 0:
            return t
        m = torch.rand(t.shape, device=t.device, dtype=t.dtype) < p
        return t.masked_fill(m, 0.0)

    @staticmethod
    def _normalize_amp_lambda_schedule(raw):
        pairs = []
        if not raw:
            return pairs
        for row in raw:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                pairs.append((int(row[0]), float(row[1])))
        pairs.sort(key=lambda z: z[0])
        return pairs

    def _lambda_amp_at_iter(self, it: int) -> float:
        if not self._amp_curriculum_enabled or not self._amp_schedule_pairs:
            return float(self._amp_reward_scale_fallback)
        P = self._amp_schedule_pairs
        if self._amp_interp_milestones:
            if it <= P[0][0]:
                return float(P[0][1])
            if it >= P[-1][0]:
                return float(P[-1][1])
            for j in range(len(P) - 1):
                t0, s0 = P[j]
                t1, s1 = P[j + 1]
                if t0 <= it < t1:
                    span = float(max(t1 - t0, 1))
                    u = (float(it) - float(t0)) / span
                    return float((1.0 - u) * s0 + u * s1)
            return float(P[-1][1])
        lam_out = float(P[0][1])
        for t_start, ls in P:
            if it >= int(t_start):
                lam_out = float(ls)
            else:
                break
        return lam_out

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            lam = float(
                self._lambda_amp_at_iter(int(it))
                if self._amp_curriculum_enabled and self._amp_schedule_pairs
                else self._amp_reward_scale_fallback
            )
            self.reward_scale_amp = lam
            if lam != self._last_logged_lambda_amp:
                print(f"[AMP curriculum] learning_iteration {int(it)}: λ_amp={lam:g}")
                self._last_logged_lambda_amp = lam

            lam_thr = float(self.min_scale_for_amp_disc) + 1e-12
            amp_amp_active = lam > lam_thr

            amp_r_roll_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

            pending_rew = []
            pending_len = []

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)

                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)

                    if amp_amp_active:
                        amp_obs_flat = self.env.get_amp_observations()
                        if amp_obs_flat.device != self.device:
                            amp_obs_flat = amp_obs_flat.to(self.device)
                        logits_rl = self.disc(amp_obs_flat)
                        probs = torch.sigmoid(logits_rl)
                        log_eps = float(self.amp_cfg.get("reward_log_eps", 1e-8))
                        r_amp = -torch.log(torch.clamp(1.0 - probs, min=log_eps))
                        rew_with_amp = rewards + lam * r_amp
                        amp_r_roll_sum += lam * r_amp.detach()
                        self.amp_obs_rollout[i] = amp_obs_flat
                    else:
                        rew_with_amp = rewards

                    critic_obs = privileged_obs if privileged_obs is not None else obs

                    obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
                    rewards_dev = rew_with_amp.to(self.device)

                    dones = dones.to(self.device)

                    self.alg.process_env_step(rewards_dev, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards_dev
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False).view(-1)
                        if new_ids.numel() > 0:
                            pending_rew.append(cur_reward_sum[new_ids].detach())
                            pending_len.append(cur_episode_length[new_ids].detach())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                if self.log_dir is not None and pending_rew:
                    rewbuffer.extend(torch.cat(pending_rew).cpu().numpy().tolist())
                    lenbuffer.extend(torch.cat(pending_len).cpu().numpy().tolist())

                collection_time = time.time() - start

                t_returns = time.time()
                self.alg.compute_returns(critic_obs.to(self.device))
                returns_time = time.time() - t_returns

            self._last_amp_stats["mean_r_amp_scaled"] = amp_r_roll_sum.mean() / float(self.num_steps_per_env)

            t_ppo = time.time()
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            ppo_time = time.time() - t_ppo

            amp_disc_loss_mean = None
            t_disc = time.time()
            if getattr(self.env, "_motion_ref_bank", None) is not None and amp_amp_active:
                amp_disc_loss_mean = self._update_discriminator_amp()
            else:
                self._last_amp_stats.pop("mean_logits_fake_mb", None)
                self._last_amp_stats.pop("mean_logits_real_mb", None)
                for _k in (
                    "disc_mean_hard_acc",
                    "disc_accuracy_expert",
                    "disc_accuracy_policy_fake",
                    "disc_skipped_batches_high_acc",
                    "fake_pool_rows",
                ):
                    self._last_amp_stats.pop(_k, None)
            disc_time = time.time() - t_disc

            learn_time = returns_time + ppo_time + disc_time

            locs = {
                "it": it,
                "collection_time": collection_time,
                "learn_time": learn_time,
                "learn_returns_time": returns_time,
                "learn_ppo_time": ppo_time,
                "learn_disc_amp_time": disc_time,
                "mean_value_loss": mean_value_loss,
                "mean_surrogate_loss": mean_surrogate_loss,
                "rewbuffer": rewbuffer,
                "lenbuffer": lenbuffer,
                "ep_infos": ep_infos.copy(),
                "num_learning_iterations": num_learning_iterations,
                "mean_r_amp_mean": float(self._last_amp_stats["mean_r_amp_scaled"].item()),
                "amp_lambda_amp_mean": lam,
                "amp_disc_loss_mean": amp_disc_loss_mean,
            }
            if self.log_dir is not None:
                self.log(locs)

            ep_infos.clear()

            if it % self.save_interval == 0:
                self.save(
                    os.path.join(self.log_dir, "model_{}.pt".format(it)),
                    learning_iteration=it,
                )

        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)),
            learning_iteration=self.current_learning_iteration,
        )

    def log(self, locs, width=80, pad=35):
        if self.writer is not None:
            rt = locs.get("learn_returns_time")
            pt = locs.get("learn_ppo_time")
            dt_ = locs.get("learn_disc_amp_time")
            if rt is not None:
                self.writer.add_scalar("Timing/learn_returns", rt, locs["it"])
            if pt is not None:
                self.writer.add_scalar("Timing/learn_ppo", pt, locs["it"])
            if dt_ is not None:
                self.writer.add_scalar("Timing/learn_disc_amp", dt_, locs["it"])
            ct = locs.get("collection_time")
            if ct is not None:
                self.writer.add_scalar("Timing/collection", ct, locs["it"])
            if locs.get("amp_disc_loss_mean") is not None:
                self.writer.add_scalar("Loss/discriminator_amp", locs["amp_disc_loss_mean"], locs["it"])
            mr = locs.get("mean_r_amp_mean")
            if mr is not None:
                self.writer.add_scalar("AMP/mean_step_amp_scaled", mr, locs["it"])
            lam = locs.get("amp_lambda_amp_mean")
            if lam is not None:
                self.writer.add_scalar("AMP/lambda_amp", lam, locs["it"])
            mf = self._last_amp_stats.get("mean_logits_fake_mb")
            mer = self._last_amp_stats.get("mean_logits_real_mb")
            if mf is not None:
                self.writer.add_scalar("AMP/disc_logits_fake_mean", mf, locs["it"])
            if mer is not None:
                self.writer.add_scalar("AMP/disc_logits_expert_mean", mer, locs["it"])
            dh = self._last_amp_stats.get("disc_mean_hard_acc")
            if dh is not None:
                self.writer.add_scalar("AMP/disc_mean_hard_acc", dh, locs["it"])
                self.writer.add_scalar("AMP/disc_accuracy_balanced", dh, locs["it"])
                self.writer.add_scalar("Discriminator/accuracy_balanced", dh, locs["it"])
            dex = self._last_amp_stats.get("disc_accuracy_expert")
            if dex is not None:
                self.writer.add_scalar("AMP/disc_accuracy_expert", dex, locs["it"])
                self.writer.add_scalar("Discriminator/accuracy_expert", dex, locs["it"])
            dfk = self._last_amp_stats.get("disc_accuracy_policy_fake")
            if dfk is not None:
                self.writer.add_scalar("AMP/disc_accuracy_policy_fake", dfk, locs["it"])
                self.writer.add_scalar("Discriminator/accuracy_policy_fake", dfk, locs["it"])
            sk = self._last_amp_stats.get("disc_skipped_batches_high_acc")
            if sk is not None:
                self.writer.add_scalar("AMP/disc_skipped_batches_high_acc", sk, locs["it"])
            fr = self._last_amp_stats.get("fake_pool_rows")
            if fr is not None:
                self.writer.add_scalar("AMP/fake_pool_rows", fr, locs["it"])

        super().log(locs)

    @staticmethod
    def _disc_hard_accuracy_triplet(lf: torch.Tensor, lr_ex: torch.Tensor) -> tuple:
        """(平衡准确率, 专家判真准确率, 策略假判假准确率)，均为 [0,1]。"""
        with torch.no_grad():
            pr = torch.sigmoid(lr_ex)
            pf = torch.sigmoid(lf)
            acc_ex = float((pr >= 0.5).float().mean())
            acc_fk = float((pf < 0.5).float().mean())
            return 0.5 * (acc_ex + acc_fk), acc_ex, acc_fk

    def _fk_pool_trim(self, cat: torch.Tensor) -> torch.Tensor:
        """超容量时裁剪：默认同设备上无放回均匀抽样，等价于随机扔掉部分旧/新样本。"""
        n = int(cat.shape[0])
        cap = int(self._fk_pool_cap)
        if n <= cap:
            return cat
        if self._fk_pool_overflow_resample:
            perm = torch.randperm(n, device=cat.device)[:cap]
            return cat[perm]
        return cat[-cap:]

    def _enqueue_fake_rollout(self, flat_rollout_2d: torch.Tensor):
        if self._fk_pool_cap <= 0:
            return
        x = flat_rollout_2d.reshape(-1, self.flat_amp_dim).detach()
        prev = getattr(self, "_fk_pool_buf", None)
        if prev is None or prev.shape[0] == 0:
            self._fk_pool_buf = self._fk_pool_trim(x)
            return
        cat = torch.cat([prev, x], dim=0)
        self._fk_pool_buf = self._fk_pool_trim(cat)

    def _assemble_fake_minibatch(self, fk_piece: torch.Tensor) -> torch.Tensor:
        """fk_piece: [bk, D] 来自当前 rollout 打散片段；按比例混入历史策略池。"""
        bk = int(fk_piece.shape[0])
        if bk == 0:
            return fk_piece
        buf = self._fk_pool_buf
        mix = float(self._fk_pool_mix)
        if buf is None or buf.shape[0] == 0 or self._fk_pool_cap <= 0 or mix <= 1e-8:
            return fk_piece
        n_pool = min(bk, int(round(bk * mix)))
        n_curr = bk - n_pool
        rows = []
        if n_pool > 0:
            bi = min(n_pool, int(buf.shape[0]))
            rp = torch.randint(0, int(buf.shape[0]), (bi,), device=self.device)
            rows.append(buf[rp])
            n_curr += n_pool - bi
        if n_curr > 0:
            ni = min(n_curr, int(fk_piece.shape[0]))
            rc = torch.randint(0, int(fk_piece.shape[0]), (ni,), device=self.device)
            rows.append(fk_piece[rc])
        if not rows:
            return fk_piece
        out = torch.cat(rows, dim=0)
        if out.shape[0] < bk:
            short = bk - out.shape[0]
            rc = torch.randint(0, int(fk_piece.shape[0]), (short,), device=self.device)
            out = torch.cat([out, fk_piece[rc]], dim=0)
        elif out.shape[0] > bk:
            perm = torch.randperm(out.shape[0], device=self.device)[:bk]
            out = out[perm]
        return out

    def _sample_expert_amp(self, batch_n: int) -> torch.Tensor:
        bank = self.env._motion_ref_bank
        device = self.device
        hl = int(self.amp_cfg["history_frames"])
        cids = bank.sample_clip_indices(batch_n, device)
        center_t = bank.sample_phase_times(cids, device)

        dof_default = torch.as_tensor(self.env.default_dof_pos[0], device=device)
        osp = float(self.env.cfg.normalization.obs_scales.dof_pos)
        osv = float(self.env.cfg.normalization.obs_scales.dof_vel)
        hw = self.amp_cfg.get("history_window_s", None)
        hw_f = float(hw) if hw is not None else None
        if hw_f is not None and hw_f <= 0:
            hw_f = None
        return bank.gather_amp_dof_features(
            cids,
            center_t,
            hl,
            dof_default,
            osp,
            osv,
            history_window_s=hw_f,
        )

    def _update_discriminator_amp(self):
        flat = self.amp_obs_rollout.reshape(-1, self.flat_amp_dim)
        rollout_n = flat.shape[0]
        nb = max(1, int(self.amp_cfg.get("disc_minibatches", 4)))
        batch_fake = max(rollout_n // nb, 1)
        passes = max(1, int(self.amp_cfg.get("num_updates_per_iteration", 3)))
        acc_thr = float(self.disc_acc_stop_above)

        tot_loss = 0.0
        n_batches_updated = 0
        n_batches_total = 0
        n_skipped_high_acc = 0
        sum_hard_acc = 0.0
        sum_acc_expert = 0.0
        sum_acc_policy_fake = 0.0

        mean_logit_fake = 0.0
        mean_logit_real = 0.0

        self.disc.train()
        for _ in range(passes):
            pid = torch.randperm(rollout_n, device=self.device)
            shuffled = flat[pid]
            idx0 = 0
            while idx0 < rollout_n:
                fk_piece = shuffled[idx0 : idx0 + batch_fake]
                bk = fk_piece.shape[0]
                idx0 += batch_fake
                if bk == 0:
                    continue

                fk_src = self._assemble_fake_minibatch(fk_piece)
                fk_train = self._masked_amp_for_disc_train(fk_src.detach())
                ex_raw = self._sample_expert_amp(int(bk)).detach()
                ex_train = self._masked_amp_for_disc_train(ex_raw)

                lf = self.disc(fk_train)
                lr_ex = self.disc(ex_train)
                mean_logit_fake += float(lf.mean().detach())
                mean_logit_real += float(lr_ex.mean().detach())
                n_batches_total += 1

                acc_bal, acc_ex, acc_fk = self._disc_hard_accuracy_triplet(lf, lr_ex)
                sum_hard_acc += acc_bal
                sum_acc_expert += acc_ex
                sum_acc_policy_fake += acc_fk

                skip_high_acc = acc_thr > 0.0 and acc_bal > acc_thr

                if skip_high_acc:
                    n_skipped_high_acc += 1
                    continue

                tgt_f = torch.full_like(lf, self._label_smooth_lo)
                tgt_r = torch.full_like(lr_ex, self._label_smooth_hi)
                loss = F.binary_cross_entropy_with_logits(lf, tgt_f) + F.binary_cross_entropy_with_logits(
                    lr_ex, tgt_r
                )

                self.disc_opt.zero_grad()
                loss.backward()
                gn = float(self.amp_cfg.get("disc_grad_norm", 1.0))
                torch.nn.utils.clip_grad_norm_(self.disc.parameters(), gn)
                self.disc_opt.step()

                tot_loss += loss.item()
                n_batches_updated += 1

        self.disc.eval()
        self._enqueue_fake_rollout(flat.detach())

        if self._fk_pool_buf is not None:
            self._last_amp_stats["fake_pool_rows"] = float(self._fk_pool_buf.shape[0])
        else:
            self._last_amp_stats["fake_pool_rows"] = 0.0

        denom_logit = float(max(n_batches_total, 1))
        self._last_amp_stats["mean_logits_fake_mb"] = mean_logit_fake / denom_logit
        self._last_amp_stats["mean_logits_real_mb"] = mean_logit_real / denom_logit

        if n_batches_total > 0:
            d_ = float(n_batches_total)
            mean_bal = sum_hard_acc / d_
            mean_ex = sum_acc_expert / d_
            mean_fk = sum_acc_policy_fake / d_
            self._last_amp_stats["disc_mean_hard_acc"] = mean_bal
            self._last_amp_stats["disc_accuracy_expert"] = mean_ex
            self._last_amp_stats["disc_accuracy_policy_fake"] = mean_fk
        else:
            self._last_amp_stats.pop("disc_mean_hard_acc", None)
            self._last_amp_stats.pop("disc_accuracy_expert", None)
            self._last_amp_stats.pop("disc_accuracy_policy_fake", None)
        self._last_amp_stats["disc_skipped_batches_high_acc"] = float(n_skipped_high_acc)

        if n_batches_updated == 0:
            return None
        return tot_loss / float(n_batches_updated)

    def save(self, path, infos=None, learning_iteration=None):
        # Parent rsl OnPolicyRunner only bumps current_learning_iteration after the whole learn() loop,
        # so periodic ckpts would save wrong "iter" unless we pin the loop index explicitly.
        it_save = (
            self.current_learning_iteration
            if learning_iteration is None
            else learning_iteration
        )
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": int(it_save),
                "infos": infos,
                "discriminator_state_dict": self.disc.state_dict(),
                "discriminator_optimizer_state_dict": self.disc_opt.state_dict(),
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            odisc = loaded_dict.get("discriminator_optimizer_state_dict")
            if odisc is not None:
                self.disc_opt.load_state_dict(odisc)

        dd = loaded_dict.get("discriminator_state_dict")
        if dd is not None:
            self.disc.load_state_dict(dd)

        stored_it = loaded_dict.get("iter", None)
        if stored_it is None:
            stored_it = 0
        stored_it = int(stored_it)
        fn_it = extract_iteration_from_checkpoint_path(path)
        if fn_it is not None and stored_it == 0 and fn_it > 0:
            stored_it = fn_it

        self.current_learning_iteration = stored_it
        self.reward_scale_amp = (
            float(self._lambda_amp_at_iter(int(self.current_learning_iteration)))
            if self._amp_curriculum_enabled and self._amp_schedule_pairs
            else self._amp_reward_scale_fallback
        )
        self._last_logged_lambda_amp = float("nan")
        return loaded_dict.get("infos")
