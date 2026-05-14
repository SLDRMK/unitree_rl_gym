# SPDX-FileNotice: Extends NVIDIA / ETH rsl OnPolicyRunner for AMP discriminator reward training.
import os
import time
from collections import deque

import torch
import torch.nn.functional as F

from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from legged_gym.runners.discriminator_amp import AMPDiscriminator


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
        self.reward_scale_amp = float(self.amp_cfg.get("reward_scale", 1.0))

        self._last_amp_stats = {}

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

            amp_r_roll_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)

                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)

                    amp_obs_flat = torch.as_tensor(self.env.get_amp_observations(), device=self.device)
                    logits_rl = self.disc(amp_obs_flat)
                    probs = torch.sigmoid(logits_rl)
                    log_eps = float(self.amp_cfg.get("reward_log_eps", 1e-8))
                    r_amp = -torch.log(torch.clamp(1.0 - probs, min=log_eps))
                    rew_with_amp = rewards + self.reward_scale_amp * r_amp
                    amp_r_roll_sum += r_amp.detach()
                    self.amp_obs_rollout[i] = amp_obs_flat

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
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                collection_time = time.time() - start

                self.alg.compute_returns(critic_obs.to(self.device))

            self._last_amp_stats["mean_r_amp"] = amp_r_roll_sum.mean() / float(self.num_steps_per_env)

            mean_value_loss, mean_surrogate_loss = self.alg.update()

            amp_disc_loss_mean = None
            if getattr(self.env, "_motion_ref_bank", None) is not None:
                amp_disc_loss_mean = self._update_discriminator_amp()

            learn_time = time.time() - start - collection_time

            locs = {
                "it": it,
                "collection_time": collection_time,
                "learn_time": learn_time,
                "mean_value_loss": mean_value_loss,
                "mean_surrogate_loss": mean_surrogate_loss,
                "rewbuffer": rewbuffer,
                "lenbuffer": lenbuffer,
                "ep_infos": ep_infos.copy(),
                "num_learning_iterations": num_learning_iterations,
                "mean_r_amp_mean": float(self._last_amp_stats["mean_r_amp"].item()),
                "amp_disc_loss_mean": amp_disc_loss_mean,
            }
            if self.log_dir is not None:
                self.log(locs)

            ep_infos.clear()

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        if self.writer is not None:
            if locs.get("amp_disc_loss_mean") is not None:
                self.writer.add_scalar("Loss/discriminator_amp", locs["amp_disc_loss_mean"], locs["it"])
            mr = locs.get("mean_r_amp_mean")
            if mr is not None:
                self.writer.add_scalar("AMP/mean_step_r_amp", mr, locs["it"])
            mf = self._last_amp_stats.get("mean_logits_fake_mb")
            mer = self._last_amp_stats.get("mean_logits_real_mb")
            if mf is not None:
                self.writer.add_scalar("AMP/disc_logits_fake_mean", mf, locs["it"])
            if mer is not None:
                self.writer.add_scalar("AMP/disc_logits_expert_mean", mer, locs["it"])

        super().log(locs)

    def _sample_expert_amp(self, batch_n: int) -> torch.Tensor:
        bank = self.env._motion_ref_bank
        device = self.device
        hl = int(self.amp_cfg["history_frames"])
        cids = bank.sample_clip_indices(batch_n, device)
        center_t = bank.sample_phase_times(cids, device)

        dof_default = torch.as_tensor(self.env.default_dof_pos[0], device=device)
        osp = float(self.env.cfg.normalization.obs_scales.dof_pos)
        osv = float(self.env.cfg.normalization.obs_scales.dof_vel)
        return bank.gather_amp_dof_features(cids, center_t, hl, dof_default, osp, osv)

    def _update_discriminator_amp(self):
        flat = self.amp_obs_rollout.reshape(-1, self.flat_amp_dim)
        rollout_n = flat.shape[0]
        nb = max(1, int(self.amp_cfg.get("disc_minibatches", 4)))
        batch_fake = max(rollout_n // nb, 1)
        passes = max(1, int(self.amp_cfg.get("num_updates_per_iteration", 3)))

        tot_loss = 0.0
        n_batches = 0
        self.disc.train()
        mean_logit_fake = 0.0
        mean_logit_real = 0.0

        for _ in range(passes):
            pid = torch.randperm(rollout_n, device=self.device)
            shuffled = flat[pid]
            idx0 = 0
            while idx0 < rollout_n:
                fk = shuffled[idx0 : idx0 + batch_fake]
                bk = fk.shape[0]
                idx0 += batch_fake
                if bk == 0:
                    continue

                ex = self._sample_expert_amp(bk)
                lf = self.disc(fk.detach())
                lr_ex = self.disc(ex.detach())

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
                n_batches += 1
                mean_logit_fake += float(lf.mean().detach())
                mean_logit_real += float(lr_ex.mean().detach())

        self.disc.eval()

        if n_batches > 0:
            self._last_amp_stats["mean_logits_fake_mb"] = mean_logit_fake / float(n_batches)
            self._last_amp_stats["mean_logits_real_mb"] = mean_logit_real / float(n_batches)

        return tot_loss / max(n_batches, 1)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
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

        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")
