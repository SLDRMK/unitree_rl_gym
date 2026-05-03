
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent

class G1Robot(LeggedRobot):
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()
        self._update_gait_phase()
        self._init_staged_training()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _update_gait_phase(self):
        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

    def _init_staged_training(self):
        self.lower_body_policy = None
        self.lower_body_dof_indices = None
        self.lower_body_actions = None
        self.student_lower_body_actions = None
        self.upper_body_dof_indices = None
        self.waist_dof_indices = None
        self.upper_body_periodic_scales = None

        staged_cfg = getattr(self.cfg, "staged_training", None)
        if staged_cfg is None:
            return
        dof_name_to_idx = {name: i for i, name in enumerate(self.dof_names)}
        self.upper_body_dof_indices = self._resolve_dof_indices(dof_name_to_idx, staged_cfg.upper_body_joint_names, "Upper-body")
        self.waist_dof_indices = self._resolve_dof_indices(dof_name_to_idx, staged_cfg.waist_joint_names, "Waist")
        self.upper_body_periodic_scales = torch.tensor(
            staged_cfg.upper_body_periodic_scales,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        if self.upper_body_periodic_scales.numel() != self.upper_body_dof_indices.numel():
            raise ValueError("upper_body_periodic_scales must match upper_body_joint_names length")

        if staged_cfg.stage != "upper_body":
            return
        if not staged_cfg.lower_body_checkpoint:
            raise ValueError("upper_body stage requires --lower_body_checkpoint or cfg.staged_training.lower_body_checkpoint")

        self.lower_body_dof_indices = self._resolve_dof_indices(dof_name_to_idx, staged_cfg.lower_body_joint_names, "Lower-body")
        self.lower_body_actions = torch.zeros(
            self.num_envs,
            staged_cfg.lower_body_num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.student_lower_body_actions = torch.zeros_like(self.lower_body_actions)

        policy_classes = {
            "ActorCritic": ActorCritic,
            "ActorCriticRecurrent": ActorCriticRecurrent,
        }
        policy_class = policy_classes[staged_cfg.lower_body_policy_class_name]
        self.lower_body_policy = policy_class(
            staged_cfg.lower_body_num_observations,
            staged_cfg.lower_body_num_privileged_obs,
            staged_cfg.lower_body_num_actions,
            actor_hidden_dims=staged_cfg.lower_body_actor_hidden_dims,
            critic_hidden_dims=staged_cfg.lower_body_critic_hidden_dims,
            activation=staged_cfg.lower_body_activation,
            rnn_type=staged_cfg.lower_body_rnn_type,
            rnn_hidden_size=staged_cfg.lower_body_rnn_hidden_size,
            rnn_num_layers=staged_cfg.lower_body_rnn_num_layers,
            init_noise_std=staged_cfg.lower_body_init_noise_std,
        ).to(self.device)

        checkpoint = torch.load(staged_cfg.lower_body_checkpoint, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError(
                "lower_body_checkpoint must be a training checkpoint like logs/g1/.../model_*.pt, "
                "not an exported TorchScript policy such as policy_lstm_1.pt"
            )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.lower_body_policy.load_state_dict(state_dict)
        self.lower_body_policy.eval()
        print(f"Loaded frozen lower-body policy from: {staged_cfg.lower_body_checkpoint}")

    def _resolve_dof_indices(self, dof_name_to_idx, joint_names, label):
        missing = [name for name in joint_names if name not in dof_name_to_idx]
        if missing:
            raise ValueError(f"{label} joints not found in asset: {missing}")
        return torch.tensor(
            [dof_name_to_idx[name] for name in joint_names],
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )

    def _staged_training_cfg(self):
        return getattr(self.cfg, "staged_training", None)

    def _is_upper_body_stage(self):
        staged_cfg = self._staged_training_cfg()
        return staged_cfg is not None and staged_cfg.stage == "upper_body"

    def _warmup_weight(self, warmup_s):
        if warmup_s <= 0.:
            return 1.
        elapsed_s = self.common_step_counter * self.dt
        return min(elapsed_s / warmup_s, 1.0)

    def _upper_body_action_weight(self):
        staged_cfg = self._staged_training_cfg()
        if staged_cfg is None:
            return 1.
        return staged_cfg.upper_body_action_scale * self._warmup_weight(staged_cfg.upper_body_action_warmup_s)

    def _get_lower_body_policy_obs(self):
        lower_ids = self.lower_body_dof_indices
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)
        return torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                          self.projected_gravity,
                          self.commands[:, :3] * self.commands_scale,
                          (self.dof_pos[:, lower_ids] - self.default_dof_pos[:, lower_ids]) * self.obs_scales.dof_pos,
                          self.dof_vel[:, lower_ids] * self.obs_scales.dof_vel,
                          self.lower_body_actions,
                          sin_phase,
                          cos_phase
                          ), dim=-1)

    def _compute_lower_body_actions(self):
        if self.lower_body_policy is None:
            return None
        with torch.inference_mode():
            lower_obs = self._get_lower_body_policy_obs()
            self.lower_body_actions = self.lower_body_policy.act_inference(lower_obs)
        return self.lower_body_actions

    def _reset_lower_body_policy(self, env_ids):
        if self.lower_body_policy is None or len(env_ids) == 0:
            return
        self.lower_body_actions[env_ids] = 0.
        self.student_lower_body_actions[env_ids] = 0.
        if not getattr(self.lower_body_policy, "is_recurrent", False):
            return
        hidden_states = self.lower_body_policy.memory_a.hidden_states
        if hidden_states is None:
            return
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        dones[env_ids] = True
        self.lower_body_policy.memory_a.reset(dones)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._reset_lower_body_policy(env_ids)

    def _reset_dofs(self, env_ids):
        if self._is_upper_body_stage():
            self.dof_pos[env_ids] = self.default_dof_pos
            self.dof_vel[env_ids] = 0.

            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            return
        return super()._reset_dofs(env_ids)

    def step(self, actions):
        if self.lower_body_policy is not None:
            actions = actions.to(self.device).clone()
            self.student_lower_body_actions = actions[:, self.lower_body_dof_indices].clone()
            actions[:, self.lower_body_dof_indices] = self._compute_lower_body_actions()
            actions[:, self.upper_body_dof_indices] *= self._upper_body_action_weight()
        return super().step(actions)

    def _post_physics_step_callback(self):
        self.update_feet_state()
        self._update_gait_phase()
        
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        
    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)

    def _reward_lower_body_action_match(self):
        if self.lower_body_policy is None:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        return torch.sum(torch.square(self.student_lower_body_actions - self.lower_body_actions), dim=1)

    def _upper_body_reward_active(self):
        return self.upper_body_dof_indices is not None

    def _upper_body_periodic_weight(self):
        staged_cfg = getattr(self.cfg, "staged_training", None)
        if staged_cfg is None:
            return 1.
        return self._warmup_weight(staged_cfg.upper_body_periodic_warmup_s)

    def _upper_body_periodic_target(self):
        staged_cfg = self.cfg.staged_training
        phase_signal = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        amplitude = staged_cfg.upper_body_periodic_amplitude * self._upper_body_periodic_weight()
        default_pos = self.default_dof_pos[:, self.upper_body_dof_indices]
        return default_pos + amplitude * phase_signal * self.upper_body_periodic_scales.unsqueeze(0)

    def _reward_upper_body_pos(self):
        if not self._upper_body_reward_active():
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        upper_ids = self.upper_body_dof_indices
        return torch.sum(torch.abs(self.dof_pos[:, upper_ids] - self.default_dof_pos[:, upper_ids]), dim=1)

    def _reward_upper_body_vel(self):
        if not self._upper_body_reward_active():
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        return torch.sum(torch.square(self.dof_vel[:, self.upper_body_dof_indices]), dim=1)

    def _reward_upper_body_action(self):
        if not self._upper_body_reward_active():
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        return torch.sum(torch.square(self.actions[:, self.upper_body_dof_indices]), dim=1)

    def _reward_upper_body_action_rate(self):
        if not self._upper_body_reward_active():
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        upper_ids = self.upper_body_dof_indices
        return torch.sum(torch.square(self.actions[:, upper_ids] - self.last_actions[:, upper_ids]), dim=1)

    def _reward_waist_still(self):
        if self.waist_dof_indices is None:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        waist_ids = self.waist_dof_indices
        pos_error = torch.abs(self.dof_pos[:, waist_ids] - self.default_dof_pos[:, waist_ids])
        vel_error = 0.1 * torch.square(self.dof_vel[:, waist_ids])
        return torch.sum(pos_error + vel_error, dim=1)

    def _reward_upper_body_periodic(self):
        if not self._upper_body_reward_active():
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        upper_ids = self.upper_body_dof_indices
        target = self._upper_body_periodic_target()
        return torch.sum(torch.abs(self.dof_pos[:, upper_ids] - target), dim=1)
    