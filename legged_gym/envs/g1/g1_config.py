from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }
    
    class env(LeggedRobotCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

class G1UpperBodyCfg( G1RoughCfg ):
    class init_state( G1RoughCfg.init_state ):
        default_joint_angles = {
           'left_hip_pitch_joint' : -0.1,
           'left_hip_roll_joint' : 0,
           'left_hip_yaw_joint' : 0.,
           'left_knee_joint' : 0.3,
           'left_ankle_pitch_joint' : -0.2,
           'left_ankle_roll_joint' : 0,
           'right_hip_pitch_joint' : -0.1,
           'right_hip_roll_joint' : 0,
           'right_hip_yaw_joint' : 0.,
           'right_knee_joint' : 0.3,
           'right_ankle_pitch_joint': -0.2,
           'right_ankle_roll_joint' : 0,
           'waist_yaw_joint' : 0.,
           'left_shoulder_pitch_joint' : 0.35,
           'left_shoulder_roll_joint' : 0.16,
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint' : 0.87,
           'left_wrist_roll_joint' : 0.,
           'right_shoulder_pitch_joint' : 0.35,
           'right_shoulder_roll_joint' : -0.16,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.87,
           'right_wrist_roll_joint' : 0.,
        }

    class env( G1RoughCfg.env ):
        num_observations = 80
        num_privileged_obs = 83
        num_actions = 23

    class control( G1RoughCfg.control ):
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     'waist': 300,
                     'shoulder': 80,
                     'elbow': 60,
                     'wrist': 20,
                     }
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'waist': 6,
                     'shoulder': 2,
                     'elbow': 2,
                     'wrist': 1,
                     }

    class asset( G1RoughCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_23dof.urdf'
        armature = 0.01

    class commands( G1RoughCfg.commands ):
        class ranges( G1RoughCfg.commands.ranges ):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]

    class domain_rand( G1RoughCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class staged_training:
        # base: no override, upper_body: frozen lower-body controller, joint_finetune: train all DoFs.
        stage = 'upper_body'
        lower_body_checkpoint = ''
        lower_body_policy_class_name = 'ActorCriticRecurrent'
        lower_body_num_observations = 47
        lower_body_num_privileged_obs = 50
        lower_body_num_actions = 12
        lower_body_actor_hidden_dims = [32]
        lower_body_critic_hidden_dims = [32]
        lower_body_activation = 'elu'
        lower_body_rnn_type = 'lstm'
        lower_body_rnn_hidden_size = 64
        lower_body_rnn_num_layers = 1
        lower_body_init_noise_std = 0.8
        lower_body_joint_names = [
            'left_hip_pitch_joint',
            'left_hip_roll_joint',
            'left_hip_yaw_joint',
            'left_knee_joint',
            'left_ankle_pitch_joint',
            'left_ankle_roll_joint',
            'right_hip_pitch_joint',
            'right_hip_roll_joint',
            'right_hip_yaw_joint',
            'right_knee_joint',
            'right_ankle_pitch_joint',
            'right_ankle_roll_joint',
        ]
        upper_body_joint_names = [
            'waist_yaw_joint',
            'left_shoulder_pitch_joint',
            'left_shoulder_roll_joint',
            'left_shoulder_yaw_joint',
            'left_elbow_joint',
            'left_wrist_roll_joint',
            'right_shoulder_pitch_joint',
            'right_shoulder_roll_joint',
            'right_shoulder_yaw_joint',
            'right_elbow_joint',
            'right_wrist_roll_joint',
        ]
        waist_joint_names = ['waist_yaw_joint']
        upper_body_action_scale = 0.55
        upper_body_action_warmup_s = 600.0
        upper_body_periodic_amplitude = 0.18
        upper_body_periodic_warmup_s = 600.0
        upper_body_constraint_decay_s = 1200.0
        upper_body_constraint_min_weight = 0.25
        upper_body_motion_reward_warmup_s = 1200.0
        upper_body_motion_vel_clip = 2.0
        upper_body_motion_command_threshold = 0.1
        upper_body_periodic_scales = [
            0.0,   # waist_yaw_joint
            1.0,   # left_shoulder_pitch_joint
            0.2,   # left_shoulder_roll_joint
            0.0,   # left_shoulder_yaw_joint
            -0.5,  # left_elbow_joint
            0.0,   # left_wrist_roll_joint
            -1.0,  # right_shoulder_pitch_joint
            -0.2,  # right_shoulder_roll_joint
            0.0,   # right_shoulder_yaw_joint
            -0.5,  # right_elbow_joint
            0.0,   # right_wrist_roll_joint
        ]

    class rewards( G1RoughCfg.rewards ):
        only_positive_rewards = False
        class scales( G1RoughCfg.rewards.scales ):
            termination = -200.0
            lower_body_action_match = -1.0
            upper_body_pos = -0.12
            upper_body_vel = -0.03
            upper_body_action = -0.01
            upper_body_action_rate = -0.03
            waist_still = -0.3
            upper_body_periodic = -0.25
            upper_body_motion_vel = 0.025

class G1UpperBodyMotionRefCfg(G1UpperBodyCfg):
    """档位 A：对照 mink pickle 全局关节相位做运动匹配塑形（不改 PPO/RNN 本体）。"""

    class motion_ref:
        enabled = True
        # 设置目录或外层用环境变量 MOTION_REF_DATA_DIR 覆盖（需在启动前传给 Python）。
        data_dir = ""
        glob_pattern = "*.pkl"
        clip_limit = None
        # σ 为关节误差向量 x=q-q_ref 的 L2 范数尺度（弧度）；每步 σ <- max(sigma_min, min(E[||x||_2], σ))。
        # 奖励用 mse = mean((q-qref)^2)（见 err_reduce），exp(-mse / σ^2)。
        sigma = 1.0
        sigma_min = 0.02
        err_reduce = "mean"  # "sum" 保留旧语义（会与 29dof 切片缩放不一致，易出现极小 exp）
        warmup_s = 0.0
        command_gate = True
        command_threshold = 0.08
        curriculum_enabled = True
        # 对 batch 内各 env 的 ||x||_2 先取均值，再对此标量做 EMA（0 表示不用 EMA，直接用当前步均值）
        curriculum_norm_ema_alpha = 0.0

    class rewards(G1UpperBodyCfg.rewards):
        class scales(G1UpperBodyCfg.rewards.scales):
            motion_ref_dof = 0.6

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1'

class G1UpperBodyCfgPPO( G1RoughCfgPPO ):
    class policy( G1RoughCfgPPO.policy ):
        init_noise_std = 0.8
        actor_hidden_dims = [256, 128, 128]
        critic_hidden_dims = [256, 128, 128]
    class runner( G1RoughCfgPPO.runner ):
        policy_class_name = "ActorCritic"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1_upper'

class G1UpperBodyMotionRefCfgPPO(G1UpperBodyCfgPPO):
    class runner(G1UpperBodyCfgPPO.runner):
        experiment_name = 'g1_upper_motion_ref'


class G1UpperBodyAmpCfg(G1UpperBodyMotionRefCfg):
    """AMP：用判别器对齐 mink 参考动作分布（多帧关节特征），不使用逐步追踪 q≈q_ref。

    motion_ref 仍用于加载专家 clip；motion_ref_dof=0（不启用关节参考追踪奖励）。
    """

    class commands(G1UpperBodyMotionRefCfg.commands):
        class ranges(G1UpperBodyMotionRefCfg.commands.ranges):
            def mul(xs, ratio):
                return [x * ratio for x in xs]
            lin_vel_x = [0.0, 1.0]
            lin_vel_y = mul(G1UpperBodyMotionRefCfg.commands.ranges.lin_vel_y, 0.1)
            ang_vel_yaw = mul(G1UpperBodyMotionRefCfg.commands.ranges.ang_vel_yaw, 0.1)
            heading = mul(G1UpperBodyMotionRefCfg.commands.ranges.heading, 0.1)

    class rewards(G1UpperBodyMotionRefCfg.rewards):
        class scales(G1UpperBodyMotionRefCfg.rewards.scales):
            motion_ref_dof = 0.0

    class amp:
        enabled = True
        # 判别器输入：在 history_window_s 秒内均匀取 history_frames 帧（拼接相对 q₀ 的 dof 与 dof_vel）。
        # 默认 0.9s / 10 帧 → 相邻帧间隔 0.1s（仿真环形缓冲与专家轨迹一致）。
        history_frames = 10
        history_window_s = 0.9
        # hidden_dims = [512, 256]
        hidden_dims = [128, 128]
        activation = 'elu'
        # disc_learning_rate = 3e-4
        disc_learning_rate = 1e-5
        disc_weight_decay = 0.0
        # num_updates_per_iteration = 5
        num_updates_per_iteration = 1
        disc_minibatches = 4
        disc_grad_norm = 1.0
        label_smoothing = 0.1
        reward_log_eps = 1e-8

        # 判别训练：准确率超阈值则跳过优化（降至阈值以下再继续），抑制 D 过快过拟合
        disc_stop_train_accuracy_above = 0.85  # ≤0 关闭门控（始终更新）
        # 负样本 = 本轮 rollout + 特征池。容量：-1 自动；0 关闭。-1 时 cap=max(8192, 8×steps×envs)
        fake_amp_pool_capacity_rows = -1
        # 合并后超容量时 True：从 [旧∪新] 无放回均匀抽满容量（随机淘汰部分旧/新）；False：FIFO 只保留末尾
        fake_pool_overflow_resample = True
        fake_pool_mix_fraction = 0.5  # 每个 fake minibatch 中从池中采样的比例（余下来自当前 rollout）
        # 判别器输入随机 mask（仅训练时）；每维独立Bernoulli置 0；0 关闭
        train_feature_mask_prob = 0.1

        # ---- λ_amp 课程（与 reward_scale_schedule_iters 第一项对齐；关闭课程时用 reward_scale 常数）----
        curriculum_enabled = True
        # True：在里程碑之间对 λ_amp 线性插值；False：阶梯常数（每个 milestone 生效到下一里程碑）
        curriculum_interp_between_milestones = False
        # (learning_iteration ≥ 阈值, λ_amp)；参考：Phase0 纯 PPO；Phase1 小 AMP 0.02~0.05；Phase2 抬到 0.1→0.2；终值常用 0.1~0.3
        # 以下默认值按 G1UpperBodyAmpCfgPPO.runner.max_iterations=15000：最后一档 λ=0.25 自迭代 ≥8500 持续到训练结束（较长末段）。
        # reward_scale_schedule_iters = (
        #     (0, 0.0),
        #     (2000, 0.035),
        #     (5000, 0.10),
        #     (7000, 0.20),
        #     (8500, 0.25),
        # )
        reward_scale_schedule_iters = (
            (0, 0.0),
            (2000, 0.035),
            (6000, 0.07),
            (12000, 0.1),
            (18000, 0.15),
        )
        # curriculum_enabled=False 时作为固定 λ_amp
        reward_scale = 0.25
        # λ_amp≤此值时不跑判别器前向与更新（通常为 0：Phase0 纯 PPO）
        min_scale_for_amp_disc = 0.0


class G1UpperBodyAmpCfgPPO(G1UpperBodyMotionRefCfgPPO):
    class runner(G1UpperBodyMotionRefCfgPPO.runner):
        experiment_name = 'g1_upper_amp'
        algo_runner_class = 'OnPolicyRunnerAMP'
        max_iterations = 25000

    class amp(G1UpperBodyAmpCfg.amp):
        """训练侧与 env 对齐的 AMP 超参字典（class_to_dict 展开）"""
        pass

