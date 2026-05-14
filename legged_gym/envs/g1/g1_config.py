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

    motion_ref 仍用于加载数据集；motion_ref_dof 奖励关闭。
    """

    class rewards(G1UpperBodyMotionRefCfg.rewards):
        class scales(G1UpperBodyMotionRefCfg.rewards.scales):
            motion_ref_dof = 0.0

    class amp:
        enabled = True
        # 判别器看的「扩张」帧数：沿时间串联 (q−q₀)·s_q 与 q̇·s_v
        history_frames = 4
        hidden_dims = [512, 256]
        activation = 'elu'
        disc_learning_rate = 3e-4
        disc_weight_decay = 0.0
        num_updates_per_iteration = 5
        disc_minibatches = 4
        disc_grad_norm = 1.0
        label_smoothing = 0.1
        # r_amp = reward_scale · (−log(1 − σ(D logits)))
        reward_scale = 1.0
        reward_log_eps = 1e-8


class G1UpperBodyAmpCfgPPO(G1UpperBodyMotionRefCfgPPO):
    class runner(G1UpperBodyMotionRefCfgPPO.runner):
        experiment_name = 'g1_upper_amp'
        algo_runner_class = 'OnPolicyRunnerAMP'

    class amp(G1UpperBodyAmpCfg.amp):
        """训练侧与 env 对齐的 AMP 超参字典（class_to_dict 展开）"""
        pass

