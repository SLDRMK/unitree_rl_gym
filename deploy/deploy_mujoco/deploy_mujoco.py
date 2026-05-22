import select
import sys
import threading
import time
import tty
import termios

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def resolve_path(path):
    return path.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)


def reset_policy_memory(policy):
    if hasattr(policy, "reset_memory"):
        policy.reset_memory()


def run_policy(policy, obs):
    obs_tensor = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        action = policy(obs_tensor).detach().cpu().numpy().squeeze()
    return np.asarray(action, dtype=np.float32)


def build_policy_observation(
    num_policy_actions,
    num_policy_obs,
    qj,
    dqj,
    quat,
    omega,
    cmd,
    previous_action,
    default_angles,
    scales,
    phase,
):
    obs = np.zeros(num_policy_obs, dtype=np.float32)
    qj_rel = (qj[:num_policy_actions] - default_angles[:num_policy_actions]) * scales["dof_pos"]
    dqj_scaled = dqj[:num_policy_actions] * scales["dof_vel"]
    gravity_orientation = get_gravity_orientation(quat)
    omega_scaled = omega * scales["ang_vel"]
    sin_phase = np.sin(2 * np.pi * phase)
    cos_phase = np.cos(2 * np.pi * phase)

    obs[:3] = omega_scaled
    obs[3:6] = gravity_orientation
    obs[6:9] = cmd * scales["cmd"]
    obs[9 : 9 + num_policy_actions] = qj_rel
    obs[9 + num_policy_actions : 9 + 2 * num_policy_actions] = dqj_scaled
    obs[9 + 2 * num_policy_actions : 9 + 3 * num_policy_actions] = previous_action[:num_policy_actions]
    obs[9 + 3 * num_policy_actions : 9 + 3 * num_policy_actions + 2] = np.array([sin_phase, cos_phase])
    return obs


def load_policy(path, name):
    policy = torch.jit.load(resolve_path(path))
    policy.eval()
    reset_policy_memory(policy)
    print(f"Loaded {name} policy: {resolve_path(path)}")
    return policy


def get_actuated_joint_names(model):
    joint_names = []
    for actuator_id in range(model.nu):
        joint_id = model.actuator_trnid[actuator_id, 0]
        joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
    return joint_names


def get_model_indices(model, joint_names):
    qpos_indices = []
    qvel_indices = []
    actuator_indices = []
    torque_limits = []

    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint '{name}' was not found in Mujoco model")

        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"Actuator '{name}' was not found in Mujoco model")

        actuator_joint_id = model.actuator_trnid[actuator_id, 0]
        if actuator_joint_id != joint_id:
            actuator_joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, actuator_joint_id)
            raise ValueError(f"Actuator '{name}' drives '{actuator_joint_name}', expected '{name}'")

        qpos_indices.append(model.jnt_qposadr[joint_id])
        qvel_indices.append(model.jnt_dofadr[joint_id])
        actuator_indices.append(actuator_id)

        lower, upper = model.jnt_actfrcrange[joint_id]
        torque_limits.append(max(abs(lower), abs(upper)))

    return (
        np.asarray(qpos_indices, dtype=np.int32),
        np.asarray(qvel_indices, dtype=np.int32),
        np.asarray(actuator_indices, dtype=np.int32),
        np.asarray(torque_limits, dtype=np.float32),
    )


def validate_config(num_actions, policy_num_actions, default_angles, kps, kds, joint_names, model):
    if model.nu != num_actions:
        raise ValueError(f"Config num_actions={num_actions}, but Mujoco model has {model.nu} actuators")
    if len(joint_names) != num_actions:
        raise ValueError(f"joint_names length={len(joint_names)}, expected {num_actions}")
    if len(default_angles) != num_actions:
        raise ValueError(f"default_angles length={len(default_angles)}, expected {num_actions}")
    if len(kps) != num_actions or len(kds) != num_actions:
        raise ValueError(f"kps/kds length must both equal num_actions={num_actions}")
    if policy_num_actions > num_actions:
        raise ValueError(f"policy_num_actions={policy_num_actions}, robot num_actions={num_actions}")


def reset_robot_state(model, data, default_angles, qpos_indices):
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qpos[qpos_indices] = default_angles
    mujoco.mj_forward(model, data)


class RealtimeKeyboardController:
    def __init__(self, cmd_init, step=0.2, max_value=2.0):
        self.running = True
        self.current_cmd = "无命令"
        self.vel_cmd = np.array(cmd_init, dtype=np.float32)
        self.step = step
        self.max_value = max_value
        self._lock = threading.Lock()

    def start_keyboard_listener(self):
        """Start a background thread for non-blocking keyboard control."""
        if not sys.stdin.isatty():
            print("stdin 不是终端，跳过键盘控制，使用配置中的初始速度命令。")
            return

        print("实时键盘控制说明:")
        print("  w/s: 前进/后退，每次 +/-0.2")
        print("  a/d: 左移/右移，每次 +/-0.2")
        print("  j/l: 左转/右转，每次 +/-0.2")
        print("  z 或 空格: 速度清零")
        print("  q: 退出程序")
        print(
            f"初始速度命令: "
            f"[{self.vel_cmd[0]:.2f}, {self.vel_cmd[1]:.2f}, {self.vel_cmd[2]:.2f}]"
        )

        def keyboard_listener():
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                while self.running:
                    if not select.select([sys.stdin], [], [], 0.01)[0]:
                        continue
                    key = sys.stdin.read(1).lower()
                    self.process_key(key)
            except KeyboardInterrupt:
                self.running = False
            except Exception as exc:
                self.running = False
                print(f"键盘监听错误: {exc}")
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
        keyboard_thread.start()

    def process_key(self, key):
        with self._lock:
            if key == "q":
                self.running = False
                self.current_cmd = "退出程序"
            elif key == "w":
                self.vel_cmd[0] = min(self.vel_cmd[0] + self.step, self.max_value)
                self.current_cmd = f"前进加速: {self.vel_cmd[0]:.1f}"
            elif key == "s":
                self.vel_cmd[0] = max(self.vel_cmd[0] - self.step, -self.max_value)
                self.current_cmd = f"后退加速: {self.vel_cmd[0]:.1f}"
            elif key == "a":
                self.vel_cmd[1] = min(self.vel_cmd[1] + self.step, self.max_value)
                self.current_cmd = f"左移加速: {self.vel_cmd[1]:.1f}"
            elif key == "d":
                self.vel_cmd[1] = max(self.vel_cmd[1] - self.step, -self.max_value)
                self.current_cmd = f"右移加速: {self.vel_cmd[1]:.1f}"
            elif key == "j":
                self.vel_cmd[2] = min(self.vel_cmd[2] + self.step, self.max_value)
                self.current_cmd = f"左转加速: {self.vel_cmd[2]:.1f}"
            elif key == "l":
                self.vel_cmd[2] = max(self.vel_cmd[2] - self.step, -self.max_value)
                self.current_cmd = f"右转加速: {self.vel_cmd[2]:.1f}"
            elif key in {"z", " "}:
                self.vel_cmd[:] = 0.0
                self.current_cmd = "速度清零"
            else:
                return

            print(
                f"{self.current_cmd}，当前速度命令: "
                f"[{self.vel_cmd[0]:.2f}, {self.vel_cmd[1]:.2f}, {self.vel_cmd[2]:.2f}]"
            )

    def get_vel_cmd(self):
        with self._lock:
            return self.vel_cmd.copy()


def _apply_tracking_side_camera(viewer, body_id: int, side: str, distance: float, elevation: float, azimuth_deg):
    """用 mjCAMERA_TRACKING：球坐标在被跟踪 body's 坐标系内，可实现侧向锁定并随机器人平移与偏航转动。"""
    cam = viewer.cam
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = int(body_id)
    cam.distance = float(distance)
    cam.elevation = float(elevation)
    cam.orthographic = False
    # Unitree MJCF：pelvis 左腿在 body's +Y 侧 ⇒ 机器人的「右侧」大致为 body's -local Y，
    # 观感若左右反了，可用 --camera-follow-azimuth 覆盖或改用 `--camera-follow-side left`。
    if azimuth_deg is not None:
        cam.azimuth = float(azimuth_deg)
    elif side == "right":
        cam.azimuth = -90.0
    elif side == "left":
        cam.azimuth = 90.0
    else:
        cam.azimuth = -90.0


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    parser.add_argument(
        "--camera-follow-side",
        choices=("none", "right", "left"),
        default="right",
        help="侧向锁定相机（MuJoCo 跟踪模式）：在机器人身体坐标系固定方位角与距离并随躯干运动。默认 none（自由视角）。",
    )
    parser.add_argument(
        "--camera-follow-distance",
        type=float,
        default=2.5,
        help="跟踪相机与被跟踪 body's 焦距距离（MuJoCo 球坐标半径），默认 2.8。",
    )
    parser.add_argument(
        "--camera-track-body",
        type=str,
        default="pelvis",
        help="跟踪相机的 body 名称，默认 pelvis（G1/H1 MJCF）。",
    )
    parser.add_argument(
        "--camera-follow-elevation",
        type=float,
        default=-12.0,
        help="跟踪相机俯仰角（度），略俯视便于看全身，默认 -12。",
    )
    parser.add_argument(
        "--camera-follow-azimuth",
        type=float,
        default=None,
        help="覆盖自动方位角（度）；不设则右侧约 -90°、左侧约 +90°。",
    )
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = resolve_path(config["policy_path"])
        lower_body_policy_path = config.get("lower_body_policy_path")
        xml_path = resolve_path(config["xml_path"])

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        policy_num_actions = config.get("policy_num_actions", num_actions)
        policy_num_obs = config.get("policy_num_obs", config.get("num_obs", 11 + 3 * policy_num_actions))
        lower_body_num_actions = config.get("lower_body_num_actions", 12)
        lower_body_num_obs = config.get("lower_body_num_obs", 47)
        upper_body_action_scale = config.get("upper_body_action_scale", 1.0)
        clip_actions = config.get("clip_actions", None)
        clip_torques = config.get("clip_torques", True)
        joint_names = config.get("joint_names", None)
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    policy_action = np.zeros(policy_num_actions, dtype=np.float32)
    lower_body_action = np.zeros(lower_body_num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs_scales = {
        "ang_vel": ang_vel_scale,
        "dof_pos": dof_pos_scale,
        "dof_vel": dof_vel_scale,
        "cmd": cmd_scale,
    }

    counter = 0
    printed_policy_debug = False
    keyboard_controller = RealtimeKeyboardController(cmd_init=cmd)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    if joint_names is None:
        joint_names = get_actuated_joint_names(m)
    validate_config(num_actions, policy_num_actions, default_angles, kps, kds, joint_names, m)
    qpos_indices, qvel_indices, actuator_indices, torque_limits = get_model_indices(m, joint_names)
    reset_robot_state(m, d, default_angles, qpos_indices)

    follow_side = None if args.camera_follow_side == "none" else args.camera_follow_side
    cam_track_body_id = None
    if follow_side is not None:
        cam_track_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, args.camera_track_body)
        if cam_track_body_id < 0:
            raise ValueError(
                f"未找到 body {args.camera_track_body!r}，请改用该 MJCF 中 `<body name=...>` 的名称（常见如 pelvis、torso_link）。"
            )
        if args.camera_follow_azimuth is not None:
            az_note = f"azimuth={args.camera_follow_azimuth:g}°（手动指定）"
        elif follow_side == "right":
            az_note = "方位角≈-90°（机体右侧近似）"
        elif follow_side == "left":
            az_note = "方位角≈+90°（机体左侧近似）"
        else:
            az_note = "方位角（默认侧向）"

        print(
            f"[相机跟踪] TRACKING(body={args.camera_track_body}), 侧={follow_side}, "
            f"distance={args.camera_follow_distance:g}, elevation={args.camera_follow_elevation:g}, {az_note}"
        )

    # load policy
    policy = load_policy(policy_path, "primary")
    lower_body_policy = load_policy(lower_body_policy_path, "lower-body") if lower_body_policy_path else None
    policy_mode = "lower+upper/full composite" if lower_body_policy is not None else "single"
    print(
        f"Policy mode: {policy_mode}, robot_actions={num_actions}, "
        f"policy_actions={policy_num_actions}, policy_obs={policy_num_obs}"
    )
    keyboard_controller.start_keyboard_listener()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while (
            viewer.is_running()
            and keyboard_controller.running
            and time.time() - start < simulation_duration
        ):
            step_start = time.time()
            tau = pd_control(
                target_dof_pos,
                d.qpos[qpos_indices],
                kps,
                np.zeros_like(kds),
                d.qvel[qvel_indices],
                kds,
            )
            if clip_torques:
                tau = np.clip(tau, -torque_limits, torque_limits)
            d.ctrl[actuator_indices] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                cmd = keyboard_controller.get_vel_cmd()

                # create observation
                qj = d.qpos[qpos_indices]
                dqj = d.qvel[qvel_indices]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period

                obs = build_policy_observation(
                    policy_num_actions,
                    policy_num_obs,
                    qj,
                    dqj,
                    quat,
                    omega,
                    cmd,
                    policy_action,
                    default_angles,
                    obs_scales,
                    phase,
                )
                policy_action = run_policy(policy, obs)
                raw_policy_action = policy_action.copy()

                if lower_body_policy is None:
                    action[:] = 0.0
                    if policy_action.shape[0] > num_actions:
                        raise ValueError(f"Policy output has {policy_action.shape[0]} actions, robot expects {num_actions}")
                    action[: policy_action.shape[0]] = policy_action
                else:
                    lower_obs = build_policy_observation(
                        lower_body_num_actions,
                        lower_body_num_obs,
                        qj,
                        dqj,
                        quat,
                        omega,
                        cmd,
                        lower_body_action,
                        default_angles,
                        obs_scales,
                        phase,
                    )
                    lower_body_action = run_policy(lower_body_policy, lower_obs)
                    action[:] = 0.0
                    action[:lower_body_num_actions] = lower_body_action
                    if policy_action.shape[0] == num_actions:
                        action[lower_body_num_actions:] = (
                            policy_action[lower_body_num_actions:] * upper_body_action_scale
                        )
                    elif policy_action.shape[0] == num_actions - lower_body_num_actions:
                        action[lower_body_num_actions:] = policy_action * upper_body_action_scale
                    else:
                        raise ValueError(
                            f"Primary policy output has {policy_action.shape[0]} actions, "
                            f"expected {num_actions} or {num_actions - lower_body_num_actions}"
                        )
                if policy_num_actions == num_actions:
                    policy_action = action.copy()
                if clip_actions is not None:
                    action = np.clip(action, -clip_actions, clip_actions)
                    if lower_body_policy is not None:
                        lower_body_action = action[:lower_body_num_actions].copy()
                    if policy_num_actions == num_actions:
                        policy_action = action.copy()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles
                if not printed_policy_debug:
                    print(
                        "First policy step: "
                        f"obs[min={obs.min():.3f}, max={obs.max():.3f}], "
                        f"raw_action[min={raw_policy_action.min():.3f}, max={raw_policy_action.max():.3f}], "
                        f"applied_action[min={action.min():.3f}, max={action.max():.3f}], "
                        f"target_delta[min={(target_dof_pos - default_angles).min():.3f}, "
                        f"max={(target_dof_pos - default_angles).max():.3f}], "
                        f"qpos_error_norm={np.linalg.norm(qj[:num_actions] - default_angles):.3f}, "
                        f"torque_limit[min={torque_limits.min():.1f}, max={torque_limits.max():.1f}]"
                    )
                    printed_policy_debug = True

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            if cam_track_body_id is not None:
                _apply_tracking_side_camera(
                    viewer,
                    cam_track_body_id,
                    follow_side,
                    args.camera_follow_distance,
                    args.camera_follow_elevation,
                    args.camera_follow_azimuth,
                )
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("程序退出")
