import os
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, get_load_path, task_registry


def get_loaded_run_dir(train_cfg):
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    load_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
    return os.path.dirname(load_path), load_path


def reset_inference_memory(actor_critic, dones):
    if not getattr(actor_critic, "is_recurrent", False):
        return
    hidden_states = actor_critic.memory_a.hidden_states
    if hidden_states is None:
        return
    actor_critic.memory_a.reset(dones)

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    actor_critic = ppo_runner.alg.actor_critic

    staged_cfg = getattr(env_cfg, "staged_training", None)
    stage = getattr(staged_cfg, "stage", "base")
    print(f"Playing task={args.task}, stage={stage}, num_actions={env.num_actions}, num_obs={env.num_obs}")
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path, load_path = get_loaded_run_dir(train_cfg)
        export_policy_as_jit(actor_critic, path)
        print('Loaded checkpoint from: ', load_path)
        print('Exported policy as jit script to run dir: ', path)

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        reset_inference_memory(actor_critic, dones)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
