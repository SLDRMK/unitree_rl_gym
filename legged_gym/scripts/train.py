import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    n_learn = int(train_cfg.runner.max_iterations)
    if getattr(args, "train_to_iteration", None) is not None:
        tgt = int(args.train_to_iteration)
        cur = int(getattr(ppo_runner, "current_learning_iteration", 0))
        n_learn = max(0, tgt - cur)
        print(
            f"[train] Train-to-global-iteration {tgt}: "
            f"{n_learn} iterations remaining (counter at {cur})."
        )
        if n_learn == 0:
            print("[train] Already at target iteration; exiting without further training.")
            return

    ppo_runner.learn(num_learning_iterations=n_learn, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
