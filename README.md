<div align="center">
  <h1 align="center">Unitree RL GYM</h1>
  <p align="center">
    <span> 🌎English </span> | <a href="README_zh.md"> 🇨🇳中文 </a>
  </p>
</div>

<p align="center">
  <strong>This is a repository for reinforcement learning implementation based on Unitree robots, supporting Unitree Go2, H1, H1_2, and G1.</strong> 
</p>

<div align="center">

| <div align="center"> Isaac Gym </div> | <div align="center">  Mujoco </div> |  <div align="center"> Physical </div> |
|--- | --- | --- |
| [<img src="https://oss-global-cdn.unitree.com/static/32f06dc9dfe4452dac300dda45e86b34.GIF" width="240px">](https://oss-global-cdn.unitree.com/static/5bbc5ab1d551407080ca9d58d7bec1c8.mp4) | [<img src="https://oss-global-cdn.unitree.com/static/244cd5c4f823495fbfb67ef08f56aa33.GIF" width="240px">](https://oss-global-cdn.unitree.com/static/5aa48535ffd641e2932c0ba45c8e7854.mp4) | [<img src="https://oss-global-cdn.unitree.com/static/78c61459d3ab41448cfdb31f6a537e8b.GIF" width="240px">](https://oss-global-cdn.unitree.com/static/0818dcf7a6874b92997354d628adcacd.mp4) |

</div>

---

## 📦 Installation and Configuration

Please refer to [setup.md](/doc/setup_en.md) for installation and configuration steps.

## 🔁 Process Overview

The basic workflow for using reinforcement learning to achieve motion control is:

`Train` → `Play` → `Sim2Sim` → `Sim2Real`

- **Train**: Use the Gym simulation environment to let the robot interact with the environment and find a policy that maximizes the designed rewards. Real-time visualization during training is not recommended to avoid reduced efficiency.
- **Play**: Use the Play command to verify the trained policy and ensure it meets expectations.
- **Sim2Sim**: Deploy the Gym-trained policy to other simulators to ensure it’s not overly specific to Gym characteristics.
- **Sim2Real**: Deploy the policy to a physical robot to achieve motion control.

## 🛠️ User Guide

### 1. Training

Run the following command to start training:

```bash
python legged_gym/scripts/train.py --task=xxx
```

#### ⚙️ Parameter Description
- `--task`: Required parameter; values include `go2`, `g1`, `h1`, `h1_2`, plus G1 23DoF variants such as `g1_upper`, `g1_upper_motion_ref`, `g1_upper_amp` (see subsection below).

- `--headless`: Defaults to starting with a graphical interface; set to true for headless mode (higher efficiency).
- `--resume`: Resume training from a checkpoint in the logs.
- `--experiment_name`: Name of the experiment to run/load.
- `--run_name`: Name of the run to execute/load.
- `--load_run`: Name of the run to load; defaults to the latest run.
- `--checkpoint`: Checkpoint number to load; defaults to the latest file.
- `--num_envs`: Number of environments for parallel training.
- `--seed`: Random seed.
- `--max_iterations`: Maximum number of training iterations.
- `--sim_device`: Simulation computation device; specify CPU as `--sim_device=cpu`.
- `--rl_device`: Reinforcement learning computation device; specify CPU as `--rl_device=cpu`.

**Default Training Result Directory**: `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`

#### G1: motion reference imitation (optional)

This repo includes an optional task **`g1_upper_motion_ref`**: full-body PPO on the 23-DoF G1 with a **reference motion shaping** reward using mink-retargeted walking clips (pickle files from your pipeline, e.g. `*_poses.pkl`). It does not replace `g1_upper`; it uses experiment name `g1_upper_motion_ref` and logs under `logs/g1_upper_motion_ref/`.

**Train** (from the repository root):

```bash
export MOTION_REF_DATA_DIR=/path/to/mink/pickles   # directory of retargeted *.pkl clips

bash legged_gym/scripts/train_g1_upper_motion_ref.sh
```

You can also set `NUM_ENVS`, `MAX_ITERATIONS`, `RUN_NAME`, or call `train.py` with `--task=g1_upper_motion_ref --training_stage=joint_finetune` and the same environment variable. If `motion_ref.data_dir` is empty in config, **`MOTION_REF_DATA_DIR` must be set** or the environment will raise at startup.

**Tune** (see `legged_gym/envs/g1/g1_config.py`, class `G1UpperBodyMotionRefCfg`): `motion_ref_dof` scale, `motion_ref.err_reduce` (`mean` vs `sum`), `command_gate`, **`motion_ref.sigma` / `sigma_min` (L² norm scale in rad for joint error vector, curriculum updates `σ ← max(σ_min, min(mean ‖q−q_ref‖₂, σ))`)**, `curriculum_norm_ema_alpha`, etc.

#### G1: AMP-style motion prior (optional)

Task **`g1_upper_amp`** uses an **AMP-style discriminator** instead of dense per-step tracking of reference joint angles (**`motion_ref_dof` scale is 0** on this task): the discriminator classifies stacked **multi-frame joint features** (scaled relative `dof_pos` and `dof_vel`) from simulator rollouts vs expert trajectories sampled from the same mink pickles. Rewards add **λ_amp · (−log(clamp(1 − σ(D(x)), eps)))**; **λ_amp** is scheduled by default (see below). Discriminator training uses **binary cross-entropy with label smoothing**. When **λ_amp** is at or below **`min_scale_for_amp_disc`** (default 0), **no discriminator forward pass or update** runs for that iteration (pure PPO walking phase).

- Experiment folder: **`logs/g1_upper_amp/`**
- Train script (repo root):

```bash
export MOTION_REF_DATA_DIR=/path/to/mink/pickles

bash legged_gym/scripts/train_g1_upper_amp.sh
```

Equivalent: `python legged_gym/scripts/train.py --task=g1_upper_amp --training_stage=joint_finetune ...`. Same **`MOTION_REF_DATA_DIR`** (or filled `motion_ref.data_dir`) as `g1_upper_motion_ref` so clips load.

Tune **`G1UpperBodyAmpCfg` / nested `amp`** (and mirrored `train_cfg`): **`curriculum_enabled`**, **`reward_scale_schedule_iters`** (list of `(learning_iteration, λ_amp)` milestones; default follows a small-AMP → ramp → ~0.1–0.25 pattern for 10k iters), **`curriculum_interp_between_milestones`** (linear ramp between milestones), **`min_scale_for_amp_disc`**, plus `history_frames`, `hidden_dims`, `label_smoothing`, constant fallback **`reward_scale`** when curriculum is off, `disc_learning_rate`, `num_updates_per_iteration`, etc. Checkpoints **`model_*.pt` also contain discriminator weights** (`discriminator_state_dict`) for `--resume`. TensorBoard: **`AMP/lambda_amp`**, **`AMP/mean_step_amp_scaled`**.

### 2. Play

To visualize the training results in Gym, run the following command:

```bash
python legged_gym/scripts/play.py --task=xxx
```

**Description**:

- Play’s parameters are the same as Train’s.
- By default, it loads the latest model from the experiment folder’s last run.
- You can specify other models using `load_run` and `checkpoint`.

**G1 motion reference policy** (`g1_upper_motion_ref`): set **`MOTION_REF_DATA_DIR`** the same way as training so the env can load motion clips. Do not pass `--headless` if you want the Isaac Gym viewer.

```bash
export MOTION_REF_DATA_DIR=/path/to/mink/pickles

python legged_gym/scripts/play.py \
  --task=g1_upper_motion_ref \
  --training_stage=joint_finetune \
  --load_run=May13_12-16-09_g1_upper_motion_ref_mink
```

Optional: `--checkpoint=8000` or `--checkpoint=logs/g1_upper_motion_ref/<run>/model_8000.pt` to pick a specific checkpoint (default: latest `model_*.pt` in that run). Checkpoints live under `logs/g1_upper_motion_ref/<date>_<run_name>/`.

**G1 AMP policy** (`g1_upper_amp`): set **`MOTION_REF_DATA_DIR`** like training so the motion bank initializes. Inference only needs the Actor–Critic; the runner still constructs the discriminator when loading checkpoints that include discriminator weights.

```bash
export MOTION_REF_DATA_DIR=/path/to/mink/pickles

python legged_gym/scripts/play.py \
  --task=g1_upper_amp \
  --training_stage=joint_finetune \
  --load_run=<your_amp_run_folder_name>
```

Optional: `--checkpoint=<iter>` or a full path to `logs/g1_upper_amp/<date>_<run_name>/model_*.pt`.

#### 💾 Export Network

With the default **`EXPORT_POLICY = True`** flag in `legged_gym/scripts/play.py`, the Actor network is exported into the directory of the loaded checkpoint (same run folder as `model_*.pt`):

- Standard networks (MLP) are exported as `policy_1.pt`.
- RNN networks are exported as `policy_lstm_1.pt`.

### Play Results

| Go2 | G1 | H1 | H1_2 |
|--- | --- | --- | --- |
| [![go2](https://oss-global-cdn.unitree.com/static/ba006789e0af4fe3867255f507032cd7.GIF)](https://oss-global-cdn.unitree.com/static/d2e8da875473457c8d5d69c3de58b24d.mp4) | [![g1](https://oss-global-cdn.unitree.com/static/32f06dc9dfe4452dac300dda45e86b34.GIF)](https://oss-global-cdn.unitree.com/static/5bbc5ab1d551407080ca9d58d7bec1c8.mp4) | [![h1](https://oss-global-cdn.unitree.com/static/fa04e73966934efa9838e9c389f48fa2.GIF)](https://oss-global-cdn.unitree.com/static/522128f4640c4f348296d2761a33bf98.mp4) |[![h1_2](https://oss-global-cdn.unitree.com/static/83ed59ca0dab4a51906aff1f93428650.GIF)](https://oss-global-cdn.unitree.com/static/15fa46984f2343cb83342fd39f5ab7b2.mp4)|

---

### 3. Sim2Sim (Mujoco)

Run Sim2Sim in the Mujoco simulator:

```bash
python deploy/deploy_mujoco/deploy_mujoco.py {config_name}
```

#### Parameter Description
- `config_name`: Configuration file; default search path is `deploy/deploy_mujoco/configs/`.

#### Example: Running G1

```bash
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

#### ➡️ Replace Network Model

The default model is located at `deploy/pre_train/{robot}/motion.pt`; custom-trained models are saved in `logs/g1/exported/policies/policy_lstm_1.pt`. Update the `policy_path` in the YAML configuration file accordingly.

#### Simulation Results

| G1 | H1 | H1_2 |
|--- | --- | --- |
| [![mujoco_g1](https://oss-global-cdn.unitree.com/static/244cd5c4f823495fbfb67ef08f56aa33.GIF)](https://oss-global-cdn.unitree.com/static/5aa48535ffd641e2932c0ba45c8e7854.mp4)  |  [![mujoco_h1](https://oss-global-cdn.unitree.com/static/7ab4e8392e794e01b975efa205ef491e.GIF)](https://oss-global-cdn.unitree.com/static/8934052becd84d08bc8c18c95849cf32.mp4)  |  [![mujoco_h1_2](https://oss-global-cdn.unitree.com/static/2905e2fe9b3340159d749d5e0bc95cc4.GIF)](https://oss-global-cdn.unitree.com/static/ee7ee85bd6d249989a905c55c7a9d305.mp4) |


---

### 4. Sim2Real (Physical Deployment)

Before deploying to the physical robot, ensure it’s in debug mode. Detailed steps can be found in the [Physical Deployment Guide](deploy/deploy_real/README.md):

```bash
python deploy/deploy_real/deploy_real.py {net_interface} {config_name}
```


#### Parameter Description
- `net_interface`: Network card name connected to the robot, e.g., `enp3s0`.
- `config_name`: Configuration file located in `deploy/deploy_real/configs/`, e.g., `g1.yaml`, `h1.yaml`, `h1_2.yaml`.

#### Deployment Results

| G1 | H1 | H1_2 |
|--- | --- | --- |
| [![real_g1](https://oss-global-cdn.unitree.com/static/78c61459d3ab41448cfdb31f6a537e8b.GIF)](https://oss-global-cdn.unitree.com/static/0818dcf7a6874b92997354d628adcacd.mp4) | [![real_h1](https://oss-global-cdn.unitree.com/static/fa07b2fd2ad64bb08e6b624d39336245.GIF)](https://oss-global-cdn.unitree.com/static/ea0084038d384e3eaa73b961f33e6210.mp4) | [![real_h1_2](https://oss-global-cdn.unitree.com/static/a88915e3523546128a79520aa3e20979.GIF)](https://oss-global-cdn.unitree.com/static/12d041a7906e489fae79d55b091a63dd.mp4) |

---

#### Deploy with C++
There is also an example of deploying the G1 pre-trained model in C++. the C++ code is located in the following directory.

```
deploy/deploy_real/cpp_g1
```

First, navigate to the directory above.

```base
cd deploy/deploy_real/cpp_g1
```

The C++ implementation depends on the LibTorch library, download the LibTorch

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.7.1+cpu.zip
```

To build the project, executable the following steps

```bash
mkdir build
cd build
cmake ..
make -j4
```

After successful compilation, executate the program with:

```base
./g1_deploy_run {net_interface}
```

Replace `{net_interface}` with your actual network interface name (e.g., eth0, wlan0).

## 🎉 Acknowledgments

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [legged\_gym](https://github.com/leggedrobotics/legged_gym): The foundation for training and running codes.
- [rsl\_rl](https://github.com/leggedrobotics/rsl_rl.git): Reinforcement learning algorithm implementation.
- [mujoco](https://github.com/google-deepmind/mujoco.git): Providing powerful simulation functionalities.
- [unitree\_sdk2\_python](https://github.com/unitreerobotics/unitree_sdk2_python.git): Hardware communication interface for physical deployment.

---

## 🔖 License

This project is licensed under the [BSD 3-Clause License](./LICENSE):
1. The original copyright notice must be retained.
2. The project name or organization name may not be used for promotion.
3. Any modifications must be disclosed.

For details, please read the full [LICENSE file](./LICENSE).

