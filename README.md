<div align="center">
  <h1 align="center">Unitree RL GYM</h1>
  <p align="center">
    <span> 🌎English </span> | <a href="README_zh.md"> 🇨🇳中文 </a>
  </p>
</div>

<p align="center">
  <strong>Unitree-focused RL stack for Go2 / H1 / H1_2 / G1.</strong><br/>
  In this fork, the <strong>main line</strong> is full-body Unitree G1 locomotion trained with <strong>AMP-style</strong> (discriminator-shaped) imitation so the gait stays close to human walking from retargeted AMASS clips. The original <strong>standard PPO</strong> tasks (<code>g1</code>, <code>g1_upper</code> without motion prior) remain as <strong>baselines</strong>. The alternate task <strong><code>g1_upper_motion_ref</code></strong> (dense per-step reference-joint shaping) was kept only as documentation of an <strong>unsuccessful</strong> approach for our setup—not the recommended path.
</p>

## 📦 Installation and Configuration

Please refer to [setup.md](/doc/setup_en.md) for installation and configuration steps.

## 🎯 Data: AMASS walking subsets

We build expert motion from [**AMASS**](https://amass.is.tue.mpg.de/) (registration required on the [official download page](https://amass.is.tue.mpg.de/download.php)). For this project we use the **BMLrub**, **CMU**, and **KIT** subsets only (see walking filters below). Obey AMASS **license and citation** when redistributing or publishing.

**Example — CMU Subject 07 SMPL poses** · source [`pics/07_01_poses_render.mp4`](pics/07_01_poses_render.mp4)

<video src="pics/07_01_poses_render.mp4" controls muted playsinline width="720"></video>

## 🔄 Motion retargeting (two pipelines)

Expert trajectories for `g1_upper_amp` / `g1_upper_motion_ref` must be **retargeted** from human motion to the Unitree G1 model. We maintain two related codebases; both implement a **shared walking subset filter** (name rules + motion statistics) so training sees comparable clip collections.

### 1) [SLDRMK/AMASS-POST-PROCESS](https://github.com/SLDRMK/AMASS-POST-PROCESS)

- **Fork of** [TeleHuman/PBHC](https://github.com/TeleHuman/PBHC) (KungfuBot / PBHC motion processing lineage).
- **Core code:** [`smpl_retarget/`](https://github.com/SLDRMK/AMASS-POST-PROCESS/tree/main/smpl_retarget) — **Mink** differential IK retargeting (`mink_retarget/convert_fit_motion.py`), building on [Mink](https://github.com/kevinzakka/mink) and ideas from MaskedMimic / PHC-style stacks.
- **Improvements in this fork (summary):** refactored **relative positional** `FrameTask`s from SMPL parent-local bone directions; **`TorsoUprightTask`** (penalizes lateral tilt of torso / pelvis / head local +Z); retuned costs (`ROOT_*`, `RELATIVE_*`, `POSTURE_SCALE`, `TORSO_UPRIGHT_SCALE`, etc.). Per-clip aggregates are written under `smpl_retarget/retargeted_motion_data/mink_adjust/<stem>.pkl` when using the shipped layout (`walking_candidates.jsonl` references `mink_aggregate_*`).
- **Motivation:** the stock PBHC-style Mink stack could produce **pigeon‑toed legs**, a **hunched / forward‑leaning torso**, and **arms clamped too close to the body** when SMPL global targets were chased too aggressively without enough relative-bone and upright regularization. The tasks and weights above target those failure modes.

**Example — Mink IK retarget (this fork)** · source [`pics/mink_retarget.webm`](pics/mink_retarget.webm)

<video src="pics/mink_retarget.webm" controls muted playsinline width="720"></video>

**Walking / dataset filter (`convert_fit_motion.py`, `--filter-walking` / `--filter-only`):**

- **Skipped directories:** folder name contains `retarget`, `smpl`, or `h1`.
- **Scanned files:** recursive `**/*.npz`, `**/*.pkl`; always skip `shape.npz` and `*stagei.npz`-style names per README rules.
- **Name gate (datasets enabled for walking):**
  - **CMU** — subjects **`07`, `08`, `17`, `35`, `39`** only.
  - **KIT** — basename must contain `walk`.
  - **BMLrub / BioMotionLab_NTroje** — basename must match `*_normal_walk*_poses.npz` (case-insensitive glob).
  - Other dataset folders → rejected by policy.
- **Motion gate:** mean root translation speed ∈ **[0.3, 2.0] m/s** (Typer defaults); FFT on dominant horizontal displacement: dominant frequency ∈ **0.4–3.0 Hz** and spectral peak ≥ **3×** median in band.

Full detail: [`AMASS-POST-PROCESS/smpl_retarget/README.md`](https://github.com/SLDRMK/AMASS-POST-PROCESS/blob/main/smpl_retarget/README.md) (local copy if you clone: `AMASS-POST-PROCESS/smpl_retarget/README.md`).

### 2) [SLDRMK/GMR](https://github.com/SLDRMK/GMR)

- **Fork of** [YanjieZe/GMR](https://github.com/YanjieZe/GMR) (General Motion Retargeting).
- **Why GMR often looks better (brief):** GMR formulates retargeting as a **whole-body** problem with an objective stack tuned for **stable, RL-friendly** robot motion (including default **joint velocity limits** and regularization that discourages joint locking and collapsed postures). That tends to avoid the **local minima** of “match every SMPL target at all costs,” which can still show up as **inward knees**, **rounded back**, or **over-adducted arms** when the per-frame task balance is wrong. In our experience on the same filtered AMASS walking clips, GMR’s solution is **smoother and more natural** for G1.

**Example — GMR retarget** · source [`pics/gmr_retarget.webm`](pics/gmr_retarget.webm)

<video src="pics/gmr_retarget.webm" controls muted playsinline width="720"></video>

- **Batch SMPL-X → robot:** `scripts/smplx_to_robot_dataset.py`. Enable **`--filter_walking`** with `--src_folder` pointing at an AMASS-style root containing **`CMU/`**, **`KIT/`**, **`BMLrub/`**, etc.
- **Name filter (aligned with `convert_fit_motion.py` intent):**

| Dataset folder | Rule |
|----------------|------|
| **CMU** | Subjects **07, 08, 17, 35, 39** |
| **BMLrub** / BioMotionLab_NTroje | `*_normal_walk*_poses.npz` or `*_normal_walk*_stageii.npz` |
| **KIT** (folder name contains `kit`) | Filename contains **`walk`** |

- **Motion gate (after name pass):** same speed band **0.3–2.0 m/s** and FFT periodicity check (**0.4–3 Hz**, peak/median ≥ **3**), documented as aligned with AMASS-POST-PROCESS.

- **Extra batch exclusions (always on in that fork):** basenames in `assets/hard_motions/*.txt` lists, or motion name substring matches (`BMLrub`, `EKUT`, `crawl`, `_lie`, `upstairs`, `downstairs`, …) removed before processing.

See upstream docs: [`GMR` README § Retargeting from SMPL-X](https://github.com/SLDRMK/GMR/blob/main/README.md) (walking filter §).

Point **`MOTION_REF_DATA_DIR`** in this repo at the folder that contains your **`*.pkl`** clips (layout expected by [`legged_gym.utils.mink_reference_motion`](legged_gym/utils/mink_reference_motion.py)).

## 🔁 Process Overview

The basic workflow for using reinforcement learning to achieve motion control is:

`Train` → `Play` → `Sim2Sim` → `Sim2Real`

- **Train**: Use the Gym simulation environment to let the robot interact with the environment and find a policy that maximizes the designed rewards. Real-time visualization during training is not recommended to avoid reduced efficiency.
- **Play**: Use the Play command to verify the trained policy and ensure it meets expectations.
- **Sim2Sim**: Deploy the Gym-trained policy to other simulators to ensure it’s not overly specific to Gym characteristics.
- **Sim2Real**: Deploy the policy to a physical robot to achieve motion control.

For **G1 human-like walking**, prioritize **`g1_upper_amp`** after preparing clips with the **Data / retargeting** section; classic **`g1` / `g1_upper`** runs serve as **baselines** without this motion prior.

## 🛠️ User Guide

### 1. Training

Run the following command to start training:

```bash
python legged_gym/scripts/train.py --task=xxx
```

#### ⚙️ Parameter Description
- `--task`: Required parameter; values include `go2`, `g1`, `h1`, `h1_2`, plus G1 23DoF variants: **`g1_upper_amp`** (motion prior, main line here), **`g1_upper`** (baseline PPO staging), **`g1_upper_motion_ref`** (dense reference shaping; legacy / not recommended—see Motion retargeting section for data sources).

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
- `--training_stage`: G1 23-DoF staged training (`upper_body` or `joint_finetune`; see scripts under `legged_gym/scripts/`).
- `--lower_body_checkpoint`: For `upper_body`, path to the 12-DoF lower-body **`model_*.pt`** checkpoint (not the exported TorchScript policy).
- `--resume_fork`: When loading a checkpoint, create a **new** timestamped run folder for TensorBoard and saves instead of staying in the checkpoint’s folder; the **iteration counter still restores from the checkpoint** (same semantics as configuring `runner.resume_continue_logdir = False` for that run).
- `--train_to_iteration`: Train until a **global** iteration counter (computes remaining steps as `target − restored_iter`; overrides additive `--max_iterations` for this launch).

**Checkpoint resume**:

- Passing **`--resume`** loads weights **and restores** `current_learning_iteration` (used by curricula such as \(\lambda_{\mathrm{amp}}\)).  
- Passing **`--checkpoint`** and/or **`--load_run`** on the CLI also **enables resume** (same loader path).
- **`resume_continue_logdir`** in Python config (**`legged_robot_config.py`** `runner`): only selects **reuse vs new log directory**—it does **not** toggle loading.

**Default Training Result Directory**: `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`

#### G1: motion reference imitation (legacy; not recommended)

Optional task **`g1_upper_motion_ref`**: dense **per-step** joint reference shaping toward mink pickles. **We did not get satisfactory results with this trajectory** compared to **`g1_upper_amp`**; the task remains in-tree for reproducibility only. Logs: **`logs/g1_upper_motion_ref/`**.

**Train** (from the repository root):

```bash
export MOTION_REF_DATA_DIR=/path/to/mink/pickles   # directory of retargeted *.pkl clips

bash legged_gym/scripts/train_g1_upper_motion_ref.sh
```

You can also set `NUM_ENVS`, `MAX_ITERATIONS`, `RUN_NAME`, or call `train.py` with `--task=g1_upper_motion_ref --training_stage=joint_finetune` and the same environment variable. If `motion_ref.data_dir` is empty in config, **`MOTION_REF_DATA_DIR` must be set** or the environment will raise at startup.

**Tune** (see `legged_gym/envs/g1/g1_config.py`, class `G1UpperBodyMotionRefCfg`): `motion_ref_dof` scale, `motion_ref.err_reduce` (`mean` vs `sum`), `command_gate`, **`motion_ref.sigma` / `sigma_min` (L² norm scale in rad for joint error vector, curriculum updates `σ ← max(σ_min, min(mean ‖q−q_ref‖₂, σ))`)**, `curriculum_norm_ema_alpha`, etc.

#### G1: AMP-style motion prior (**main imitation path**)

Task **`g1_upper_amp`** is the **preferred** setup here: **AMP-style discriminator** (no dense **`motion_ref_dof`** tracking reward). Experts are walking clips produced by either retarget fork above (**`Motion retargeting`** section).

- Experiment folder: **`logs/g1_upper_amp/`**
- Train script (repo root):

```bash
export MOTION_REF_DATA_DIR=/path/to/mink/pickles

bash legged_gym/scripts/train_g1_upper_amp.sh
```

Equivalent: `python legged_gym/scripts/train.py --task=g1_upper_amp --training_stage=joint_finetune ...`. Same **`MOTION_REF_DATA_DIR`** (or filled `motion_ref.data_dir`) as `g1_upper_motion_ref` so clips load.

Tune **`G1UpperBodyAmpCfg` / nested `amp`** (and mirrored `train_cfg`): **`curriculum_enabled`**, **`reward_scale_schedule_iters`** (list of `(learning_iteration, λ_amp)` milestones; see `legged_gym/envs/g1/g1_config.py` for defaults), **`curriculum_interp_between_milestones`** (linear ramp between milestones), **`min_scale_for_amp_disc`**, **`history_frames`**, **`history_window_s`** (seconds over which uniformly spaced AMP history frames are taken), **`hidden_dims`**, **`label_smoothing`**, constant fallback **`reward_scale`** when curriculum is off, `disc_learning_rate`, `num_updates_per_iteration`, etc. **`G1UpperBodyAmpCfgPPO.runner.max_iterations`** is **25000** by default (override with `--max_iterations`). Checkpoints **`model_*.pt` also contain discriminator weights** (`discriminator_state_dict`) and the stored **`iter`** field for resumed curricula. TensorBoard: **`AMP/lambda_amp`**, **`AMP/mean_step_amp_scaled`**, **`Timing/*`** splits where applicable.

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

#### Camera (optional)

The passive viewer can **lock a side-follow camera** in MuJoCo **tracking** mode (spherical offset in the tracked body frame, so it follows translation and yaw). Example:

```bash
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml \
  --camera-follow-side right \
  --camera-follow-distance 2.8 \
  --camera-track-body pelvis
```

Options: **`--camera-follow-side`** `none`|`right`|`left`, **`--camera-follow-distance`**, **`--camera-track-body`** (MJCF `<body name=...>`, default `pelvis`), **`--camera-follow-elevation`**, optional **`--camera-follow-azimuth`** to flip/tune framing.

#### ➡️ Replace Network Model

The default model is located at `deploy/pre_train/{robot}/motion.pt`; custom-trained models are saved in `logs/g1/exported/policies/policy_lstm_1.pt`. Update the `policy_path` in the YAML configuration file accordingly.

### 4. Sim2Real (Physical Deployment)

Before deploying to the physical robot, ensure it’s in debug mode. Detailed steps can be found in the [Physical Deployment Guide](deploy/deploy_real/README.md):

```bash
python deploy/deploy_real/deploy_real.py {net_interface} {config_name}
```


#### Parameter Description
- `net_interface`: Network card name connected to the robot, e.g., `enp3s0`.
- `config_name`: Configuration file located in `deploy/deploy_real/configs/`, e.g., `g1.yaml`, `h1.yaml`, `h1_2.yaml`.

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
- [PBHC](https://github.com/TeleHuman/PBHC) / [AMASS-POST-PROCESS fork](https://github.com/SLDRMK/AMASS-POST-PROCESS): Mink SMPL→robot retargeting pipeline used to build motion data.
- [GMR](https://github.com/YanjieZe/GMR) / [SLDRMK fork](https://github.com/SLDRMK/GMR): Alternative SMPL-X batch retargeting with compatible walking filters.
- [AMASS](https://amass.is.tue.mpg.de/): Human motion corpus (BMLrub / CMU / KIT subsets used here).
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

