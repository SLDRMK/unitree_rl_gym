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

**Example — CMU Subject 07 SMPL poses** · source [`pics/07_01_poses_render_rot0_15fps.gif`](pics/07_01_poses_render_rot0_15fps.gif)

<img src="pics/07_01_poses_render_rot0_15fps.gif" width="720" alt="CMU Subject 07 SMPL poses — rendered preview loop">

## 🔄 Motion retargeting (two pipelines)

Expert trajectories for `g1_upper_amp` / `g1_upper_motion_ref` must be **retargeted** from human motion to the Unitree G1 model. We maintain two related codebases; both implement a **shared walking subset filter** (name rules + motion statistics) so training sees comparable clip collections.

### 1) [SLDRMK/AMASS-POST-PROCESS](https://github.com/SLDRMK/AMASS-POST-PROCESS)

- **Fork of** [TeleHuman/PBHC](https://github.com/TeleHuman/PBHC) (KungfuBot / PBHC motion processing lineage).
- **Core code:** [`smpl_retarget/`](https://github.com/SLDRMK/AMASS-POST-PROCESS/tree/main/smpl_retarget) — **Mink** differential IK retargeting (`mink_retarget/convert_fit_motion.py`), building on [Mink](https://github.com/kevinzakka/mink) and ideas from MaskedMimic / PHC-style stacks.
- **Improvements in this fork (summary):** refactored **relative positional** `FrameTask`s from SMPL parent-local bone directions; **`TorsoUprightTask`** (penalizes lateral tilt of torso / pelvis / head local +Z); retuned costs (`ROOT_*`, `RELATIVE_*`, `POSTURE_SCALE`, `TORSO_UPRIGHT_SCALE`, etc.). Per-clip aggregates are written under `smpl_retarget/retargeted_motion_data/mink_adjust/<stem>.pkl` when using the shipped layout (`walking_candidates.jsonl` references `mink_aggregate_*`).
- **Motivation:** the stock PBHC-style Mink stack could produce **pigeon‑toed legs**, a **hunched / forward‑leaning torso**, and **arms clamped too close to the body** when SMPL global targets were chased too aggressively without enough relative-bone and upright regularization. The tasks and weights above target those failure modes.

**Example — Mink IK retarget (this fork)** · source [`pics/mink_retarget_rot0_15fps.gif`](pics/mink_retarget_rot0_15fps.gif)

<img src="pics/mink_retarget_rot0_15fps.gif" width="720" alt="Mink IK retarget on G1 — preview loop">

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

**Example — GMR retarget** · source [`pics/gmr_retarget_rot0_15fps.gif`](pics/gmr_retarget_rot0_15fps.gif)

<img src="pics/gmr_retarget_rot0_15fps.gif" width="720" alt="GMR retarget on G1 — preview loop">

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

```bash
python legged_gym/scripts/train.py --task=xxx
```

#### ⚙️ Common CLI flags

| Topic | Flags / behaviour |
|--------|-------------------|
| Task | **`--task`** — e.g. `go2`, `g1`, `h1`, `h1_2`. For Unitree **G1 23‑DoF**: **`g1_upper_amp`** (discriminator imitation, **recommended** main line here), **`g1_upper`** (staged baseline PPO), **`g1_upper_motion_ref`** (dense per‑step shaping; legacy / not recommended). |
| Rendering & devices | **`--headless`**, **`--sim_device`**, **`--rl_device`**, **`--num_envs`**, **`--seed`** |
| How long to train | **`--max_iterations`** (→ `learn(num_learning_iterations)`) executes that many optimisation passes **starting from restored** `current_learning_iteration` (**additive on resume**, not usually a fixed global ceiling). Prefer **`--train_to_iteration`** to stop at an absolute **`iter`** value — remaining \(\approx\) target **`−`** restored `iter`. |
| Checkpoints | **`--resume`**, **`--load_run`**, **`--checkpoint`** — any loader path restores **weights and** `current_learning_iteration` (**required** for \(\lambda_{\mathrm{amp}}\) milestones to align). |
| G1 staging | **`--training_stage`** = `upper_body` \| `joint_finetune`; **`--lower_body_checkpoint`**: training **`model_*.pt`** for 12‑DoF **`g1`**, never the exported TorchScript policy. |
| TensorBoard folders | **`--resume_fork`** (CLI) ≡ one‑shot **`runner.resume_continue_logdir = False`** in config: **new** dated run folder while **still** loading checkpoint; **`resume_continue_logdir`** does **not** turn loading on/off—only **reuse vs new** directory. |

**Default save layout**: **`logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`** (checkpoint includes **`iter`**; AMP checkpoints also bundle **`discriminator_state_dict`** and optimizer state).

#### G1 staged PPO (**without** AMP prior — baselines)

**`g1`** → **`g1_upper` + `upper_body`** (lower body frozen to a **`g1` checkpoint**) → **`g1_upper` + `joint_finetune`** full‑body refinement. Helper scripts live in **`legged_gym/scripts/`** (`train_g1_upper_stage2.sh`, `train_g1_fullbody_isaaclab.sh`, …).

#### G1: motion‑reference imitation (legacy — not recommended)

Task **`g1_upper_motion_ref`**: dense **per‑step** reference shaping (**`motion_ref_dof`**) toward pickle joint trajectories. We **did not** get results comparable to **`g1_upper_amp`** here; retained for reproducibility only. Logs: **`logs/g1_upper_motion_ref/`**.

```bash
export MOTION_REF_DATA_DIR=/path/to/mink_or_gmr_pickles   # recursive `**/glob_pattern.pkl`
bash legged_gym/scripts/train_g1_upper_motion_ref.sh
```

If **`motion_ref.data_dir`** is empty in Python config you **must** set **`MOTION_REF_DATA_DIR`** (or training fails at env init).

**Tune** (**`G1UpperBodyMotionRefCfg`**, **`motion_ref`** / rewards in [`g1_config.py`](legged_gym/envs/g1/g1_config.py)): **`motion_ref_dof`**, **`motion_ref.err_reduce`** (`mean` vs `sum`), **`command_gate`**, **`motion_ref.sigma` / `sigma_min`** — σ is rad scale on the \(\|q−q_{\mathrm{ref}}\|_2\) vector magnitude; **`curriculum_norm_ema_alpha`**, etc.

---

#### G1: AMP‑style imitation (**recommended**)

Task **`g1_upper_amp`** uses a **binary discriminator** on a short **multi‑frame** window of \(\{\, q,\dot q \,\}\) (relative default pose × env scaling) and adds **`r_amp = -\log\left(\mathrm{clamp}(1 - \sigma(\mathrm{D}(x)), \varepsilon)\right)`** weighted by \(\lambda_{\mathrm{amp}}\) to the PPO return. **`motion_ref_dof` = 0** — **no** dense joint‑tracking bonus.

Experiment directory: **`logs/g1_upper_amp/`**.

##### Quick start

```bash
# Example default in repo script: filtered CMU-only GMR output (replace with YOUR folder)
export MOTION_REF_DATA_DIR=/path/to/your_pickles
bash legged_gym/scripts/train_g1_upper_amp.sh
```

Equivalent: **`python legged_gym/scripts/train.py --task=g1_upper_amp --training_stage=joint_finetune ...`**. Overrides: env **`NUM_ENVS`**, **`MAX_ITERATIONS`**, **`RUN_NAME`**, **`MOTION_REF_DATA_DIR`** ([`train_g1_upper_amp.sh`](legged_gym/scripts/train_g1_upper_amp.sh)).

##### Expert clips & empirical data choice

- **Loader layout**: **`MOTION_REF_DATA_DIR`** (overrides empty **`motion_ref.data_dir`**) pointing at a directory of **`*.pkl`** clips; loader uses **`motion_ref.glob_pattern`** (default **`"*.pkl"`**) recursively ([`build_mink_motion_bank`](legged_gym/utils/mink_reference_motion.py)).

- **Lesson learned**: pooling **three** AMASS-derived corpora (**CMU**, **BMLrub**, **KIT**) in one bank produced **mixed stylistic regimes** under a single global discriminator—we observed **distribution conflict**. We therefore **narrowed AMP training to the filtered CMU subset only** (same subject rules as **`Data:` / Motion retargeting**). The **`train_g1_upper_amp.sh` default illustrates a CMU‑subset layout** (`robot_cmu_subset`); swap to wherever your pickles live.

##### How experts are sampled (discriminator minibatches)

| Mechanism | Implementation / parameters |
|-----------|------------------------------|
| Which clip | **`MinkReferenceMotionBank.sample_clip_indices`** — **uniform random** clip index \(\in \{0,N_{\mathrm{clips}}-1\}\). Limit files with **`motion_ref.clip_limit`** (truncate sorted path list — see [`build_mink_motion_bank`](legged_gym/utils/mink_reference_motion.py)). |
| Which phase inside a clip | **`sample_phase_times`** — continuous **uniform** over each clip duration; loops with linear interp at **`policy_dt`**. |

##### \(\lambda_{\mathrm{amp}}\) curriculum (reward mixing)

Controlled by **`G1UpperBodyAmpCfg.amp`**. Default **`curriculum_enabled = True`**, **`curriculum_interp_between_milestones = False`** → \(\lambda_{\mathrm{amp}}\) is **piecewise constant** (**no** interpolation between breakpoints).

| Milestone (`learning_iteration` ≥) | \(\lambda_{\mathrm{amp}}\) |
|:---:|:---:|
| 0 | 0.000 |
| 2000 | 0.035 |
| 6000 | 0.070 |
| 12000 | 0.100 |
| 18000 | 0.150 |

Early phase keeps **`λ = 0`**: discriminator **skipped** (**`min_scale_for_amp_disc = 0.0`**, i.e. no forward/backprop when \(\lambda\le0\)). Fallback constant **`reward_scale = 0.25`** applies when **`curriculum_enabled`** is **`False`**. Trainer default **`runner.max_iterations = 25000`** (\(\lambda\) stays at **0.15** from iter **18000** onward unless you lengthen or edit schedule).

##### Discriminator optimisation (architecture & LR)

Configured in **`G1UpperBodyAmpCfg.amp`** (mirror under **`train_cfg['amp']`**). Defaults deliberately **lighter** than a “full‑width” imitation stack—the heavy MLP discriminator overfit quickly and destabilised early training.

| Item | Default (this repo) | Notes |
|------|--------------------|-------|
| **`hidden_dims`** | **`[128, 128]`** | Comment block shows former **`[512, 256]`** for reference. |
| **`activation`** | `elu` | |
| **`disc_learning_rate`** | **`1e-5`** | Comment block keeps legacy **`3e-4`** for comparison. |
| **`label_smoothing`** | **`0.1`** | Targets **fake → `ls`**, **expert → `1−ls`** (`BCEWithLogits`). |
| **`num_updates_per_iteration`** | **`1`** | Full **`disc_minibatches`** passes per rollout update iteration. |
| **`disc_minibatches`** | `4` | |
| **`disc_grad_norm`** | `1.0` | clipping |
| **`disc_weight_decay`** | `0` | |

##### Extra stabilisers (optional but default‑on here)

| Item | Default |
|------|---------|
| **`disc_stop_train_accuracy_above`** | **`0.85`** — pause discriminator minibatches whose **balanced** hard accuracy exceeds the threshold (**mitigate D collapsing / overfitting**) |
| **`fake_amp_pool_capacity_rows`** | **`-1`** — auto \(\max(8192, 8 \times\text{steps}\times\text{envs})\) policy‑feature rows |
| **`fake_pool_overflow_resample`**, **`fake_pool_mix_fraction`** | **`True`**, **`0.5`** — minibatch negatives mix **replay pool** ↔ **fresh rollout** |
| **`train_feature_mask_prob`** | **`0.1`** — dropout‑style masking on **training‑time** discriminator inputs |
| Temporal stack | **`history_frames = 10`**, **`history_window_s = 0.9`** — samples **uniformly spaced in time** spanning the last **`history_window_s`** seconds up to anchor time (see **`gather_amp_dof_features`**). |

##### Logging & reproducibility

- TensorBoard (**readouts** include **`AMP/lambda_amp`**, **`AMP/mean_step_amp_scaled`**, discriminator accuracies vs skip counts, **`Timing/learn_disc_amp`**, …).  
- **Resume semantics**: restored **`iter`** must match \(\lambda_{\mathrm{amp}}\) schedule milestones; **`--train_to_iteration`** or edited **`reward_scale_schedule_iters`** avoids “silent” mismatches vs wall‑clock checkpoints.

_All numeric defaults above refer to [`legged_gym/envs/g1/g1_config.py`](legged_gym/envs/g1/g1_config.py) (`G1UpperBodyAmpCfg`, `G1UpperBodyAmpCfgPPO`)._ 

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

