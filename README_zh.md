<div align="center">
  <h1 align="center">Unitree RL GYM</h1>
  <p align="center">
    <a href="README.md">🌎 English</a> | <span>🇨🇳 中文</span>
  </p>
</div>

<p align="center">
  🎮🚪 <strong>面向 Unitree Go2、H1、H1_2、G1 的强化学习示例与训练代码。</strong><br/>
  本 fork 的<strong>主线</strong>是使用 <strong>AMP 式判别器塑形</strong> 做 G1 全身行走，使人形步态接近经重定向的人类行走数据；传统的 <strong>PPO 里程计/速度跟踪</strong>（<code>g1</code>、无运动先验的 <code>g1_upper</code> 等）保留为 <strong>baseline</strong>。<strong><code>g1_upper_motion_ref</code></strong>（逐步关节参考追踪塑形）在本仓库中标记为<strong>未走通的尝试</strong>，仅保留可复现入口，<strong>不推荐</strong>作为主要方案。
</p>

## 📦 安装配置

安装和配置步骤请参考 [setup.md](/doc/setup_zh.md)

## 🎯 数据：AMASS 行走子集

专家动作来自 [**AMASS**](https://amass.is.tue.mpg.de/)，需在 [官方下载页注册后获取](https://amass.is.tue.mpg.de/download.php)。本工作流仅使用 **BMLrub**、**CMU**、**KIT** 三个子集（与下文「行走筛选」一致）。分发或发表论文时请遵守 AMASS **许可与引用**要求。

**示例 — CMU Subject 07 SMPL 姿态渲染** · 源文件 [`pics/07_01_poses_render.mp4`](pics/07_01_poses_render.mp4)

<video src="pics/07_01_poses_render.mp4" controls muted playsinline width="720"></video>

## 🔄 动作重定向（两条管线）

`g1_upper_amp` / `g1_upper_motion_ref` 使用的专家轨迹需将人体动作重定向到 Unitree G1。我们维护两个关联仓库，并实现**一致的行走子集筛选**（名称规则 + 运动统计），保证训练侧 clip 集合可对齐。

### 1） [SLDRMK/AMASS-POST-PROCESS](https://github.com/SLDRMK/AMASS-POST-PROCESS)

- **Fork 自** [TeleHuman/PBHC](https://github.com/TeleHuman/PBHC)（KungfuBot / PBHC 运动处理链路）。
- **核心代码：** [`smpl_retarget/`](https://github.com/SLDRMK/AMASS-POST-PROCESS/tree/main/smpl_retarget)，基于 [Mink](https://github.com/kevinzakka/mink) 的微分 IK 重定向（`mink_retarget/convert_fit_motion.py`），融合了 MaskedMimic / PHC 思路。
- **本 fork 改进要点（摘要）：** 重写**相对位置** `FrameTask`（在 SMPL 父关节局部帧内取骨方向）；增加 **`TorsoUprightTask`**（约束 pelvis / torso / head 等单位竖直方向，抑制侧倾）；代价权重重调（`ROOT_*`、`RELATIVE_*`、`POSTURE_SCALE`、`TORSO_UPRIGHT_SCALE` 等）。成功重定向后的**整段聚合 pickle** 写入 `smpl_retarget/retargeted_motion_data/mink_adjust/<stem>.pkl`（与 `walking_candidates.jsonl` 中 `mink_aggregate_*` 字段一致）。
- **修改动机：** 原版 PBHC 系 Mink 栈在片面追求全局 SMPL 关键点贴合时，容易出现 **腿型内八（膝/踝内收）**、**躯干佝偻前倾**、**大臂夹着身体摆动不开**等次生姿态；上述相对骨长约束与躯干直立项即针对这些问题补强。

**示例 — Mink IK 重定向（本 fork）** · 源文件 [`pics/mink_retarget.webm`](pics/mink_retarget.webm)

<video src="pics/mink_retarget.webm" controls muted playsinline width="720"></video>

**行走 / 数据集筛选（`convert_fit_motion.py`，`--filter-walking` / `--filter-only`）：**

- **跳过目录名** 含 `retarget`、`smpl` 或 `h1` 的子目录。
- **扫描** `**/*.npz` 与 `**/*.pkl`；固定跳过 `shape.npz` 与含 `stagei` 等规则见上游 README。
- **名称门控（当前策略启用的数据集）：**
  - **CMU** — 仅被试 **`07`、`08`、`17`、`35`、`39`**；
  - **KIT** — 文件名须含 `walk`；
  - **BMLrub / BioMotionLab_NTroje** — 须匹配 `*_normal_walk*_poses.npz`（不区分大小写的 glob）；
  - 其它数据集目录 → 当前行走策略下**直接拒绝**。
- **运动门控：** 根节点平移平均速度 ∈ **[0.3, 2.0] m/s**（Typer 默认）；对水平向主导位移做 FFT，主频 ∈ **0.4–3.0 Hz** 且谱峰 ≥ 带内中位数的 **3 倍**。

完整说明见：[AMASS-POST-PROCESS `smpl_retarget/README.md`](https://github.com/SLDRMK/AMASS-POST-PROCESS/blob/main/smpl_retarget/README.md)（克隆后本地路径：`AMASS-POST-PROCESS/smpl_retarget/README.md`）。

### 2） [SLDRMK/GMR](https://github.com/SLDRMK/GMR)

- **Fork 自** [YanjieZe/GMR](https://github.com/YanjieZe/GMR)。
- **为何 GMR 往往效果更好（简述）：** GMR 将重定向视作**全身** IK/优化问题，目标与正则（含默认 **关节速度限幅**、防「锁在一侧」的过拟合惩罚等）偏向 **可对 RL 跟踪友好、物理上不过分扭曲**的解，而不是单帧无止境地「硬贴 SMPL」。这在同一套行走筛选数据上，通常能进一步减轻 **内八、含胸夹臂、局部姿态坍缩** 等局部极小。我们经验上 G1 的观感 **更自然、更顺滑**。

**示例 — GMR 重定向** · 源文件 [`pics/gmr_retarget.webm`](pics/gmr_retarget.webm)

<video src="pics/gmr_retarget.webm" controls muted playsinline width="720"></video>

- **批量 SMPL-X → 机器人：** `scripts/smplx_to_robot_dataset.py`，加 **`--filter_walking`**，`--src_folder` 指向 AMASS 式根目录（其下含 **`CMU/`**、**`KIT/`**、**`BMLrub/`** 等）。
- **名称筛选（与 `convert_fit_motion.py` 的设计意图对齐）：**

| 数据集目录 | 规则 |
|-----------|------|
| **CMU** | 被试 **07, 08, 17, 35, 39** |
| **BMLrub** / BioMotionLab_NTroje | `*_normal_walk*_poses.npz` 或 `*_normal_walk*_stageii.npz` |
| **KIT**（目录名含 `kit`） | 文件名须含 `walk` |

- **运动门控：** 通过名称筛选后，速度与 FFT 周期性与上表 AMASS-POST-PROCESS 一致（默认 **0.3–2.0 m/s**，**0.4–3 Hz**，峰/中位 ≥ **3**）。

- **额外批量排除（与该 fork 脚本内恒开逻辑一致）：** `assets/hard_motions/*.txt` 列出的难例 basename，或文件名子串含 `BMLrub`、`EKUT`、`crawl`、`_lie`、`upstairs`、`downstairs` 等，在批量处理前移除。

详见：[GMR README — SMPL-X 重定向与 `--filter_walking`](https://github.com/SLDRMK/GMR/blob/main/README.md)。

将本仓库训练/Play 所用的 **`MOTION_REF_DATA_DIR`** 指到包含重定向 **`*.pkl`** 的目录（需满足 [`legged_gym.utils.mink_reference_motion`](legged_gym/utils/mink_reference_motion.py) 的扫描约定）。

## 🔁 流程说明

强化学习实现运动控制的基本流程为：

`Train` → `Play` → `Sim2Sim` → `Sim2Real`

- **Train**: 通过 Gym 仿真环境，让机器人与环境互动，找到最满足奖励设计的策略。通常不推荐实时查看效果，以免降低训练效率。
- **Play**: 通过 Play 命令查看训练后的策略效果，确保策略符合预期。
- **Sim2Sim**: 将 Gym 训练完成的策略部署到其他仿真器，避免策略小众于 Gym 特性。
- **Sim2Real**: 将策略部署到实物机器人，实现运动控制。

若目标是 **类人行走步态**：数据与重定向见上文章节，训练主推 **`g1_upper_amp`**；传统 **`g1` / `g1_upper`** 仍可作为**无运动先验**对照 baseline。

## 🛠️ 使用指南

### 1. 训练

运行以下命令进行训练：

```bash
python legged_gym/scripts/train.py --task=xxx
```

#### ⚙️  参数说明
- `--task`: 必选参数；常用 `go2`, `g1`, `h1`, `h1_2`。G1 全身：**`g1_upper_amp`**（本文主线判别器 AMP）、**`g1_upper`**（分阶段 baseline）、**`g1_upper_motion_ref`**（遗留逐步参考塑形，不推荐）。
- `--headless`: 默认启动图形界面，设为 true 时不渲染图形界面（效率更高）
- `--resume`: 从日志中选择 checkpoint 继续训练
- `--experiment_name`: 运行/加载的 experiment 名称
- `--run_name`: 运行/加载的 run 名称
- `--load_run`: 加载运行的名称，默认加载最后一次运行
- `--checkpoint`: checkpoint 编号，默认加载最新一次文件
- `--num_envs`: 并行训练的环境个数
- `--seed`: 随机种子
- `--max_iterations`: 训练的最大迭代次数
- `--sim_device`: 仿真计算设备，指定 CPU 为 `--sim_device=cpu`
- `--rl_device`: 强化学习计算设备，指定 CPU 为 `--rl_device=cpu`
- `--training_stage`: G1 23DoF 分阶段训练模式，可选 `upper_body` 或 `joint_finetune`
- `--lower_body_checkpoint`: `upper_body` 阶段使用的 12DoF 下半身训练 checkpoint，例如 `logs/g1/xxx/model_10000.pt`
- `--resume_fork`：**读 checkpoint** 时仍恢复迭代计数，但 TensorBoard/保存写入**新的时间戳 run 子目录**（不写回 checkpoint 原目录）；与配置里单次将 `runner.resume_continue_logdir` 设为 `False` 的效果一致。
- `--train_to_iteration`: 训练到**全局迭代步**（本轮剩余迭代数约为 `目标值 − checkpoint 恢复的 iter`，会覆盖仅用 `--max_iterations` 相加的语义）。

**续跑与日志目录**

- **`--resume`**：加载权重并恢复 **`current_learning_iteration`**（用于 \(\lambda_{\mathrm{amp}}\) 等课程）。  
- 命令行若指定 **`--checkpoint`** 或 **`--load_run`**，也会**自动启用与 resume 相同的加载逻辑**。  
- 配置项 **`resume_continue_logdir`**（见 `legged_robot_config.py` 里 `runner`）**只决定**沿用 checkpoint 目录还是新建目录，**不负责**「是否加载模型」。

**默认保存训练结果**：`logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`

#### G1 分阶段训练

仓库额外提供了 G1 23DoF 训练流程：

- `g1`: 原始 12DoF 下半身策略。
- `g1_upper + upper_body`: 23DoF 机器人中，下半身由 12DoF checkpoint 控制，上半身策略训练。
- `g1_upper + joint_finetune`: 23DoF 全身联合训练或微调。

第二阶段训练上半身：

```bash
bash legged_gym/scripts/train_g1_upper_stage2.sh
```

也可以手动指定下半身 checkpoint：

```bash
LOWER_BODY_CHECKPOINT=logs/g1/Apr13_07-17-29_/model_10000.pt \
NUM_ENVS=4096 \
MAX_ITERATIONS=10000 \
RUN_NAME=stage2_upper_stable \
bash legged_gym/scripts/train_g1_upper_stage2.sh
```

全身 23DoF 训练：

```bash
bash legged_gym/scripts/train_g1_fullbody_isaaclab.sh
```

该脚本使用 `g1_upper` 任务和 `joint_finetune` 阶段，不加载下半身 checkpoint，策略直接输出 23 维动作。
当前示例日志目录为 `logs/g1_upper/May02_11-49-23_fullbody_isaaclab_randomized`。
全身阶段默认保持原版 G1 风格的域随机化、观测噪声和 PPO 探索噪声：摩擦随机化 `[0.1, 1.25]`、base mass 随机化 `[-1, 3]`、push 随机化开启、观测噪声开启、`init_noise_std=0.8`、`action_scale=0.25`。`joint_finetune` 阶段会使用随机关节初始位置 reset，避免策略只适应默认站姿。

#### G1 参考运动（遗留：逐步关节追踪，非推荐）

可选任务 **`g1_upper_motion_ref`**：对 mink 生成的轨迹做**逐步**关节参考塑形。在本项目设定下**效果不如 `g1_upper_amp`**，仅保留以便对照与复现。**不建议**作为主线方案。日志：`logs/g1_upper_motion_ref/`。

训练前必须指定动作数据目录（与训练脚本内默认可改）：

```bash
export MOTION_REF_DATA_DIR=/path/to/mink/pickles   # 例如 AMASS 管线下的 retargeted_motion_data/mink

bash legged_gym/scripts/train_g1_upper_motion_ref.sh
```

也可自行调用 `python legged_gym/scripts/train.py --task=g1_upper_motion_ref --training_stage=joint_finetune ...`。若配置里 `motion_ref.data_dir` 为空，则依赖环境变量 **`MOTION_REF_DATA_DIR`**，否则环境初始化会报错。

超参见 `legged_gym/envs/g1/g1_config.py` 中的 `G1UpperBodyMotionRefCfg`：`motion_ref_dof` 权重、`motion_ref.err_reduce`、**σ 为关节误差向量 L2 范数尺度（rad），课程按 `σ ← max(σ_min, min(batch均值‖q−q_ref‖₂, σ))` 更新，奖励为 `exp(−mse/σ²)`**、`curriculum_norm_ema_alpha` 等。

#### G1 AMP（判别器对抗式运动先验，**推荐的主线模仿**）

任务 **`g1_upper_amp`**：**推荐**使用判别器对齐专家关节分布，**不**使用稠密 **`motion_ref_dof`** 逐步追踪奖励。专家轨迹来自上文 **「动作重定向」** 任一管线产出的 pickle。

- **多帧扩张输入**：判别器输入为若干历史步上拼接的 **关节相对默认位姿的缩放 dof_pos、dof_vel**（与训练中观测缩放一致）。
- **策略回报**：在每步仿真回报上叠加 **\(\lambda_{\mathrm{amp}} \cdot (-\log(1 - D(\cdot)))\)**（概率来自 `sigmoid(logits)`，含 `clamp`）。**\(\lambda_{\mathrm{amp}}\)** 默认按 **learning iteration 分段课程** 逐渐加大（见下）；关闭课程时用常数 **`reward_scale`**。
- **判别器训练**：`BCEWithLogitsLoss`；专家样本目标 **`1 - label_smoothing`**，策略 rollout 样本目标 **`label_smoothing`**。当 **\(\lambda_{\mathrm{amp}} \le \texttt{min\_scale\_for\_amp\_disc}\)**（默认 0）时，该迭代 **不跑判别器前向与更新**（纯 PPO 行走阶段）。
- **实验目录**：`g1_upper_amp`，权重与日志位于 **`logs/g1_upper_amp/`**；checkpoint 内含 **判别器与其优化器**，便于 **`--resume`**。

训练脚本（仓库根目录）：

```bash
export MOTION_REF_DATA_DIR=/path/to/mink/pickles   # 与同目录参考运动档位相同

bash legged_gym/scripts/train_g1_upper_amp.sh
```

等价调用：`python legged_gym/scripts/train.py --task=g1_upper_amp --training_stage=joint_finetune ...`。同样需要 **`MOTION_REF_DATA_DIR`**（或配置里写明 `motion_ref.data_dir`）以加载 mink clip。

超参见 `legged_gym/envs/g1/g1_config.py` 中的 **`G1UpperBodyAmpCfg`** 及其嵌套 **`amp`**：**`curriculum_enabled`**、**`reward_scale_schedule_iters`**（`(迭代阈值, λ_amp)` 列表，默认值以配置文件为准）、**`curriculum_interp_between_milestones`**（里程碑间 λ 线性插值）、**`min_scale_for_amp_disc`**，以及 **`history_frames`**、**`history_window_s`**（多帧判别器在时间窗内均匀取样）、**`hidden_dims`**、**`label_smoothing`**、**`disc_learning_rate`** 等。**`G1UpperBodyAmpCfgPPO.runner.max_iterations`** 默认为 **25000**（可用 `--max_iterations` 覆盖）。checkpoint 内含判别器权重与 **`iter`**（用于恢复课程）。**`motion_ref_dof` = 0，不启用关节参考追踪塑形**。

---

### 2. Play

如果想要在 Gym 中查看训练效果，可以运行以下命令：

```bash
python legged_gym/scripts/play.py --task=xxx
```

**说明**：

- Play 启动参数与 Train 相同。
- 默认加载实验文件夹上次运行的最后一个模型。
- 可通过 `load_run` 和 `checkpoint` 指定其他模型。

#### 💾 导出网络

Play 会导出 Actor 网络，保存到当前加载 checkpoint 所在的 run 目录中。例如加载
`logs/g1_upper/May02_11-49-20_stage2_upper_stable/model_10000.pt` 时，会导出到
`logs/g1_upper/May02_11-49-20_stage2_upper_stable/`：
- 普通网络（MLP）导出为 `policy_1.pt`
- RNN 网络，导出为 `policy_lstm_1.pt`

导出 G1 12DoF 下半身策略：

```bash
python legged_gym/scripts/play.py \
  --task=g1 \
  --load_run=Apr13_07-17-29_ \
  --checkpoint=10000
```

导出 G1 23DoF 上半身/组合策略：

```bash
python legged_gym/scripts/play.py \
  --task=g1_upper \
  --training_stage=upper_body \
  --lower_body_checkpoint=logs/g1/Apr13_07-17-29_/model_10000.pt \
  --load_run=May02_11-49-20_stage2_upper_stable \
  --checkpoint=10000
```

上述 `upper_body` 组合 Play 时：`lower_body_checkpoint` 必须使用训练 checkpoint（`model_*.pt`），不要使用导出的 TorchScript 文件（`policy_lstm_1.pt`）。

导出 G1 23DoF 全身策略：

```bash
python legged_gym/scripts/play.py \
  --task=g1_upper \
  --training_stage=joint_finetune \
  --load_run=May02_11-49-23_fullbody_isaaclab_randomized \
  --checkpoint=10000
```

**G1 参考运动策略**（`g1_upper_motion_ref`）：与训练时相同，需设置 **`MOTION_REF_DATA_DIR`** 指向 mink pickle 目录，否则无法创建环境。需要图形界面时不要加 `--headless`。

```bash
export MOTION_REF_DATA_DIR=/path/to/mink/pickles

python legged_gym/scripts/play.py \
  --task=g1_upper_motion_ref \
  --training_stage=joint_finetune \
  --load_run=May13_12-16-09_g1_upper_motion_ref_mink
```

可选：`--checkpoint=8000`，或 `--checkpoint=logs/g1_upper_motion_ref/May13_12-16-09_g1_upper_motion_ref_mink/model_8000.pt` 指定权重；默认加载该 run 下最新的 `model_*.pt`。

**G1 AMP 策略**（`g1_upper_amp`）：与训练一致，需设置 **`MOTION_REF_DATA_DIR`** 以初始化动作库。**Play / 导出策略**时使用 Actor-Critic；若 checkpoint 中带判别器权重，`make_alg_runner` 仍会按任务配置拉起对应 Runner。

```bash
export MOTION_REF_DATA_DIR=/path/to/mink/pickles

python legged_gym/scripts/play.py \
  --task=g1_upper_amp \
  --training_stage=joint_finetune \
  --load_run=<你的_amp_run_目录名>
```

可选：`--checkpoint` 填迭代序号或 **`logs/g1_upper_amp/<date>_<run_name>/model_*.pt`** 完整路径。

### 3. Sim2Sim (Mujoco)

支持在 Mujoco 仿真器中运行 Sim2Sim：

```bash
python deploy/deploy_mujoco/deploy_mujoco.py {config_name}
```

#### 参数说明
- `config_name`: 配置文件，默认查询路径为 `deploy/deploy_mujoco/configs/`

#### 示例：运行 G1

```bash
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

#### ➡️  替换网络模型

默认模型位于 `deploy/pre_train/{robot}/motion.pt`；自己训练的模型需要先通过 Play 导出为 TorchScript，再替换 yaml 配置文件中的 `policy_path`。

G1 Mujoco 目前提供三种配置：

```bash
# 12DoF 下半身单策略
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml

# 23DoF 全身单策略
python deploy/deploy_mujoco/deploy_mujoco.py g1_23dof.yaml

# 12DoF 下半身策略 + 23DoF 上半身/全身策略组合推理
python deploy/deploy_mujoco/deploy_mujoco.py g1_upper_composite.yaml
```

对应配置文件：

- `deploy/deploy_mujoco/configs/g1.yaml`: 使用 `logs/g1/<run>/policy_lstm_1.pt`
- `deploy/deploy_mujoco/configs/g1_23dof.yaml`: 使用 `logs/g1_upper/<run>/policy_1.pt`
- `deploy/deploy_mujoco/configs/g1_upper_composite.yaml`: 同时配置 `policy_path` 和 `lower_body_policy_path`

组合推理时：

- `lower_body_policy_path` 控制前 12 个下半身关节。
- `policy_path` 可以是 23DoF 策略，部署脚本会自动取上半身部分；也可以是只输出上半身动作的策略。
- `upper_body_action_scale` 用于缩放上半身动作幅度。
- `clip_actions` 用于部署时裁剪策略输出动作，默认配置为 `1.0`，可避免 MuJoCo 中过大的未约束动作造成瞬间失稳。

#### 可视化相机（可选）

被动查看器可选用 **MjCAMERA_TRACKING** 锁定在机身侧向：球坐标在给定 body 坐标系内，随机器人平移和偏航一起动。示例：

```bash
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml \
  --camera-follow-side right \
  --camera-follow-distance 2.8 \
  --camera-track-body pelvis
```

参数：**`--camera-follow-side`** `none|right|left`，**`--camera-follow-distance`**，**`--camera-track-body`**（MJCF 中 `<body name=...>`，默认 `pelvis`），**`--camera-follow-elevation`**，可选 **`--camera-follow-azimuth`** 校正左右观感。

### 4. Sim2Real (实物部署)

实现实物部署前，确保机器人进入调试模式。详细步骤请参考 [实物部署指南](deploy/deploy_real/README.zh.md)：

```bash
python deploy/deploy_real/deploy_real.py {net_interface} {config_name}
```

#### 参数说明
- `net_interface`: 连接机器人网卡名称，如 `enp3s0`
- `config_name`: 配置文件，存在于 `deploy/deploy_real/configs/`，如 `g1.yaml`，`h1.yaml`，`h1_2.yaml`

---

## 🎉  致谢

本仓库开发离不开以下开源项目的支持与贡献，特此感谢：

- [legged\_gym](https://github.com/leggedrobotics/legged_gym): 构建训练与运行代码的基础。
- [PBHC](https://github.com/TeleHuman/PBHC) / [AMASS-POST-PROCESS fork](https://github.com/SLDRMK/AMASS-POST-PROCESS): Mink SMPL→机器人重定向管线。
- [GMR](https://github.com/YanjieZe/GMR) / [SLDRMK fork](https://github.com/SLDRMK/GMR): 另一套 SMPL-X 批量重定向与行走筛选。
- [AMASS](https://amass.is.tue.mpg.de/): 人体动作数据（本工作流使用 BMLrub / CMU / KIT 子集）。
- [rsl\_rl](https://github.com/leggedrobotics/rsl_rl.git): 强化学习算法实现。
- [mujoco](https://github.com/google-deepmind/mujoco.git): 提供强大仿真功能。
- [unitree\_sdk2\_python](https://github.com/unitreerobotics/unitree_sdk2_python.git): 实物部署硬件通信接口。


---

## 🔖  许可证

本项目根据 [BSD 3-Clause License](./LICENSE) 授权：
1. 必须保留原始版权声明。
2. 禁止以项目名或组织名作举。
3. 声明所有修改内容。

详情请阅读完整 [LICENSE 文件](./LICENSE)。

