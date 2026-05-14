<div align="center">
  <h1 align="center">Unitree RL GYM</h1>
  <p align="center">
    <a href="README.md">🌎 English</a> | <span>🇨🇳 中文</span>
  </p>
</div>

<p align="center">
  🎮🚪 <strong>这是一个基于 Unitree 机器人实现强化学习的示例仓库，支持 Unitree Go2、H1、H1_2和 G1。</strong> 🚪🎮
</p>

<div align="center">

| <div align="center"> Isaac Gym </div> | <div align="center">  Mujoco </div> |  <div align="center"> Physical </div> |
|--- | --- | --- |
| [<img src="https://oss-global-cdn.unitree.com/static/32f06dc9dfe4452dac300dda45e86b34.GIF" width="240px">](https://oss-global-cdn.unitree.com/static/5bbc5ab1d551407080ca9d58d7bec1c8.mp4) | [<img src="https://oss-global-cdn.unitree.com/static/244cd5c4f823495fbfb67ef08f56aa33.GIF" width="240px">](https://oss-global-cdn.unitree.com/static/5aa48535ffd641e2932c0ba45c8e7854.mp4) | [<img src="https://oss-global-cdn.unitree.com/static/78c61459d3ab41448cfdb31f6a537e8b.GIF" width="240px">](https://oss-global-cdn.unitree.com/static/0818dcf7a6874b92997354d628adcacd.mp4) |

</div>

---

## 📦 安装配置

安装和配置步骤请参考 [setup.md](/doc/setup_zh.md)

## 🔁 流程说明

强化学习实现运动控制的基本流程为：

`Train` → `Play` → `Sim2Sim` → `Sim2Real`

- **Train**: 通过 Gym 仿真环境，让机器人与环境互动，找到最满足奖励设计的策略。通常不推荐实时查看效果，以免降低训练效率。
- **Play**: 通过 Play 命令查看训练后的策略效果，确保策略符合预期。
- **Sim2Sim**: 将 Gym 训练完成的策略部署到其他仿真器，避免策略小众于 Gym 特性。
- **Sim2Real**: 将策略部署到实物机器人，实现运动控制。

## 🛠️ 使用指南

### 1. 训练

运行以下命令进行训练：

```bash
python legged_gym/scripts/train.py --task=xxx
```

#### ⚙️  参数说明
- `--task`: 必选参数；常用 `go2`, `g1`, `h1`, `h1_2`，G1 全身另见 `g1_upper`、`g1_upper_motion_ref`（下文「G1 分阶段训练」「参考运动」）
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

#### G1 参考运动（档位 A：运动匹配塑形）

独立于 `g1_upper` 的任务 **`g1_upper_motion_ref`**：在全身 `joint_finetune` 上增加对照 mink 重定向行走数据的关节参考奖励（数据为 `convert_fit_motion.py` 产出的 `*_poses.pkl` 等）。实验目录名为 **`g1_upper_motion_ref`**，日志在 `logs/g1_upper_motion_ref/`。

训练前必须指定动作数据目录（与训练脚本内默认可改）：

```bash
export MOTION_REF_DATA_DIR=/path/to/mink/pickles   # 例如 AMASS 管线下的 retargeted_motion_data/mink

bash legged_gym/scripts/train_g1_upper_motion_ref.sh
```

也可自行调用 `python legged_gym/scripts/train.py --task=g1_upper_motion_ref --training_stage=joint_finetune ...`。若配置里 `motion_ref.data_dir` 为空，则依赖环境变量 **`MOTION_REF_DATA_DIR`**，否则环境初始化会报错。

超参见 `legged_gym/envs/g1/g1_config.py` 中的 `G1UpperBodyMotionRefCfg`：`motion_ref_dof` 权重、`motion_ref.err_reduce`、**σ 为关节误差向量 L2 范数尺度（rad），课程按 `σ ← max(σ_min, min(batch均值‖q−q_ref‖₂, σ))` 更新，奖励为 `exp(−mse/σ²)`**、`curriculum_norm_ema_alpha` 等。

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

上述 `upper_body` 组合 Play 时：`lower_body_checkpoint` 必须使用训练 checkpoint（`model_*.pt`），不要使用导出的 TorchScript 文件（`policy_lstm_1.pt`）。
  
### Play 效果

| Go2 | G1 | H1 | H1_2 |
|--- | --- | --- | --- |
| [![go2](https://oss-global-cdn.unitree.com/static/ba006789e0af4fe3867255f507032cd7.GIF)](https://oss-global-cdn.unitree.com/static/d2e8da875473457c8d5d69c3de58b24d.mp4) | [![g1](https://oss-global-cdn.unitree.com/static/32f06dc9dfe4452dac300dda45e86b34.GIF)](https://oss-global-cdn.unitree.com/static/5bbc5ab1d551407080ca9d58d7bec1c8.mp4) | [![h1](https://oss-global-cdn.unitree.com/static/fa04e73966934efa9838e9c389f48fa2.GIF)](https://oss-global-cdn.unitree.com/static/522128f4640c4f348296d2761a33bf98.mp4) |[![h1_2](https://oss-global-cdn.unitree.com/static/83ed59ca0dab4a51906aff1f93428650.GIF)](https://oss-global-cdn.unitree.com/static/15fa46984f2343cb83342fd39f5ab7b2.mp4)|

---

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

#### 运行效果

| G1 | H1 | H1_2 |
|--- | --- | --- |
| [![mujoco_g1](https://oss-global-cdn.unitree.com/static/244cd5c4f823495fbfb67ef08f56aa33.GIF)](https://oss-global-cdn.unitree.com/static/5aa48535ffd641e2932c0ba45c8e7854.mp4)  |  [![mujoco_h1](https://oss-global-cdn.unitree.com/static/7ab4e8392e794e01b975efa205ef491e.GIF)](https://oss-global-cdn.unitree.com/static/8934052becd84d08bc8c18c95849cf32.mp4)  |  [![mujoco_h1_2](https://oss-global-cdn.unitree.com/static/2905e2fe9b3340159d749d5e0bc95cc4.GIF)](https://oss-global-cdn.unitree.com/static/ee7ee85bd6d249989a905c55c7a9d305.mp4) |


---

### 4. Sim2Real (实物部署)

实现实物部署前，确保机器人进入调试模式。详细步骤请参考 [实物部署指南](deploy/deploy_real/README.zh.md)：

```bash
python deploy/deploy_real/deploy_real.py {net_interface} {config_name}
```

#### 参数说明
- `net_interface`: 连接机器人网卡名称，如 `enp3s0`
- `config_name`: 配置文件，存在于 `deploy/deploy_real/configs/`，如 `g1.yaml`，`h1.yaml`，`h1_2.yaml`

#### 运行效果

| G1 | H1 | H1_2 |
|--- | --- | --- |
| [![real_g1](https://oss-global-cdn.unitree.com/static/78c61459d3ab41448cfdb31f6a537e8b.GIF)](https://oss-global-cdn.unitree.com/static/0818dcf7a6874b92997354d628adcacd.mp4) | [![real_h1](https://oss-global-cdn.unitree.com/static/fa07b2fd2ad64bb08e6b624d39336245.GIF)](https://oss-global-cdn.unitree.com/static/ea0084038d384e3eaa73b961f33e6210.mp4) | [![real_h1_2](https://oss-global-cdn.unitree.com/static/a88915e3523546128a79520aa3e20979.GIF)](https://oss-global-cdn.unitree.com/static/12d041a7906e489fae79d55b091a63dd.mp4) |

---

## 🎉  致谢

本仓库开发离不开以下开源项目的支持与贡献，特此感谢：

- [legged\_gym](https://github.com/leggedrobotics/legged_gym): 构建训练与运行代码的基础。
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

