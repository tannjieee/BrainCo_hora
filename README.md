# BrainCo-hora In-hand Reorient Ball/Cylinder

### 环境要求

| 组件 | 版本 |
|---|---|
| Python | 3.11（conda 环境 `env_isaaclab`） |
| PyTorch | 2.7.0+cu128 |
| Isaac Sim | 5.1.0.0（pip 安装） |
| Isaac Lab | 0.54.3（editable，`~/IsaacLab/source/isaaclab`） |

### 虚拟环境配置

```bash
conda create -y -n env_isaaclab python=3.11
conda activate env_isaaclab
python -m pip install --upgrade pip
```

### 项目结构

```text
hora/
  algo/
    ppo/                        # PPO 算法实现 (Stage1)
      ppo.py
      experience.py
    padapt/                     # ProprioAdapt 蒸馏算法 (Stage2)
      padapt.py
    models/                     # ActorCritic 网络 + RunningMeanStd
      models.py
      running_mean_std.py
  tasks/
    isaaclab/                   # Isaac Lab 仿真任务
      revo3_hand_hora_env_cfg.py   # 环境配置 (obs/rew/sim/DR/PG)
      revo3_hand_hora_env.py       # 环境逻辑 (reset/step/contact)
      assets.py                    # 手部/物体 ArticulationCfg + RigidObjectCfg
      hora_compat_wrapper.py       # DirectRLEnv → Hora PPO 兼容层
  utils/                        # 工具函数 (seed/format/metric)

configs/train/
  Revo3HandHora.yaml            # 训练超参数 (PPO/RMA, max_agent_steps=300M)

train.py                        # 训练 / 测试统一入口
gen_grasp.py                    # 单半径抓握缓存生成（逐环境、逐回合球面随机重力）

tools/
  view_init_pose.py             # 初始位姿可视化 / 抓握稳定性检查
  export_onnx.py                # Stage2 checkpoint → ONNX + deploy_meta.yaml 导出
  dump_runtime_actions.py       # 单环境动作序列导出 (真机回放)

scripts/
  train_s1.sh                   # Stage 1 PPO 训练启动
  train_s2.sh                   # Stage 2 ProprioAdapt 蒸馏启动

assets/
  usd/
    revo3_right.usd             # Revo3 右手 USD (URDF 转换)
    config.yaml                 # URDF→USD 转换配置
  urdf/urdf/
    revo3_right.urdf            # Revo3 右手 URDF

cache/
  revo3_right_grasp_ball.npy       # 球抓握缓存
  revo3_right_grasp_cylinder_r25mm.npy
  ...                              # 25–35 mm 共 11 份独立圆柱缓存
  revo3_right_grasp_cylinder_r35mm.npy

RL.md                           # RL 设计文档 (obs/act/rew/sim/DR 完整 spec)
```

### Config 说明

- **环境配置**: `hora/tasks/isaaclab/revo3_hand_hora_env_cfg.py`（`Revo3HandHoraEnvCfg`）定义所有环境超参：obs/priv 维度、reward scale、DR 范围、PD 分组增益（7 组，基于硬件辨识）、逐环境重力和接触传感器等
- **网络配置**: `configs/train/Revo3HandHora.yaml` 定义 PPO 超参、网络结构、训练步数
- **手部/物体**: `hora/tasks/isaaclab/assets.py` 定义手部、球体和圆柱配置；圆柱训练环境使用 25–35 mm 的 11 个离散半径，30 mm 占 40%，其余每档占 6%
- **Checkpoint 路径**: `outputs/revo3_right/run_<task>/stage1_nn/best.pth`（Stage1）和 `outputs/revo3_right/run_<task>/stage2_nn/model_best.ckpt`（Stage2）
- **初始姿态**: `env.init_joint_pos` 从 `assets.py` 构建，用于无 cache 时的 reset；有 cache 时，`pos_diff_penalty` 使用每个环境实际抽到的 cache 关节姿态作为参照
- **PD 控制**: 7 组 per-joint-type 基础值（thumb_CMP:16.4/0.23, thumb_CMR:0.7/0.02, thumb_flexion:1.2/0.09, DIP:8.0/0.10, MPR:0.7/0.04, MCP:0.6/0.014, PIP:0.8/0.027）。每 reset 随机化 ×[0.5,2.0] per-DOF
- **观测随机化**: 五指受力模长按 `force_N × 0.1 + N(0, 0.05)` 写入策略观测；关节编码器零位按环境/关节在每回合采样 `U(-0.02, 0.02) rad`，回合内保持固定
- **质量随机化**: 物体质量在环境初始化时采样 `U(0.05, 0.20) kg`
- **特权观测**: 21 维布局为位置偏移 3、摩擦 1、质量 1、COM 3、世界系重力单位方向 3、归一化半径 1、圆柱世界轴 3、角速度 3、线速度 3；半径编码为 `(radius_mm - 30) / 5`
- **最低 env 数**: 2048（batch_size = num_envs × 16 ≥ minibatch_size=32768 且可整除）
- **重力**: 不使用重力模长课程。scene/global gravity 固定为零；每个环境每回合球面均匀采样方向，并在整个回合中以 `9.81 m/s²` 的等效世界系力作用于物体质心
- **圆柱缓存前置条件**: 圆柱训练启动前必须生成 `r25mm` 至 `r35mm` 的全部 11 份缓存；缺少任意一份都会直接报错

### 初始位姿验证

```bash
# ball: assets 初始姿态
~/IsaacLab/isaaclab.sh -p tools/view_init_pose.py --task ball --num_envs 1
# ball: 从 cache 加载 + 物理步进
~/IsaacLab/isaaclab.sh -p tools/view_init_pose.py --task ball --num_envs 1 --physics --cache
# cylinder: assets 初始姿态
~/IsaacLab/isaaclab.sh -p tools/view_init_pose.py --task cylinder --num_envs 1
# cylinder: 当前查看工具固定 30mm；显式选择对应缓存 + 物理步进
~/IsaacLab/isaaclab.sh -p tools/view_init_pose.py --task cylinder --num_envs 1 --physics \
  --cache_file revo3_right_grasp_cylinder_r30mm.npy
```

### 生成抓握缓存

```bash
# ball -> cache/revo3_right_grasp_ball.npy
~/IsaacLab/isaaclab.sh -p gen_grasp.py --task ball --num_envs 8192 --headless

# cylinder：每个命令只采一个半径，默认输出到对应的 rXXmm 文件
~/IsaacLab/isaaclab.sh -p gen_grasp.py --task cylinder --cylinder_radius_mm 25 --num_envs 8192 --headless
~/IsaacLab/isaaclab.sh -p gen_grasp.py --task cylinder --cylinder_radius_mm 26 --num_envs 8192 --headless
~/IsaacLab/isaaclab.sh -p gen_grasp.py --task cylinder --cylinder_radius_mm 27 --num_envs 8192 --headless
~/IsaacLab/isaaclab.sh -p gen_grasp.py --task cylinder --cylinder_radius_mm 28 --num_envs 8192 --headless
~/IsaacLab/isaaclab.sh -p gen_grasp.py --task cylinder --cylinder_radius_mm 29 --num_envs 8192 --headless
~/IsaacLab/isaaclab.sh -p gen_grasp.py --task cylinder --cylinder_radius_mm 30 --num_envs 8192 --headless
~/IsaacLab/isaaclab.sh -p gen_grasp.py --task cylinder --cylinder_radius_mm 31 --num_envs 8192 --headless
~/IsaacLab/isaaclab.sh -p gen_grasp.py --task cylinder --cylinder_radius_mm 32 --num_envs 8192 --headless
~/IsaacLab/isaaclab.sh -p gen_grasp.py --task cylinder --cylinder_radius_mm 33 --num_envs 8192 --headless
~/IsaacLab/isaaclab.sh -p gen_grasp.py --task cylinder --cylinder_radius_mm 34 --num_envs 8192 --headless
~/IsaacLab/isaaclab.sh -p gen_grasp.py --task cylinder --cylinder_radius_mm 35 --num_envs 8192 --headless
```

对应输出为 `cache/revo3_right_grasp_cylinder_r25mm.npy` 至 `cache/revo3_right_grasp_cylinder_r35mm.npy`。采集默认采用每环境、每回合固定的球面均匀 `9.81 m/s²` 重力方向，global gravity 为零。已有输出默认不会被覆盖；确认重采时追加 `--force_overwrite`。

### 训练

```bash
conda activate env_isaaclab
cd BrainCo_hora

# Stage 1 (默认 num_envs=16384, max_agent_steps=300M)
scripts/train_s1.sh --task ball --num_envs 16384 --headless
scripts/train_s1.sh --task cylinder --num_envs 16384 --headless

# 当前 21 维特权观测版本必须使用新输出目录（示例为 4096 环境）
scripts/train_s1.sh run_cylinder_v2 --task cylinder --num_envs 4096 --headless

# Stage 2
scripts/train_s2.sh outputs/revo3_right/run_ball/stage1_nn/best.pth --task ball --num_envs 16384 --headless
scripts/train_s2.sh outputs/revo3_right/run_cylinder/stage1_nn/best.pth --task cylinder --num_envs 16384 --headless

# 当前 cylinder 续训结果进入 Stage 2；输出写入同一 run 的 stage2_nn/ 和 stage2_tb/
scripts/train_s2.sh outputs/revo3_right/run_cylinder_v2_continue/stage1_nn/last.pth \
  --task cylinder --num_envs 16384 --headless

# 追加 --force_overwrite 覆盖已有输出
```

圆柱环境在 16384 个环境下精确分配为：30 mm 共 6554 个环境，其他十档各 983 个环境。半径在环境生命周期内固定，reset 时只从相同半径的缓存抽取初始抓取。

当前 `priv_info` 为 21 维，并加入三维重力方向和归一化半径。旧的 8 维或 18 维特权观测 Stage1 checkpoint 与当前 critic/env_mlp 不兼容，不能续训或用于当前环境评测；由这些旧 Stage1 模型得到的 Stage2 checkpoint 也应重新训练。触觉缩放、关节零位、半径和重力随机化均已改变训练分布，不应依赖 `strict=False` 强行复用旧模型。

Stage 2 冻结 Stage 1 actor/env_mlp，只训练 `adapt_tconv`。训练同时约束 latent MSE 和 teacher/student action MSE；rollout 先由 teacher 引导，再线性切换到 student。Stage1、Stage2 与实机部署的 actor `obs` 和 `proprio_hist` 均保留同顺序的 5 路指尖受力模长。每路力观测先乘 `0.1`，仿真训练再叠加归一化空间中标准差为 `0.05` 的高斯噪声；关节编码器零位每回合按关节随机偏置。

### 断点续训

```bash
# ball: Stage 1 续训 → outputs/revo3_right/run1_continue/
scripts/train_s1.sh --task ball --num_envs 16384 --headless \
  --checkpoint outputs/revo3_right/run_ball/stage1_nn/last.pth

# ball: Stage 2 续训 → 保持在原 run_ball/stage2_nn/
scripts/train_s2.sh outputs/revo3_right/run_ball/stage2_nn/model_last.ckpt \
  --task ball --num_envs 16384 --headless

# cylinder: Stage 1 续训
scripts/train_s1.sh --task cylinder --num_envs 16384 --headless \
  --checkpoint outputs/revo3_right/run_cylinder/stage1_nn/last.pth

# cylinder: Stage 2 续训
scripts/train_s2.sh outputs/revo3_right/run_cylinder/stage2_nn/model_last.ckpt \
  --task cylinder --num_envs 16384 --headless
```

### 推理可视化

```bash
# ball: Stage 1 policy
~/IsaacLab/isaaclab.sh -p train.py \
  --task ball --algo PPO --num_envs 32 --test \
  --checkpoint outputs/revo3_right/run_ball/stage1_nn/best.pth

# ball: Stage 2 policy
~/IsaacLab/isaaclab.sh -p train.py \
  --task ball --algo ProprioAdapt --num_envs 32 --test \
  --checkpoint outputs/revo3_right/run_ball/stage2_nn/model_best.ckpt

# cylinder: Stage 1 policy
~/IsaacLab/isaaclab.sh -p train.py \
  --task cylinder --algo PPO --num_envs 32 --test \
  --checkpoint outputs/revo3_right/run_cylinder/stage1_nn/best.pth

# cylinder: 400 个策略步（20 秒）的无界面满重力定量评测
~/IsaacLab/isaaclab.sh -p train.py \
  --task cylinder --algo PPO --num_envs 256 --headless --test --test_steps 400 \
  --checkpoint outputs/revo3_right/run_cylinder_v2/stage1_nn/best.pth

# cylinder: 单环境、实时节拍、三分之四近景，自动录制 10 秒 MP4
~/IsaacLab/isaaclab.sh -p train.py \
  --task cylinder --algo PPO --num_envs 1 --test --real-time --video \
  --video_seconds 10 \
  --camera_eye 0.55 -0.55 1.85 --camera_lookat 0 0 1.50 \
  --checkpoint outputs/revo3_right/run_cylinder_v2_continue/stage1_nn/last.pth

# MP4 默认写入 outputs/revo3_right/videos/；无界面录制可额外传 --headless。

# cylinder: Stage 2 policy
~/IsaacLab/isaaclab.sh -p train.py \
  --task cylinder --algo ProprioAdapt --num_envs 32 --test \
  --checkpoint outputs/revo3_right/run_cylinder/stage2_nn/model_best.ckpt
```

### Tool

#### 导出 ONNX

`tools/export_onnx.py` 仅用于导出 Stage2 / ProprioAdapt checkpoint。导出的 ONNX 内部已包含 `running_mean_std` 和 `sa_mean_std`，部署侧输入原始观测即可，不要额外做 RMS 归一化。

```bash
# ball: Stage 2 checkpoint
python tools/export_onnx.py \
  --checkpoint outputs/revo3_right/run_ball/stage2_nn/model_best.ckpt \
  --output outputs/revo3_right/onnx/ball_stage2.onnx

# cylinder: Stage 2 checkpoint
python tools/export_onnx.py \
  --checkpoint outputs/revo3_right/run_cylinder/stage2_nn/model_best.ckpt \
  --output outputs/revo3_right/onnx/cylinder_stage2.onnx
```

#### 导出单环境动作序列

输出每回合 3 个文件：`.raw_action.txt`（策略原始增量输出）、`.target.txt`（delta 叠加后的位置目标）、`.jointpos.txt`（实际关节角）。

```bash
# ball: Stage 1 (PPO)
~/IsaacLab/isaaclab.sh -p tools/dump_runtime_actions.py \
  --checkpoint outputs/revo3_right/run_ball/stage1_nn/best.pth \
  --task ball --algo PPO --episodes 1

# ball: Stage 2 (ProprioAdapt)
~/IsaacLab/isaaclab.sh -p tools/dump_runtime_actions.py \
  --checkpoint outputs/revo3_right/run_ball/stage2_nn/model_best.ckpt \
  --task ball --algo ProprioAdapt --episodes 1

# cylinder: Stage 1 (PPO)
~/IsaacLab/isaaclab.sh -p tools/dump_runtime_actions.py \
  --checkpoint outputs/revo3_right/run_cylinder/stage1_nn/best.pth \
  --task cylinder --algo PPO --episodes 1

# cylinder: Stage 2 (ProprioAdapt)
~/IsaacLab/isaaclab.sh -p tools/dump_runtime_actions.py \
  --checkpoint outputs/revo3_right/run_cylinder/stage2_nn/model_best.ckpt \
  --task cylinder --algo ProprioAdapt --episodes 1
```

### 监控训练

```bash
conda activate env_isaaclab
tensorboard --logdir /home/m4ximilian/BrainCo_hora/outputs
```

### URDF 转 USD（可选）

```bash
~/IsaacLab/isaaclab.sh -p /home/m4ximilian/IsaacLab_Revo3/BrainCo/tool/convert_urdf_new.py \
  assets/urdf/urdf/revo3_right.urdf \
  assets/usd/revo3_right.usd \
  --fix-base
```

tmux ls
tmux new -s train
tmux attach -t train
tmux kill-session -t train
