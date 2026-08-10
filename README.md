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
gen_grasp.py                    # 抓握缓存生成 (reset_buf 即时重试 + 6向重力循环)

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
  revo3_right_grasp_cylinder.npy   # 圆柱抓握缓存

RL.md                           # RL 设计文档 (obs/act/rew/sim/DR 完整 spec)
```

### Config 说明

- **环境配置**: `hora/tasks/isaaclab/revo3_hand_hora_env_cfg.py`（`Revo3HandHoraEnvCfg`）定义所有环境超参：obs 维度、reward scale、DR 范围、PD 分组增益（7 组, 基于硬件辨识）、重力课程、接触传感器等
- **网络配置**: `configs/train/Revo3HandHora.yaml` 定义 PPO 超参、网络结构、训练步数
- **手部/物体**: `hora/tasks/isaaclab/assets.py` 定义 `REVO3_HAND_BALL_CFG`、`REVO3_HAND_CYLINDER_CFG`、`BALL_OBJECT_CFG`、`CYLINDER_OBJECT_CFG`，按 `--task` 自动选择
- **Checkpoint 路径**: `outputs/revo3_right/run_<task>/stage1_nn/best.pth`（Stage1）和 `outputs/revo3_right/run_<task>/stage2_nn/model_best.ckpt`（Stage2）
- **初始姿态**: `env.init_joint_pos` 从 `assets.py` 构建，用于无 cache 时的 reset；有 cache 时，`pos_diff_penalty` 使用每个环境实际抽到的 cache 关节姿态作为参照
- **PD 控制**: 7 组 per-joint-type 基础值（thumb_CMP:16.4/0.23, thumb_CMR:0.7/0.02, thumb_flexion:1.2/0.09, DIP:8.0/0.10, MPR:0.7/0.04, MCP:0.6/0.014, PIP:0.8/0.027）。每 reset 随机化 ×[0.5,2.0] per-DOF
- **最低 env 数**: 2048（batch_size = num_envs × 16 ≥ minibatch_size=32768 且可整除）
- **满重力 checkpoint**: `best.pth` 只在 9.81m/s² 下连续稳定评测 25 个 PPO epoch 后更新；课程阶段最优另存为 `best_curriculum.pth`

### 初始位姿验证

```bash
# ball: assets 初始姿态
~/IsaacLab/isaaclab.sh -p tools/view_init_pose.py --task ball --num_envs 1
# ball: 从 cache 加载 + 物理步进
~/IsaacLab/isaaclab.sh -p tools/view_init_pose.py --task ball --num_envs 1 --physics --cache
# cylinder: assets 初始姿态
~/IsaacLab/isaaclab.sh -p tools/view_init_pose.py --task cylinder --num_envs 1
# cylinder: 从 cache 加载 + 物理步进
~/IsaacLab/isaaclab.sh -p tools/view_init_pose.py --task cylinder --num_envs 1 --physics --cache
```

### 生成抓握缓存

```bash
# ball -> cache/revo3_right_grasp_ball.npy
~/IsaacLab/isaaclab.sh -p gen_grasp.py --task ball --num_envs 8192 --headless

# cylinder -> cache/revo3_right_grasp_cylinder.npy
~/IsaacLab/isaaclab.sh -p gen_grasp.py --task cylinder --num_envs 8192 --headless

# 指定输出文件名
~/IsaacLab/isaaclab.sh -p gen_grasp.py --task cylinder --num_envs 8192 --headless \
  --cache_file revo3_right_hand.npy
```

### 训练

```bash
conda activate env_isaaclab
cd BrainCo_hora

# Stage 1 (默认 num_envs=16384, max_agent_steps=300M)
scripts/train_s1.sh --task ball --num_envs 16384 --headless
scripts/train_s1.sh --task cylinder --num_envs 16384 --headless

# 本次 18 维特权观测版本建议使用新输出目录（示例为 4096 环境）
scripts/train_s1.sh run_cylinder_v2 --task cylinder --num_envs 4096 --headless

# Stage 2
scripts/train_s2.sh outputs/revo3_right/run_ball/stage1_nn/best.pth --task ball --num_envs 16384 --headless
scripts/train_s2.sh outputs/revo3_right/run_cylinder/stage1_nn/best.pth --task cylinder --num_envs 16384 --headless

# 当前 cylinder 续训结果进入 Stage 2；输出写入同一 run 的 stage2_nn/ 和 stage2_tb/
scripts/train_s2.sh outputs/revo3_right/run_cylinder_v2_continue/stage1_nn/last.pth \
  --task cylinder --num_envs 16384 --headless

# 追加 --force_overwrite 覆盖已有输出
```

旧 Stage1 checkpoint 的 critic/env_mlp 输入是 8 维，不能续训或评测当前 18 维网络；需要从头训练。若在旧输出目录启动全新训练，原 `best.pth` 会先备份为 `best.pre_retrain_<时间>.pth`，避免被误当作新的满重力模型。

Stage 2 冻结 Stage 1 actor/env_mlp，只训练 `adapt_tconv`。训练同时约束 latent MSE 和 teacher/student action MSE；rollout 先由 teacher 引导，再线性切换到 student。Stage1、Stage2 与实机部署的 actor `obs` 和 `proprio_hist` 均保留同顺序、同单位的 5 路指尖触觉。

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
