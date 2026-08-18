# BrainCo-hora In-hand Reorientation

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
  object_registry.py              # 物体任务注册表（采集/训练共用）
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
    objects/                    # 已居中、带动态凸分解碰撞的训练物体
      manifest.json             # 原始名称、固定缩放、尺寸、cache 映射
  urdf/urdf/
    revo3_right.urdf            # Revo3 右手 URDF

cache/
  revo3_right_grasp_ball.npy       # 球抓握缓存
  revo3_right_grasp_cylinder.npy   # 圆柱抓握缓存

RL.md                           # RL 设计文档 (obs/act/rew/sim/DR 完整 spec)
REWARD_AND_LOSS.md              # 当前 Stage1 奖励公式、日志缩放与 PPO loss
```

### Config 说明

- **环境配置**: `hora/tasks/isaaclab/revo3_hand_hora_env_cfg.py`（`Revo3HandHoraEnvCfg`）定义所有环境超参：obs 维度、reward scale、DR 范围、PD 分组增益（7 组, 基于硬件辨识）、重力课程、接触传感器等
- **网络配置**: `configs/train/Revo3HandHora.yaml` 定义 PPO 超参、网络结构、训练步数
- **手部/物体**: `hora/object_registry.py` 与 `assets/usd/objects/manifest.json` 是采集、训练和工具共用的注册表；`assets.py` 根据 `--task` 构建对应 `RigidObjectCfg`
- **Checkpoint 路径**: `outputs/revo3_right/run_<task>/stage1_nn/best.pth`（Stage1）和 `outputs/revo3_right/run_<task>/stage2_nn/model_best.ckpt`（Stage2）
- **初始姿态**: `env.init_joint_pos` 从 `assets.py` 构建，用于无 cache 时的 reset；有 cache 时每个环境从 cache 独立采样 reset 姿态，但奖励不再约束策略回到该姿态
- **PD 控制**: 7 组 per-joint-type 基础值（thumb_CMP:16.4/0.23, thumb_CMR:0.7/0.02, thumb_flexion:1.2/0.09, DIP:8.0/0.10, MPR:0.7/0.04, MCP:0.6/0.014, PIP:0.8/0.027）。每 reset 随机化 ×[0.8,1.2] per-DOF
- **最低 env 数**: 2048（batch_size = num_envs × 16 ≥ minibatch_size=32768 且可整除）
- **满重力 checkpoint**: `best.pth` 只在 9.81m/s²、低 reset、目标轴角速度 ≥0.5rad/s 且稳定旋转率 ≥30% 连续保持 25 个 PPO epoch 后更新；课程阶段最优另存为 `best_curriculum.pth`

### 扫描物体资产

以下缩放由 `assets/usd/objects/manifest.json` 在资产生成时传给
`UsdFileCfg.scale`。`training.usda` 只保存单位缩放的居中几何和
`convexDecomposition` 碰撞，因此修改 manifest 的 `scale` 后重新启动
查看、采集或训练脚本即可生效，不需要再编辑 USD。

| `--task` | 原始资产 | 缩放 | 训练尺寸 X×Y×Z (mm) |
|---|---|---:|---:|
| `great_dinos_triceratops` | Great_Dinos_Triceratops_Toy | 1.7 | 118.1×77.9×134.6 |
| `perricone_eye_cream` | Perricone_MD_Hypoallergenic_Firming_Eye_Cream_05_oz | 1.0 | 57.0×57.3×49.0 |
| `qabsorb_coq10` | QAbsorb_CoQ10_53iUqjWjW3O | 1.0 | 57.0×57.3×99.3 |
| `sapota_planter` | Sapota_Threshold_4_Ceramic_Round_Planter_Red | 0.7 | 74.2×76.7×68.0 |
| `toys_r_us_foobler` | Toys_R_Us_Treat_Dispenser_Smart_Puzzle_Foobler | 0.5 | 77.4×77.0×76.7 |
| `wilton_sprinkles` | Wilton_Pearlized_Sugar_Sprinkles_525_oz_Gold | 1.3 | 59.5×59.3×174.8 |

每种物体使用独立抓握缓存和输出目录，防止不同几何之间误用缓存或 checkpoint。

每个扫描物体还在 `manifest.json` 中独立配置：

- `grasp_seed.hand_pose_profile`：内置物体可选 `ball`/`cylinder`；扫描物体使用 `custom`。
- `grasp_seed.hand_joint_pos_rad`：`custom` 模式必须提供该物体独立的完整 21 关节角。
- `grasp_seed.object_pos_m` / `object_quat_wxyz`：无缓存时的物体初始位姿。
- `rotation.local_axis`：物体系中的语义旋转轴。
- `rotation.target_axis_world`：策略需要绕其旋转的世界坐标轴。
- `rotation.axis_bidirectional`：是否把正负轴视为等价。
- `rotation.enforce_axis_alignment` / `tilt_tolerance_deg`：轴对齐开关与容差。

奖励会先把 `local_axis` 按物体四元数变换到世界系，再与
`target_axis_world` 比较；旋转奖励直接使用物体世界角速度在
`target_axis_world` 上的投影。抓握采集和训练读取完全相同的配置。
六个扫描物体的 21 关节种子相互独立；当前数值是按物体尺寸生成的
待校准初值，开始抓握采集前应使用 `view_init_pose.py` 逐个检查穿模与接触。

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

# 扫描物体：默认显示物体局部 RGB 坐标轴；终端同时输出可复制回
# manifest 的 hand_joint_pos_rad 模板
~/IsaacLab/isaaclab.sh -p tools/view_init_pose.py \
  --task great_dinos_triceratops --num_envs 1

# 打开 21 关节实时编辑器（弧度）；滑块/数值框调整后立即更新手部姿态
~/IsaacLab/isaaclab.sh -p tools/view_init_pose.py \
  --task great_dinos_triceratops --num_envs 1 --edit_joints
```

`--edit_joints` 仅用于默认的冻结展示模式，不能和 `--physics` 或
`--headless` 同时使用。面板按手指分组显示所有关节及其 USD 限位：
`Reset` 恢复打开面板时的姿态，`Print JSON` 输出当前配置，
`Save to manifest.json` 只保存当前扫描物体的
`grasp_seed.hand_joint_pos_rad`。关闭 Isaac Sim 时也会在终端输出最终 JSON。
可通过 `--joint_step 0.005` 修改数值框的微调步长。

### 生成抓握缓存

```bash
# ball -> cache/revo3_right_grasp_ball.npy
~/IsaacLab/isaaclab.sh -p gen_grasp.py --task ball --num_envs 8192 --headless

# cylinder -> cache/revo3_right_grasp_cylinder.npy
~/IsaacLab/isaaclab.sh -p gen_grasp.py --task cylinder --num_envs 8192 --headless

# 指定输出文件名
~/IsaacLab/isaaclab.sh -p gen_grasp.py --task cylinder --num_envs 8192 --headless \
  --cache_file revo3_right_hand.npy

# 扫描物体示例；输出自动使用 cache/revo3_right_grasp_<task>.npy
~/IsaacLab/isaaclab.sh -p gen_grasp.py \
  --task great_dinos_triceratops --num_envs 8192 --target_count 8192 --headless

# 依次采集全部六种扫描物体
for task in great_dinos_triceratops perricone_eye_cream qabsorb_coq10 \
            sapota_planter toys_r_us_foobler wilton_sprinkles; do
  ~/IsaacLab/isaaclab.sh -p gen_grasp.py \
    --task "$task" --num_envs 8192 --target_count 8192 --headless
done
```

### 训练

```bash
conda activate env_isaaclab
cd BrainCo_hora

# Stage 1 (默认 num_envs=16384, max_agent_steps=300M)
scripts/train_s1.sh --task ball --num_envs 16384 --headless
scripts/train_s1.sh --task cylinder --num_envs 16384 --headless

# 扫描物体 Stage 1；默认输出 outputs/revo3_right/run_great_dinos_triceratops/
scripts/train_s1.sh --task great_dinos_triceratops --num_envs 16384 --headless

# 本次 18 维特权观测版本建议使用新输出目录（示例为 4096 环境）
scripts/train_s1.sh run_cylinder_v2 --task cylinder --num_envs 4096 --headless

# sapota_planter：从已有满重力旋转策略做“仅策略权重”热启动。
# 价值函数、价值归一化、优化器、训练步数和 best 分数会全部重置；
# 下方 30M steps 用于验证无姿态约束、10mm XY 免罚区和自碰撞惩罚。
scripts/train_s1.sh run_sapota_planter_rotation_v5_free_xy_self_collision \
  --task sapota_planter --num_envs 4096 --headless \
  --cache_file revo3_right_grasp_sapota_planter_full_g_narrow.npy \
  --checkpoint outputs/revo3_right/run_sapota_planter_rotation_v4_signed_1rads/stage1_nn/ep_950_step_0062M_g_9.81_reward_4.44.pth \
  --weights_only --fixed_train_gravity 9.81 \
  --max_agent_steps 30000000

# sapota_planter：从 v5 旋转策略热启动，加强局部 Z 轴对世界 Z 轴的
# 倾角约束和 X/Y 角速度阻尼。奖励已变化，因此必须使用 --weights_only。
scripts/train_s1.sh run_sapota_planter_rotation_v6_axis_stable \
  --task sapota_planter --num_envs 4096 --headless \
  --cache_file revo3_right_grasp_sapota_planter_full_g_narrow.npy \
  --checkpoint outputs/revo3_right/run_sapota_planter_rotation_v5_free_xy_self_collision/stage1_nn/ep_450_step_0029M_g_9.81_reward_19.86.pth \
  --weights_only --fixed_train_gravity 9.81 \
  --max_agent_steps 30000000

# Stage 2
scripts/train_s2.sh outputs/revo3_right/run_ball/stage1_nn/best.pth --task ball --num_envs 16384 --headless
scripts/train_s2.sh outputs/revo3_right/run_cylinder/stage1_nn/best.pth --task cylinder --num_envs 16384 --headless

# 扫描物体 Stage 2，--task 必须与 Stage 1 和抓握缓存一致
scripts/train_s2.sh \
  outputs/revo3_right/run_great_dinos_triceratops/stage1_nn/best.pth \
  --task great_dinos_triceratops --num_envs 16384 --headless

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

Stage 1 默认是严格断点续训，会恢复 actor、critic、归一化统计、optimizer、
学习率和步数。奖励定义发生变化时应增加 `--weights_only`：只保留 actor、
env_mlp、动作标准差和 observation RMS，以 `warmstart_learning_rate=1e-4`
重新训练；critic/value RMS/optimizer/计数器不会从旧奖励 checkpoint 恢复。

新的 `best.pth` / `best_full_gravity.pth` 必须连续 25 个 epoch 同时满足：
满重力、窗口 reset rate 不高于 0.3%、目标轴平均角速度至少 0.50 rad/s，且
稳定旋转帧比例至少 30%。`best_curriculum.pth` 仍只是诊断用，不应直接进入 Stage 2。

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
