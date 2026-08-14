# Revo3 灵巧手物体旋转 — RL 设计文档

## 1. RL 框架

两阶段训练，基于 Rapid Motor Adaptation (RMA) 思想。

### Stage1 — PPO 教师策略

```
                    ┌──────────┐
   obs (141) ──────→│          │
   priv_info (18) ─→│ ActorCritic │──→ mu (21) ─→ Normal(μ,σ) ─→ action (21)
                    │          │──→ value (1)
                    └──────────┘
```

- **算法**: PPO + GAE-lambda
- **特权信息**: `env_mlp` 将 18 维 priv_info（位置偏差/摩擦/质量/COM/重力大小/物体配置轴的世界方向/物体角速度/线速度）编码为 8 维隐变量，tanh 后拼入观测 → 149 维入 `actor_mlp`
- **网络**: `actor_mlp` [512,256,128] ELU, value 头 Linear(128,1), mu 头 Linear(128,21), 可学习 log_std
- **PPO 超参**: lr=3e-4, gamma=0.99, tau=0.95, kl_threshold=0.01, e_clip=0.2, critic_coef=2, entropy_coef=0.001, bounds_loss_coef=0.001
- **数据流**: horizon=16；4096 环境时为 65536 transitions/epoch，minibatch=32768，3 mini-epochs
- **奖励缩放**: ×0.01 后用于 GAE
- **训练上限**: 300M agent steps
- **最低 env 数**: 2048（batch_size = num_envs × 16 必须 ≥ minibatch_size 且可整除）

### Stage2 — ProprioAdapt 学生蒸馏

```
                    ┌──────────────────┐
   obs (141) ──────→│                  │
   proprio_hist     │   ActorCritic    │──→ mu (21) ─→ action (21)
   (30×47) ────────→│ (freeze 除 adapt) │
                    └──────────────────┘
                         ↑
                    adapt_tconv  ←── MSE loss ←── env_mlp(priv_info).tanh()
                    (可训练)                      (冻结, 真值)
```

- **目标**: 让 `adapt_tconv` 从 30 帧本体历史 (proprio_hist) 中估计出 `env_mlp(priv_info)` 的 8 维隐变量，部署时不再需要 priv_info
- **网络**: ProprioAdaptTConv: channel_transform(47→32→32) → Conv1d(32,32,9,2)→Conv1d(32,32,5,1)→Conv1d(32,32,5,1) → flatten(96) → Linear(96,8) → tanh
- **损失**: latent MSE + teacher/student action MSE；action loss 让 latent 误差按其策略影响参与优化
- **可训练**: 仅 adapt_tconv 参数 (lr=3e-4 Adam), 其余全部冻结
- **训练方式**: 在线 (每步一次 forward+backward, 无需 experience buffer)
- **Rollout**: 前 10M transitions 使用 teacher，随后 40M transitions 线性切换到纯 student
- **加载**: 从 Stage1 checkpoint strict=False 暖启动
- **训练上限**: 300M agent steps
- **Stage2 触觉策略**: `enable_contact_in_obs=True` — actor 与 adapt_tconv 历史均使用真实 5 路指尖触觉
- **部署输入**: Stage2 ONNX 不接收 `priv_info`。`priv_info` 只在训练时通过冻结的 `env_mlp(priv_info)` 生成 teacher latent target；部署时由 `adapt_tconv(proprio_hist)` 替代。
- **观测 ABI**: Stage2 保留 Stage1 actor 的 `obs(141)` 输入维度；实机必须提供同顺序、同单位、同采样率的触觉字段。

### 训练配置 (YAML)

```yaml
# configs/train/Revo3HandHora.yaml
network:
  mlp:       {units: [512, 256, 128]}
  priv_mlp:  {units: [256, 128, 8]}
ppo:
  learning_rate: 5e-3    gamma: 0.99          tau: 0.95
  kl_threshold: 0.02     horizon_length: 8    minibatch_size: 32768
  mini_epochs: 5         e_clip: 0.2          critic_coef: 4
  entropy_coef: 0.0      bounds_loss_coef: 0.0001
  max_agent_steps: 300M
  reward_scale: 0.01
stage2:
  learning_rate: 3e-4
  latent_loss_coef: 1.0
  action_loss_coef: 0.25
  teacher_warmup_steps: 10M
  teacher_mix_steps: 40M
  best_after_steps: 50M
  best_min_episodes: 4096
  save_interval_steps: 25M
  log_interval: 20
```

### 检查点

| | Stage1 | Stage2 |
|---|---|---|
| 保存 | model+optimizer+agent_steps+epoch_num+best_rewards+last_lr+rms+vms → `.pth` | model+optimizer+agent_steps+best_rewards+rms+sa_ms → `.ckpt` |
| 触发 | epoch_num % 500 == 0 | iter_num % 500 == 0 |
| best | mean_rewards > best_rewards | mean_rewards > best_rewards |

### 域随机化 (DR)

| 参数 | 值 | 时机 |
|---|---|---|
| 物体形状 | 每次运行由 `--task` 选择球、圆柱或注册表中的扫描物体 | — |
| 物体缩放 | 球/圆柱为 1.0；扫描物体使用 `assets/usd/objects/manifest.json` 中各自的固定比例 | — |
| 物体质量 | U(0.01, 0.20) kg | init 一次 |
| 摩擦 | 手 metal_base=0.1, object_base=0.5, scale×U(0.5, 2.0) | init 一次 |
| COM | U(-0.01, 0.01) m | init 一次 |
| PD gains | per-joint-type base × [0.5, 2.0], 每 DOF 独立 | 每 reset |
| 随机外力 | force_scale=2.0, prob=0.25, decay=0.9 | 每步 |
| 重力课程 | (0,0,-0.05) → 递增 0.05/step → 上限 10 m/s² | 自动 |

DR 在 init 时随机化一次，PD 每 reset 重新随机化。

### 终止条件

- **高度越界**: 物体 Z 超出 [init_Z - 0.02, init_Z + 0.02] 窗口 (±2cm)
- **超时**: episode 长度 ≥ 20s (400 步 @20Hz)
- **重力课程**: 当高度越界率 < 0.05% 且 step > 1000 时, 重力递增 0.05 m/s², 上限 10 m/s²

### 物理配置

| 参数 | 值 |
|---|---|
| 物理频率 | 240 Hz |
| 控制频率 | 20 Hz (decimation=12) |
| solver | TGS (type=1), 8 position iter, 0 velocity iter |
| 碰撞 | contact_offset=0.002, rest_offset=0.0 |
| 手部 | disable_gravity, fix_root_link, 自碰撞开 |
| 物体 | gyroscopic_forces 开, max_depenetration 1000 |
| 初始重力 | (0, 0, -0.05) — 重力课程递增至 10 |

### PD控制

| 模式 | 发送给 PhysX 的指令 | 执行效果 |
|---|---|---|
| `torque_control=True` | `set_joint_effort_target(torques)` | **阻抗控制**：关节行为像弹簧+阻尼，碰到物体会自然屈服 |
| `torque_control=False` | `set_joint_position_target(targets)` | **刚性位置跟踪**：PhysX 内部用 USD 默认 stiffness=100 强行拉到目标位置 |

力矩控制的本质是**阻抗控制 (Impedance Control)**：关节不强制到达目标位置，而是像一个虚拟弹簧-阻尼系统——偏离目标越远，回复力矩越大，但允许外力推开关节。这对抓取操作至关重要：

1. **柔顺性 (Compliance)**：手指碰到物体时会被推开而非硬顶。位置控制模式下物体会被弹飞或穿透手指。
2. **力限制**：策略不需要学习精确的接触力，PD 参数天然定义了最大接触力 ≈ `p_gain × 最大位置偏差`。
3. **Sim-to-real**：真机底层是电流/力矩控制。仿真用阻抗控制可以匹配真机执行器的动力学特性。

#### PD 参数含义

```
torque = p_gain × (target - joint_pos) - d_gain × joint_vel
```

`p_gain` 和 `d_gain` 是虚拟弹簧的**刚度**和**阻尼**：

| 参数 | 物理含义 | 调大效果 | 调小效果 |
|---|---|---|---|
| `p_gain` (刚度) | 位置偏差→力矩的增益 | 跟踪更紧，但接触冲击大 | 更柔顺，但跟踪松 |
| `d_gain` (阻尼) | 速度→力矩的增益 | 响应迟钝，能量耗散大 | 响应快，但可能振荡 |

训练设置了高度随机化 (`Kp×[0.5,2.0]`, `Kd×[0.5,2.0]`, per-DOF 独立)，以提高策略对刚度/阻尼变化的鲁棒性。

#### 来源与基础值

- **来源**: 显式硬编码定义 (`revo3_hand_hora_env_cfg.py` L33-58)，不从 URDF/USD 读取
- **基础值**: 按关节类型分组，来源于真实硬件动力学辨识 (`Dynamic_identication/controller_para/parameter.yaml`)

| 组名 | 包含关节 | Kp | Kd |
|---|---|---|---|
| thumb_CMP | thumb_CMP | 16.4 | 0.23 |
| thumb_CMR | thumb_CMR | 0.7 | 0.02 |
| thumb_flexion | thumb_MCP, thumb_PIP | 1.2 | 0.09 |
| DIP | 全部 5 个 DIP | 8.0 | 0.10 |
| MPR | 4 个 MPR | 0.7 | 0.04 |
| MCP | 4 个 finger MCP | 0.6 | 0.014 |
| PIP | 4 个 finger PIP | 0.8 | 0.027 |

- **初始化**: `self._p_gain_base` / `self._d_gain_base` 存储 per-DOF 基础值 (shape `(21,)`)，`self.p_gain` / `self.d_gain` 扩展为 `(num_envs, 21)`

### 初始姿态

- **定义**: `env.__init__` 中 `self.init_joint_pos` (shape `(1,21)`)，从 `assets.py` 的 `robot_cfg.init_state.joint_pos` 构建
- **用途**: 无 cache 时的 reset 起始姿态；有 cache 时，`pos_diff_penalty` 参照为每个环境实际抽到的 cache 关节姿态

---

## 2. 观测 (Observation)

### 策略观测 (141 维)

3 帧滑动窗口，每帧 47 维:

| 偏移 | 维度 | 内容 |
|---|---|---|
| 0-20 | 21 | 关节位置 (归一化到 [-1,1]) |
| 21-41 | 21 | 当前关节目标值 |
| 42-46 | 5 | 5 指 DIP 对目标物体的过滤后合力（20Hz） |

- 关节位置加噪 ±0.02 rad
- 接触力: 每 0.05s 从 `force_matrix_w` 读取一次，默认不额外保持旧采样值（`contact_latency=0`）
- 整帧 clamp [-5, 5]
- 观测归一化: RunningMeanStd (PPO 训练时在线更新)

Stage2 / ProprioAdapt 部署约定:

- `obs` 仍是 141 维，即 3×47，不删除触觉维度
- `obs` 帧顺序为 `[t-2, t-1, t]`，按时间顺序展平
- `obs` 中三段接触力必须填入实测值: `[42:47]`, `[89:94]`, `[136:141]`
- `priv_info` 不拼入 `obs`，也不是 Stage2 ONNX 输入

### 特权信息 priv_info (18 维)

| 偏移 | 内容 |
|---|---|
| 0-2 | 物体相对于初始位置的偏移 (xyz) |
| 3 | 摩擦系数缩放因子 |
| 4 | 物体质量 (kg) |
| 5-7 | 物体质心偏移 (xyz) |
| 8 | 重力大小 (m/s²) |
| 9-11 | manifest 中配置的物体局部旋转轴在世界坐标系中的方向 |
| 12-14 | 物体世界坐标角速度 |
| 15-17 | 物体世界坐标线速度 |

- 仅 Stage1 可用, Stage2 被 adapt_tconv 蒸馏

### 本体历史 proprio_hist (30×47)

30 帧滑动窗口 (1.5s 历史 @20Hz), 结构与单帧观测相同。仅 Stage2 使用, 经 `sa_mean_std` 归一化。

部署约定:

- 形状为 `[B,30,47]`
- 帧顺序为 oldest → newest，最后一帧对应当前策略时刻 `t`
- 每帧布局同单帧观测: `[joint_pos_unscaled(21), cur_targets(21), contact_forces(5)]`
- 仿真和实机的 `proprio_hist` 每帧 `[42:47]` 均填入真实触觉值

### 接触传感器配置

5 个 ContactSensor, prim 路径为 DIP_Link:

- `history_length=3`: 传感器内部仍保留最近 3 个物理帧，训练观测不对它们做跨帧平均
- `filter_prim_paths_expr=["/World/envs/env_.*/object"]`: 仅检测与物体的接触
- 合力: 每个策略步从 `force_matrix_w` 读取一次当前“指尖—目标物体”过滤后合力并取模，频率 20Hz、周期 0.05s；不使用 1s 训练窗口
- 延迟: 默认 `contact_latency=0`，每个策略步都更新；需要做 sim2real 延迟随机化时可单独调高
- `enable_tactile=True`: 接触力写入观测
- `enable_contact_in_obs=True` (Stage1/Stage2): actor 和 adapt_tconv 历史都保留真实触觉
- `binary_contact=False`: 使用连续力值 (非二值)
- `enable_contact_pos=False`: 接触位置不写入观测 (置零)

观测中的 5 维接触力 = `[thumb_force, index_force, middle_force, ring_force, pinky_force]`。

**Sim2real**: 仿真通过 PhysX 接触求解获取接触力，实机通过 5 路指尖触觉提供对应合力。部署前需要统一指尖顺序、牛顿单位、量程、偏置、采样周期和延迟，并用实机标定数据验证触觉分布。

---

## 3. 动作 (Action)

### 动作空间

21 维连续动作 ∈ [-1, 1], 对应 Revo3 右手全部 21 个关节。

### 完整动作管线

策略输出**关节位置增量**，经过 PD 控制器转换为**力矩**，发送给物理引擎执行。分四步：

```
Step 1: Policy 输出位置增量
        action = policy(obs)                     # 21 维, clip [-1, 1]

Step 2: 累积为位置目标
        target = prev_target + (1/24) × action   # delta 叠加, 步长 ~2.4°/step
        target = clamp(target, joint_lower, joint_upper)

Step 3: PD 转换为力矩
        torque = p_gain × (target - joint_pos) - d_gain × joint_vel

Step 4: 发送力矩指令给 PhysX
        hand.set_joint_effort_target(torques)
```

#### 为什么两层控制分离？

| 层 | 空间 | 作用 | 谁决定参数 |
|---|---|---|---|
| **策略层** (Step 1-2) | 位置空间 | 决定"关节往哪转、转多少" | Policy 网络 |
| **执行层** (Step 3-4) | 力矩空间 | 决定"用多大力去跟踪位置目标" | PD gains (仿真器配置) |

**设计意图**：

1. **策略工作在位置空间**：位置命令直观、有界（clamp 到关节限位），策略不需要学习底层力矩动力学。Delta 累积机制保证运动平滑连续。

2. **底层用力矩执行（阻抗控制）**：关节不强制到达位置目标，而是像弹簧-阻尼系统。当手指接触物体时：
   - 力矩 = `p_gain × 位置偏差 - d_gain × 速度` → 偏离目标时产生回复力
   - 如果物体对手指施加反作用力 > 回复力 → 手指被推开 → **自然的力交互**
   - 如果使用位置控制 → 手指无视外力强制到达目标 → 物体被弹飞或穿透

3. **Sim-to-real 的关键**：真机底层是电流/力矩控制。`p_gain/d_gain` 定义了虚拟弹簧的阻抗特性。如果仿真中的阻抗 ≠ 真实执行器阻抗，策略学到的行为无法迁移：
   - 仿真太硬 → 策略学"用力推" → 真机关节太软推不动
   - 仿真太软 → 策略学"大动作" → 真机关节太硬撞坏物体

#### 力矩惩罚中的力矩来源

reward 中的 `torque_penalty` 和 `work_penalty` 使用 `self.torques`——即在 `_apply_action` 中**显式计算**的 PD 命令力矩，**不是** PhysX 的 `applied_torque`。这样确保惩罚的是策略实际发出的指令，而非物理引擎内部的约束反力。

### 控制参数

`p_gain`, `d_gain` 按关节类型分为 7 组，基础值定义在 `Revo3HandHoraEnvCfg.pgain_dict / dgain_dict`，来源于硬件动力学辨识。环境初始化时按关节名匹配分组，生成 `_p_gain_base / _d_gain_base` 两个 21 维 per-DOF 基础向量，再扩展为 `(num_envs, 21)`。

| 分组 | 匹配规则 | 包含关节 | Kp | Kd |
|---|---|---|---:|---:|
| `thumb_CMP` | 名称包含 `CMP` | `right_thumb_CMP_joint` | 16.4 | 0.23 |
| `thumb_CMR` | 名称包含 `CMR` | `right_thumb_CMR_joint` | 0.7 | 0.02 |
| `thumb_flexion` | 名称包含 `thumb` 且包含 `MCP` 或 `PIP` | `right_thumb_MCP_joint`, `right_thumb_PIP_joint` | 1.2 | 0.09 |
| `DIP` | 名称包含 `DIP` | thumb/index/middle/ring/little 的 5 个 DIP | 8.0 | 0.10 |
| `MPR` | 名称包含 `MPR` | index/middle/ring/little 的 4 个 MPR | 0.7 | 0.04 |
| `MCP` | 名称包含 `MCP` | index/middle/ring/little 的 4 个 MCP | 0.6 | 0.014 |
| `PIP` | 其余 finger flexion 关节 | index/middle/ring/little 的 4 个 PIP | 0.8 | 0.027 |

- 分组匹配优先级: `CMP` → `CMR` → `thumb_flexion` → `DIP` → `MPR` → `MCP` → `PIP`
- 随机化: 每 reset 乘以 `Kp × [0.5, 2.0]`, `Kd × [0.5, 2.0]`，每 DOF 独立
- 控制频率: 240Hz 物理 ÷ 12 decimation = 20Hz
- 旋转目标轴: 每个任务由 manifest 的 `rotation.target_axis_world` 独立配置

---

## 4. 奖励 (Reward)

### 奖励公式

| 项 | 公式 | Scale | 作用 |
|---|---|---|---|
| **旋转奖励** | `clip((angvel · target_axis_world), -0.5, 0.5)` | **+2.5** | 鼓励绕该物体配置的世界目标轴旋转 |
| **线速度惩罚** | `‖obj_pos - prev_pos‖₁ / dt` | **-0.3** | 抑制物体平动 |
| **物体位置奖励** | `1 / (‖obj_pos - init_pos‖ + 0.001)` | **+0.003** | 物体保持在初始位置附近 |
| **姿态偏离惩罚** | `Σ (joint_pos - init_joint_pos)²` | **-0.4** | 保持抓取构型, 避免过度伸展 |
| **力矩惩罚** | `Σ torque²` (显式 PD 命令力矩) | **-0.1** | 抑制过大关节力矩 |
| **功惩罚** | `(Σ torque · vel)²` | **-0.5** | 抑制机械功率 |

- 角速度通过四元数差分计算: `dq = q_curr * inv(q_prev)`, `angle = 2*acos(dq.w)`, `axis = dq.xyz / sin(angle/2)`
- **姿态偏离参照**: `self.init_joint_pos` — 来自 `assets.py` 的 `robot_cfg.init_state.joint_pos`，即训练/部署时目标抓取姿态
- **力矩来源**: `self.torques` — 在 `_apply_action` 中显式计算的 PD 命令扭矩，不走 PhysX `applied_torque`

### 总奖励

```
R = 2.5×rotate - 0.3×linvel + 0.003/(d+0.001) - 0.4×pose_diff - 0.1×torque - 0.5×work
```

乘以 0.01 后作为 PPO 优化目标。

---

## 5. 工具

### 抓取缓存生成 (gen_grasp.py)

```
isaaclab.sh -p gen_grasp.py --task ball --num_envs 8192 --headless
```

**流程** (reset_buf 即时重试):
1. 从 `init_joint_pos` + ±0.15 rad 噪声开始
2. 零动作步进, 每步检查门控:
   - **指尖距离**: 5 指尖全部距物体中心 < 10cm
   - **接触**: ≥3/5 elastomer 接触传感器力 > 0.5N
   - **稳定性**: 物体旋转角 < 45°
3. 门控失败 → `reset_buf=1` 即时重试; 通过且 timeout → 收集
4. **重力循环**: 每 40 step 切换重力方向 (±X, ±Y, ±Z)
5. 输出: `cache/revo3_right_grasp_{task}.npy`
   - 格式: `[joint_pos(21), obj_local_xyz(3), quat_xyzw(4)]` × N
6. 目标数量: 512 (可配)
7. 物理设置: 重力循环, 质量 0.10kg (默认,不随机化), 摩擦 U(0.5,2.0)

### 初始位姿可视化 (view_init_pose.py)

```
isaaclab.sh -p view_init_pose.py --task ball --num_envs 1 [--physics] [--cache_file ...]
```

两种模式:
- **冻结模式** (默认): 仅渲染, 检查初始姿态
- **物理模式** (`--physics`): 零动作步进, 每 20 步打印 obj_z / hand_z, 测试被动稳定性

冻结模式可加 `--edit_joints` 打开 21 关节实时编辑面板。每个关节都可通过
滑块或数值框按弧度调整，并按 USD 关节限位约束；修改会同时应用到所有展示
环境。面板可以重置、打印 JSON，扫描物体还可将当前值直接保存到
`assets/usd/objects/manifest.json` 中对应任务的
`grasp_seed.hand_joint_pos_rad`。`--joint_step` 用于设置微调步长。

### ONNX 导出 (tools/export_onnx.py)

```
python tools/export_onnx.py --checkpoint <stage2.ckpt> --output policy.onnx
```

仅支持 **Stage2 / ProprioAdapt** 导出:
- 输入: `obs [B,141]` + `proprio_hist [B,30,47]`
- 输出: `action [B,21]`
- 无 `priv_info` 输入；`priv_info` 只在 Stage2 训练时用于生成 teacher latent target
- 归一化 (`running_mean_std`, `sa_mean_std`) 烘焙进 ONNX 图，部署侧输入原始观测，不要额外归一化
- 动态 batch 维度
- 同时输出 `.deploy_meta.yaml`，包含 IO 形状、关节顺序、动作语义及触觉 `obs` / `proprio_hist` 构造规则

单帧观测布局:

| 偏移 | 维度 | 内容 | 实机输入 |
|---|---:|---|---|
| 0-20 | 21 | `joint_pos_unscaled` | 使用关节限位公式构造 |
| 21-41 | 21 | `cur_targets` | delta 叠加并 clamp 后的位置目标 |
| 42-46 | 5 | DIP 接触力 | 5 路标定后的实测合力 |

`obs` 是 3 帧展平，触觉 slice 为 `[42:47]`, `[89:94]`, `[136:141]`。`proprio_hist` 是 30 帧原始历史，每帧 `[42:47]` 使用相同的 5 路触觉。

### 动作序列导出 (tools/dump_runtime_actions.py)

```
isaaclab.sh -p tools/dump_runtime_actions.py --checkpoint <stage2.ckpt> --task cylinder --frames 200
```

支持 **Stage1 / Stage2**:
- 单环境运行 policy
- 每步记录 cur_targets + jointpos → `.target.txt` + `.jointpos.txt`
- 输出含时间戳/奖励/done 信息, 用于真机回放比对
