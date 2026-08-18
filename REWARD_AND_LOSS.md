# Stage 1 奖励函数与 PPO Loss

本文档对应当前 `Revo3HandHoraEnv` 和 `PPO` 实现。环境以 20 Hz 输出原始单步奖励，PPO 在计算 GAE 前统一乘以 `reward_scale = 0.01`。

## 1. 记号

- `ω`：物体世界坐标系角速度，单位 rad/s。
- `a`：任务配置的世界目标旋转轴单位向量。`sapota_planter` 使用 `(0, 0, 1)`。
- `ω_target = ω · a`：目标轴上的有符号角速度。
- `p`、`p0`：物体当前位置和本回合 reset 时的位置。
- `d_xy = ||(p - p0)_xy||`，`d_z = |p_z - p0_z|`。
- `tilt`：物体配置轴与世界目标轴之间的夹角。
- `drop`：物体高度超出 reset 高度 ±20 mm 时为 1，否则为 0。

Huber 型归一化函数：

```text
h(x; s) = 0.5 (|x|/s)^2          当 |x|/s < 1
          |x|/s - 0.5            其他情况
```

## 2. 环境总奖励

```text
r_raw = r_rotate
      + r_stable_rotation
      + r_alive
      + r_tilt
      + r_off_axis
      + r_xy
      + r_z
      + r_drop
      + r_self_collision
      + r_torque
      + r_work

r_PPO = 0.01 * r_raw
```

缓存抓取关节姿态只用于 episode reset，不参与奖励。当前没有默认姿态或缓存姿态奖励/惩罚。

### 2.1 有符号目标轴旋转

```text
progress = clip(ω_target / 1.0, -1, 1)
r_rotate = 10.0 * progress
```

- `+1 rad/s` 得 `+10`。
- `-1 rad/s` 得 `-10`。
- 超过 ±1 rad/s 后不再增大奖励绝对值。
- 正反振荡的正负奖励会互相抵消。

### 2.2 稳定旋转 bonus

以下条件同时成立时 `stable = 1`：

```text
ω_target >= 0.5 rad/s
tilt <= 10 deg
d_xy <= 10 mm
d_z <= 5 mm
drop == 0
```

```text
r_stable_rotation = 0.5 * stable
```

### 2.3 存活与掉落

```text
r_alive = 0.2 * (1 - drop)
r_drop  = -20.0 * drop
```

### 2.4 物体轴倾斜

```text
r_tilt = -1.0 * (tilt / 10deg)^2
```

对于配置为双向轴的物体，轴正反方向视为等价；`sapota_planter` 使用有方向的轴对齐。
因此花盆局部 `+Z` 必须对齐世界 `+Z`。在 5°、10°、20° 时，该项缩放前
分别为 `-0.25`、`-1.0`、`-4.0`，从而抑制依靠倾斜换取旋转速度的策略。

### 2.5 非目标轴角速度

```text
ω_axis = (ω · a) a
r_off_axis = -0.5 * ||ω - ω_axis||^2
```

该项主要阻尼世界 X/Y 方向的角速度，减少花盆轴相对世界 Z 轴的动态晃动。
训练日志中的 `axis_aligned_rate` 统计倾角不超过任务容差（花盆为 10°）的
环境比例；它比单独观察平均倾角更能反映姿态是否持续稳定。

### 2.6 XY 平面漂移

10 mm 内完全免罚：

```text
xy_excess = relu(d_xy - 0.010)
r_xy = -0.15 * h(xy_excess; 0.005)
```

例如：

| XY 漂移 | `r_xy`（PPO 缩放前） |
|---:|---:|
| 5 mm | 0 |
| 10 mm | 0 |
| 12.5 mm | -0.01875 |
| 15 mm | -0.075 |
| 20 mm | -0.225 |
| 25 mm | -0.375 |

### 2.7 Z 方向漂移

Z 方向没有免罚区，5 mm 是 Huber 的二次/线性过渡尺度：

```text
r_z = -0.25 * h(d_z; 0.005)
```

### 2.8 手部自碰撞

对21个可动手指链刚体分别建立过滤接触传感器，只保留它们与其他手部刚体的法向接触，不包含指尖—物体接触。令 `F_self` 为所有这些接触对中的最大法向力：

```text
F_excess = relu(F_self - 0.5 N)
r_self_collision = -1.0 * h(F_excess; 5.0 N)
```

0.5 N 以下视为接触数值噪声，不惩罚。TensorBoard 同时记录：

- `self_collision_rate`
- `self_collision_force_mean_n`
- `self_collision_force_max_n`
- `rew/self_collision`

### 2.9 力矩与机械功率

当前仿真执行器限制为 1 Nm。对21个关节取平均：

```text
τ_norm = τ / 1 Nm
r_torque = -2.0 * mean(τ_norm^2)
r_work   = -0.1 * mean(abs(τ_norm * q_dot))
```

力矩来自策略目标经过显式 PD 计算得到的命令力矩，不是 PhysX 约束反力。机械功率逐关节取绝对值后再求平均，正负功率不会相互抵消。

## 3. 日志缩放规则

`PPO.play_steps()` 对名字以 `rew/` 开头的环境指标再乘 `reward_scale = 0.01` 后写入 TensorBoard。因此：

- `rew/rotate`、`rew/xy_drift` 等显示的是 PPO 缩放后的单步分量。
- `total_reward` 是环境原始单步奖励，没有乘 0.01。
- `episode_rewards/step` 是缩放后的 episode return。
- `episode_rewards_raw/step` 是未缩放的 episode return。

## 4. GAE 与回报

```text
δ_t = r_t + γ V(s_{t+1}) (1-d_t) - V(s_t)
A_t = δ_t + γ λ (1-d_t) A_{t+1}
R_t = A_t + V(s_t)
```

当前参数：

| 参数 | 数值 |
|---|---:|
| `reward_scale` | 0.01 |
| `gamma` | 0.99 |
| `tau / lambda` | 0.95 |
| rollout horizon | 16 steps（0.8 s） |
| advantage normalization | 开启 |
| value normalization | 开启 |
| timeout value bootstrap | 开启 |

## 5. PPO Actor Loss

令：

```text
ratio = π_new(a|s) / π_old(a|s)
```

当前实现：

```text
surr1 = A * ratio
surr2 = A * clip(ratio, 1-epsilon, 1+epsilon)
L_actor = mean(max(-surr1, -surr2))
```

`epsilon = 0.2`。

策略分布是逐动作维度独立的 Normal 分布：

```text
a_raw ~ Normal(mu(s), sigma)
```

训练环境实际接收 `clip(a_raw, -1, 1)`，TensorBoard 的 `policy/action_saturation_rate` 用于监控原始采样超出范围的比例。当前 bounds loss 只约束策略均值，没有完全消除采样动作裁剪与 log-prob 之间的差异。

## 6. Critic Loss

```text
V_clip = V_old + clip(V_new - V_old, -0.2, 0.2)
L_value = mean(max((V_new - R)^2, (V_clip - R)^2))
```

Actor 和 critic 当前共享 `actor_mlp` 主干，末端分别使用 `mu` 和 `value` 线性层。

## 7. Entropy 与动作边界 Loss

```text
L_bounds = mean(sum(relu(abs(mu) - 1.1)^2))
H = mean(Normal(mu, sigma).entropy())
```

动作 log standard deviation 被限制在：

```text
-2.3 <= log(sigma) <= -0.7
```

## 8. 最终优化目标

```text
L_total = L_actor
        + 0.5 * critic_coef * L_value
        - entropy_coef * H
        + bounds_loss_coef * L_bounds
```

当前系数：

| 参数 | 数值 |
|---|---:|
| `critic_coef` | 1.0 |
| `entropy_coef` | 0.0001 |
| `bounds_loss_coef` | 0.001 |
| gradient norm clip | 1.0 |
| mini epochs | 3 |
| minibatch size | 32768 |
| base learning rate | 3e-4 |
| weights-only warm-start LR | 1e-4 |
| adaptive KL target | 0.01 |

学习率按每个 mini epoch 的平均 KL 自适应：

- `KL > 2 * target`：学习率除以 1.5。
- `KL < 0.5 * target`：学习率乘以 1.5。
- 学习率限制在 `[1e-6, 1e-2]`。

## 9. 满重力 best checkpoint 条件

以下条件连续满足25个 PPO epoch 后，策略才有资格更新 `best.pth` 和 `best_full_gravity.pth`：

```text
gravity >= 9.79 m/s^2
gravity_reset_rate_window <= 0.003
mean target_angvel >= 0.5 rad/s
stable_rotation_rate >= 0.30
```

满足门槛后仍按 episode mean reward 选择更优 checkpoint。`best_curriculum.pth` 只用于诊断，不应直接作为 Stage 2 teacher。

## 10. 有限步测试输出

使用 `--test --test_steps N` 时，终端汇总以下关键指标：

- 目标轴平均角速度和稳定旋转率。
- 平均倾角。
- XY/Z 平均漂移。
- 自碰撞率和每环境最大自碰撞力的均值。
- 高度 reset 和 timeout 数量。
