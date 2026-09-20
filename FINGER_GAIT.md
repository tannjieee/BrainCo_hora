# 连续旋转与换指

## 2026-09-20：duck 绝对朝向观测与日志同步优化

`train.py --object_orientation auto` 对新 duck Stage1 和 duck 的 `--weights_only`
热启动使用 24 维特权观测：保留原 18 维，追加物体局部 X、Y 轴的世界方向。
该 6D 编码表示绝对姿态而不是相对 reset 的转角，能区分不同初始鸭嘴朝向，
也没有 Euler yaw 的 ±π 跳变或四元数 q/-q 的符号跳变。
141 维本体/触觉观测和 21 维动作不变；Stage1 actor 与 critic 均经过原特权编码器使用新增信息。

旧 checkpoint 的评估、严格续训、Stage2 默认按模型输入宽度自动保留旧格式；
可以用 `--object_orientation none` 明确运行旧观测对照。
18→24 只能在 `--weights_only` 新运行中迁移：首层新增六列初始化为零，
初始 actor 函数保持一致，随后这些列可以学习；critic、优化器和训练计数按热启动规则重置。
不能把这次变更当作保持优化器状态的严格续训。显式 `rotation6d` 搭配旧模型严格续训/评估会报错。
新增格式写入运行元数据；超时终态观测同样包含 reset 前的朝向。

```bash
DUCK_RUN=run_rubber_duck_orientation_s42 DUCK_PHYSICS_HZ=120 \
  bash scripts/duck_gait_large.sh --checkpoint PATH/TO/last.pth \
  --weights_only --object_orientation rotation6d --device cuda:2
```

这条命令会启动一个新实验；已有服务器训练不受本地修改影响。公平比较应另建
`--weights_only --object_orientation none` 对照，两组使用同一不可变 checkpoint、缓存和预算。
不以旧运行严格续训 vs 新运行热启动作为单变量朝向消融。

Stage2 可以按 24 维加载 teacher，但学生仍只使用本体/触觉历史；这不保证学生能推断
不对称物体的绝对 yaw。若蒸馏受可观测性限制，需要另行加入实际可用的视觉/姿态估计输入。

PPO 的环境标量日志改为 GPU 上累计、rollout 结束统一回读，减少每步多次 CPU/GPU 同步。
奖励缩放、回合计数加权的整圈率和净圈数、动作限幅统计保持原定义。
验证覆盖姿态轴顺序、q/-q、±π 连续性、热启动前后 actor 等价性、新列可训练性、旧模型加载、
复用日志缓冲区和真实 Isaac Lab 的超时观测/PPO 更新。没有修改换指奖励权重或宣称收敛改善。

## 2026-09-20：缩短 rollout 与物理频率对照

根据后续指令，正式运行切换为 rollout 16；16384 环境、4000 轮、每 50 轮保存不变。
`--physics_hz 120` 同时设置 `dt=1/120` 与 `decimation=6`，策略频率保持 20 Hz。
`scripts/duck_gait_large.sh` 默认物理频率仍为 240 Hz，只有显式设置
`DUCK_PHYSICS_HZ=120` 才采用经过本次初步对照的较低频率。

`tests/check_duck_physics_rate.py` 对比同一初始抓握与相同动作：每组 512 环境，
5 秒零动作保持及 10 秒早期策略动作回放；两组分别使用标称动力学和训练随机化范围。
随机化组还核对关节 PD、摩擦、质量、重心参数的哈希相同。

| 5 秒保持测试 | 240 Hz 存活率 | 120 Hz 存活率 |
|---|---:|---:|
| 标称动力学，seed 12345 | 88.1% | 87.1% |
| 训练随机化，seed 6789 | 75.2% | 82.6% |

两组均通过预设的相对退化筛选：存活率最多下降 2 个百分点、平均首次掉落时间至少为参考的 95%、
跟踪 RMS 不超过参考的 1.25 倍加 0.0001 rad；接触力波动、接触切换、峰值力及有效帧数也参与检查。
这只证明这批初始状态和早期动作的数值表现没有明显退化；早期策略回放均未存活到 10 秒，
因此不能据此证明长期连续旋转稳定，也不能声称较低频率更接近真实硬件。

保留旧运行目录，在新目录 `run_rubber_duck_gait_h16_p120_16384_i4000_s42_20260920`
从已备份的第 19 轮 checkpoint 续训，保留模型、优化器及归一化状态。
`--max_iterations` 按更新轮数控制剩余预算，避免改变 rollout 后把历史交互数错误换算成更新轮数。
继承的 19,922,944 次交互加上后续 3981 轮，共计划 1,063,518,208 次交互。
新的 checkpoint 记录 rollout、环境数、轮数上限和实际计划总交互数；续训 ETA 与超过 24 小时的显示也已修正。

远端日志：`remote_runs/duck_h16_120hz_16384_4000_20260920/`。
物理频率对照报告：`remote_runs/duck_rate_check_20260920/`。

## 2026-09-20：16384 环境正式训练

按用户新指令停止此前的 2048 环境筛选队列，并保留基线第 50 轮 checkpoint。
新的独立任务使用 `scripts/duck_gait_large.sh` 从头训练：16384 环境、rollout 64、
gamma 0.995、40 秒回合、minibatch 32768、4000 次 PPO 更新，总预算
4,194,304,000 次环境交互。每 50 轮保存编号 checkpoint 并更新 `last.pth`，
TensorBoard 指标仍逐轮记录。固定重力、旋转速度、缓存、奖励和随机化设置沿用长时序方案。

`train.py --max_iterations` 按环境数和 rollout 长度计算总预算，与 `--max_agent_steps` 互斥；
续训时表示总轮数上限而不是追加轮数。`--save_frequency` 显式指定 checkpoint 间隔。

```bash
bash scripts/duck_gait_large.sh --device cuda:2
```

远端正式任务日志：`remote_runs/duck_large_16384_4000_20260920/`。
模型目录：`outputs/revo3_right/run_rubber_duck_gait_long_16384_i4000_s42_20260920/`。

## 2026-09-20：正式缓存与训练对照

服务器已完成 8192 条 5 秒验证缓存，40 个朝向/接触分组全部满额。
`revo3_right_grasp_rubber_duck_gait_v2.npy` 的 SHA256 为
`d1ac6a8939037d9ec1fea193ecf77962438669df2e3b004d78e3abf91ffd4f05`。
这是静态抓握验证，不代表策略已学会动态换指。

`scripts/duck_gait_compare.sh` 提供两组从头训练配置：

| 参数 | baseline | long |
|---|---:|---:|
| 并行环境 | 2048 | 2048 |
| rollout 步数 | 16 | 64 |
| gamma | 0.99 | 0.995 |
| 回合时限 | 20 秒 | 40 秒 |
| minibatch | 32768 | 32768 |
| 环境交互预算 | 20,971,520 | 20,971,520 |
| seed | 42 | 42 |

两组固定 9.81 m/s² 重力、0.5 rad/s 目标速度，使用同一完整缓存和 gait v2 奖励。
学习率、网络、随机化和每份样本的训练轮数沿用原配置。
rollout 增大后每次 PPO 更新的数据量也增大，因此这是长时序配置组合的筛选，不能把效果单独归因于某一个参数。

```bash
bash scripts/duck_gait_compare.sh baseline
bash scripts/duck_gait_compare.sh long
bash scripts/duck_gait_compare.sh eval PATH/TO/last.pth
```

评估统一为 128 环境、seed 12345、60 秒回合及 1200 策略步，使用确定性动作。
训练期间的平均回合奖励因时限不同不能直接排名；比较统一评估的首回合存活率、净转角、整圈率与限位外推。
当前是单随机种子的短程筛选，之后应按结果决定延长训练和增加种子；不自动判定赢家或启动朝向输入实验。

`scripts/run_duck_comparison.py` 可在一张空闲 GPU 上顺序运行两组，每组训练后自动评估。
它拒绝覆盖已有实验，记录命令与源码哈希，并核对实际完成预算、模型数值和评估日志，避免仿真退出码掩盖失败。
两组的 `last.pth` 及 TensorBoard 数据保存在各自独立的输出目录。
`train.py` 新增 `--horizon_length`、`--gamma`、`--episode_length_s`；展开后的参数写入配置，回合时限同时写入 checkpoint 元数据。

## 2026-09-20：第二版实现

此版本针对“初始抓握附近旋转、到限位后掉落”，提供可分组验证的新状态采集器和奖励对照。它没有预先证明鸭子能连续换指，也没有生成完整的新训练数据集。

### 经过物理验证的多样抓握

`gen_grasp.py` 新增 `--yaw_bins`、`--yaw_span_deg`、`--free_fingers`、`--finger_open_fraction`、`--seed_cache`、`--max_steps`。轴旋转候选与空闲手指组合分别设定配额，容易成功的原始姿态不能填满其余分组。`--yaw_bins 1` 保留原始物体朝向；多分组时，实际绕配置的目标世界轴旋转，而非假定所有任务都是世界 Z 轴。

种子缓存只提供关节姿态；物体仍从当前 manifest 的位置、姿态和尺寸生成。为选中的空闲手指降低 MCP/PIP/DIP 弯曲目标，保留侧摆和拇指对掌。候选随后必须通过完整的重力保持验证：稳定接触、物体高度/水平漂移、倾角，以及指定手指确实无接触。所有新姿态先物理验证，绝不直接修改旧缓存四元数后当作有效样本。

保存的是已验证的初始零速度状态，仍兼容 `(N,28)` 缓存格式。成功时同时保存 `.report.json`，包含逐组配额、采集参数、物体元数据和缓存 SHA-256。覆盖不足时只写报告，不写最终 `.npy`；旧缓存默认禁止覆盖。候选在最后一帧失稳，即使达到时限，也不计为成功。某个分组持续无法通过代表当前采样器/约束下未找到解，需要检查几何或改采样参数，不应解除验证来凑数。

### 奖励与指标

- `--finger_gait` 的运行元数据版本升级为 2；开启该模式时，旋转收益在目标速度处最大，超过目标后衰减，到两倍目标时为零。反转仍受罚。`--rotation_speed 0.5` 同步设置稳定旋转和 checkpoint 速度门槛，支持低速对照；此数值是实验起点。
- 支撑门控改用连续帧确认、带滞回的接触状态。至少两个触点仍只是支撑近似，并不证明力闭合；不奖励触点切换次数。
- 持续顶限位的宽限计数只有在目标回撤到距两侧限位至少 0.05 rad 后才清零。暂停或极小回撤不会重新获得宽限期；回撤本身不罚。该指标衡量目标外推，不代表真实机械卡死。
- 新增实际/目标关节接近限位比例、目标跟踪误差、已结束回合净转角。结合整圈完成率和存活时间，区分真实换指进展与单纯放慢前半圈。
- 默认不开启 gait 时保留原奖励和观察/动作维度。v1 gait checkpoint 可测试或用 `--weights_only` 迁移，不能严格续训为 v2。

### 运行入口

先激活兼容的 Isaac Lab 环境。采集和训练是两个独立动作：

```bash
export ISAACLAB_PATH="$HOME/IsaacLab/isaaclab.sh"
bash scripts/duck_gait.sh collect
# 仅在采集成功后执行；脚本检查报告覆盖率与缓存哈希。
bash scripts/duck_gait.sh train
bash scripts/duck_gait.sh eval outputs/revo3_right/run_rubber_duck_s1_gait_v2/stage1_nn/last.pth
```

默认采集 8 个朝向区间 × 5 组（原始、食指/中指/无名指/小指空闲），每个候选保持 5 秒；最多 20000 策略步。不会保证每个组合都有可行解。通过环境变量 `DUCK_ENVS`、`DUCK_CACHE`、`DUCK_RUN`、`DUCK_SPEED` 配置运行，也可以在命令末尾覆盖普通 CLI 参数。小规模训练需同时设置合法的 `--minibatch_size`。

服务器上传与已验证运行环境见 [DEPLOY_BRAINCO.md](DEPLOY_BRAINCO.md)。上传不启动采集或训练。

### 本地验证结果

- 19 项 CPU 单元测试通过，覆盖终帧失稳拒收、配额防偏置、四元数旋转、限位暂停绕过、接触去抖和速度奖励。
- Isaac Lab 集成测试通过：摩擦/重置/timeout 终态、PPO rollout 与反向传播、缓存不变性。
- duck 的 gait v2 模式完成 1024 步冒烟训练，最终 checkpoint 正确保存至最新步数，奖励版本和速度参数写入元数据。
- 32 环境普通采集成功生成 8 条样本；64 环境两组采集成功生成普通抓握 4 条、无名指离开抓握 4 条，均经过 2 秒保持验证。两组采集允许被指定释放的指尖离开接近距离范围，其余指尖仍受约束。
- 多朝向短测试未满足全部配额，正确生成不完整报告并拒绝写出最终缓存；非零退出状态已验证。

这些是流程和数值正确性测试，2 秒冒烟采集不能替代正式 5 秒筛选，1024 步训练也不能证明策略已学会换指。原始模型与 v2 种子缓存未修改；冒烟输出留在本地 outputs，不随代码上传。

## 第一版历史记录

以下记录描述 v1；支撑接触来源、限位计数恢复规则、速度奖励和多样状态采集范围以以上 v2 说明为准。

目的：针对鸭子在固定接触模式下旋转至关节余量耗尽、随后下滑的行为，改善学习信号并提供可比较的诊断。代码验证不等于已经学会连续旋转；仍需要新训练和确定性对照评估。

## 使用与兼容性

新奖励通过 PPO CLI `--finger_gait` 显式开启。默认不改变旧奖励；观察维度仍为 actor 141、privileged 18、动作 21，旧模型仍能回放。新增诊断指标在两种模式下都记录。

本版不增加朝向输入维度，不自动旋转物体来制造可能无效的中间抓握，也不生成新的接触切换状态缓存。这些是后续独立实验，不能视为本次已完成。

建议从新的输出目录进行对照训练。改变 gait 模式/参数时严格续训会拒绝；如需迁移旧策略，显式指定 `--checkpoint ... --weights_only`，重新学习 value，不沿用旧分数门槛。迁移旧的局部策略也可能延续原有局部最优，不保证优于从头训练。

本地小规模运行示例（不会由文档自动执行）：

```bash
conda activate env_isaaclab
cd /home/tan/hora2/BrainCo_hora
~/IsaacLab/isaaclab.sh -p train.py \
  --task rubber_duck --algo PPO --finger_gait \
  --cache_file revo3_right_grasp_rubber_duck_v2.npy \
  --num_envs 256 --minibatch_size 4096 \
  --fixed_train_gravity 9.81 --seed 42 \
  --output_name run_rubber_duck_s1_gait_v1 --headless
```

云端大规模训练仍需先同步此次代码；本次修改不自动部署、不自动启动训练。

## 奖励变化

保留原始奖励，追加：

`Δr = max(r_rotate_normalized, 0) × (support_gate − 1) × rotate_scale − 2 × blocked_push + 5 × new_full_turns × valid_gate`

所有数值为 PPO 乘 0.01 之前的环境奖励。

- **支撑门控**：至少 2 个物体接触指尖（原始过滤力 ≥0.05 N），允许其他手指暂时离开。高度偏移 ≤5 mm 时高度系数为 1，至 20 mm 线性降至 0；向下速度由 0 至 0.05 m/s 时速度系数由 1 降至 0。两系数相乘；掉落帧为 0。只衰减正向旋转收益，不削弱反向旋转惩罚。接触数只表示有支撑的近似条件，并不证明力闭合。
- **持续无效外推**：请求目标超出关节控制限位且方向继续向外时逐关节累计持续步数。前 4 步不罚；之后按被裁剪位移/动作尺度的平方（上限 1）取关节均值，乘 −2。停在限位但不继续外推、向内回撤，都不会因此受罚。不使用“回到初始抓握”的姿态惩罚。
- **新整圈奖励**：以相邻实际四元数的最短旋转增量在目标世界轴上的投影累积有符号转角。每个回合首次达到新的正向完整圈数时奖励一次，并要求支撑 gate 有效且倾角合格。不通过角速度乘回合时长估算，不因四元数正负号翻转或跨 ±180° 跳变而误计圈数；退回再前进到旧圈数不会重复奖励。在无效状态跨过的圈数也更新 high-water，不会之后补领奖励。

该累计量是沿目标轴投影的旋转路径积分，不是任意倾斜下的 Euler yaw；每个控制步必须小于 π，当前 20 Hz、约 1 rad/s 的任务满足该采样假设。不要把平均 `net_turns` 当作整回合最终圈数。

所有新奖励系数是待验证的第一版参数，并无长训练成功保证。没有强制每一步切换接触，也不直接奖励触点变化，防止抖动或无意义松指刷分。

## 指标

- `gait/net_turns`：当前回合有符号累计转角/2π，按环境取均值。
- `gait/new_full_turns`：本步新跨过的正向整圈 high-water。
- `gait/completed_one_turn_rate`、`gait/completed_mean_turns`：PPO rollout 内按已结束回合数加权统计，而非各步比例简单平均；没有回合结束时不输出该比率。
- `gait/contact_{finger}`：逐指去抖接触率。on 阈值为接触阈值，off 为一半；需要连续 2 帧确认切换。
- `gait/releases`、`gait/recontacts`：释放与重新接触次数；初始首次建立接触不算重新接触。reset 清理接触历史。
- `gait/blocked_{joint}`：逐关节持续外推超限比例。
- `gait/support_gate`、`gait/blocked_push`、`rew/gait_adjustment`：门控与奖励修改诊断。
- 有限 `--test --test_steps 400` 额外输出首次掉落时间（按测试长度截断）、首回合 5/10/20 秒存活率，以及结束回合圈数。

诊断状态只在物理步奖励计算时推进，在 reset 时按 env_id 清零；观察获取和 timeout 终态快照不推进转角/接触诊断。普通接触观察的噪声与延迟逻辑仍只由原观察采样路径负责。

## 保存与缓存保护

PPO 达到步数预算后额外保存最新 `last.pth`，避免 1145 轮结束时文件仍停在 1100 轮。checkpoint 通过同目录临时文件加原子替换写入。

测试现在校验 checkpoint 中的物体姿态/缩放/轴、手指种子及缓存 SHA-256。鸭子 v2 回放必须传 `--cache_file revo3_right_grasp_rubber_duck_v2.npy`；发现默认旧缓存与新模型混用会报错，而非静默运行。对缺少 env_runtime 的旧 checkpoint 仍保留兼容，不能提供同等级别的一致性保证。

## 验证命令

```bash
python -m unittest discover -s tests -p test_stage1_core.py -v
python -m unittest discover -s tests -p test_finger_gait.py -v
~/IsaacLab/isaaclab.sh -p tests/check_stage1_integration.py
```

`tests/test_initial_pose_editor.py` 是独立 Isaac GUI 集成脚本，不要用普通 unittest 的 `test_*.py` 全目录发现方式启动它。
