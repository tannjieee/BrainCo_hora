# `sim-trace` 使用示例：导出固定初始姿态的 20 帧仿真 Trace

本文说明如何使用 `scripts/sim2real.sh sim-trace` 生成下面这份可复现的仿真记录：

```text
outputs/revo3_right/traces/sim_joint_order_latent_cache7942_20f.npz
```

该流程只启动 Isaac Lab 仿真，不连接 Revo3 真机，也不发送电机命令。

## 1. 这个命令做什么

此示例使用 cylinder 的 Stage 2 `ProprioAdapt` checkpoint，从抓握缓存第 7942 行初始化单个
仿真环境，并记录 20 个 policy step。策略频率为 20 Hz，因此记录总跨度约 1 秒，时间轴为
0–0.95 秒。

同时传入 ONNX 文件后，程序会对每一帧使用同一份原始 observation/history，比较 checkpoint
输出与 ONNX 输出。这是推理一致性检查；仿真环境中的动作仍来自 checkpoint。

调用关系为：

```text
scripts/sim2real.sh sim-trace
└── tools/dump_runtime_actions.py --num_envs 1 --episodes 1 --headless
```

后面三个参数由 wrapper 自动追加，保证 NPZ 只包含一个无歧义的单环境 episode。因此 NPZ
的 `metadata_json.command` 记录的是实际执行工具 `tools/dump_runtime_actions.py`。

## 2. 前置条件

从仓库根目录 `/home/tan/hora/BrainCo` 执行本文命令。先检查仿真和部署 Python 环境：

```bash
scripts/sim2real.sh env-check
```

确认以下输入文件存在：

```text
outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt
outputs/revo3_right/onnx/cylinder_stage2.onnx
cache/revo3_right_grasp_cylinder.npy
```

`scripts/sim2real.sh` 默认使用：

```text
SIM_PY=/home/tan/miniconda3/envs/env_isaaclab/bin/python
REAL_PY=/home/tan/miniconda3/envs/revo3/bin/python
```

如果本机路径不同，可以在命令前设置绝对路径形式的 `SIM_PY` 或 `REAL_PY`。

## 3. 完整生成命令

```bash
scripts/sim2real.sh sim-trace \
  --task cylinder --algo ProprioAdapt \
  --checkpoint outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt \
  --onnx outputs/revo3_right/onnx/cylinder_stage2.onnx \
  --cache-row 7942 --max_frames 20 \
  --trace-npz outputs/revo3_right/traces/sim_joint_order_latent_cache7942_20f.npz \
  --output_dir outputs/revo3_right/action_dump/joint_order_latent_cache7942
```

当前仓库中该 NPZ 已经存在，而导出工具会拒绝覆盖已有 trace。再次实验时请保留旧文件并修改
输出名，例如把 `20f.npz` 改为 `20f_v2.npz`，同时为 `--output_dir` 使用新的目录。

## 4. 参数说明

| 参数 | 本例取值 | 作用 |
|---|---|---|
| `sim-trace` | — | 运行单环境、单 episode、headless 的仿真 trace 导出 |
| `--task` | `cylinder` | 使用圆柱在手旋转任务及其抓握缓存 |
| `--algo` | `ProprioAdapt` | 加载 Stage 2 student/adaptation policy |
| `--checkpoint` | `model_best_latent.ckpt` | 产生仿真动作的 checkpoint |
| `--onnx` | `cylinder_stage2.onnx` | 逐帧检查 checkpoint 与 ONNX 动作一致性 |
| `--cache-row` | `7942` | 固定 env 0 的抓握缓存初始行，便于复现 |
| `--max_frames` | `20` | 最多记录 20 个 policy step |
| `--episode-length-s` | 默认 `20` | 仅为本次导出覆盖 episode 时限；长轨迹应设置得略长于 `max_frames / 20` |
| `--trace-npz` | `...20f.npz` | 统一、带元数据的逐帧 trace 输出 |
| `--output_dir` | `...cache7942` | 三个便于人工阅读的 TXT 输出目录 |

未显式传入 `--seed` 时默认值为 42。固定 cache row 并不关闭 domain randomization；质量、
质心、摩擦、PD 增益和输入关节噪声等随机量仍由 seed 决定。

## 5. 输出文件

### 统一 NPZ

```text
outputs/revo3_right/traces/sim_joint_order_latent_cache7942_20f.npz
```

常用数组如下：

| 数组 | 形状 | 单位/含义 |
|---|---:|---|
| `sample_time_s` | `[20]` | 从第 0 帧开始的时间，s |
| `policy_pos_rad` | `[20, 21]` | 仿真实际关节角，policy order，rad |
| `policy_target_rad` | `[20, 21]` | 动作积分并限位后的目标关节角，rad |
| `action` | `[20, 21]` | clip 到 `[-1,1]` 的增量动作 |
| `force_n` | `[20, 5]` | 网络实际接收的五指接触量：物理合力模长 N × `contact_force_scale` |
| `metadata_json` | scalar | checkpoint/ONNX SHA、关节顺序、seed、采样率等来源信息 |

21 个关节的列名读取 `metadata_json.joint_order`。五指力读取
`metadata_json.contact_order`，本例顺序为：

```text
thumb_DIP, index_DIP, middle_DIP, ring_DIP, little_DIP
```

### 人工可读 TXT

`--output_dir` 中会生成带运行时间戳的三个文件：

```text
ep00_cylinder_proprioadapt_<timestamp>.raw_action.txt
ep00_cylinder_proprioadapt_<timestamp>.target.txt
ep00_cylinder_proprioadapt_<timestamp>.jointpos.txt
```

- `raw_action.txt`：checkpoint 原始的 21 维增量动作；
- `target.txt`：增量动作积分并限位后的目标角；
- `jointpos.txt`：仿真的实际关节角。

## 6. 判断导出是否成功

成功时终端末尾应包含类似输出：

```text
[OK] Episode 1/1: 20 frames -> ep00_cylinder_proprioadapt_<timestamp>.{raw_action,target,jointpos}.txt
[OK] Unified policy trace: .../sim_joint_order_latent_cache7942_20f.npz
[OK] Simulator raw obs/history -> checkpoint vs ONNX max_abs_error=<value>
```

最后的 `max_abs_error` 越接近 0 越好。本文对应的现有 NPZ 中，逐帧比较
`checkpoint_action` 与 `onnx_action_raw` 得到的最大绝对误差为 `9.536743e-07`。若重新生成后
误差明显变大，先不要使用该 ONNX 做真机部署，应检查 checkpoint、ONNX 导出版本和归一化
契约。

## 7. 绘制关节角和五指力曲线

```bash
/home/tan/miniconda3/envs/revo3/bin/python tools/plot_policy_trace.py \
  outputs/revo3_right/traces/sim_joint_order_latent_cache7942_20f.npz \
  --output-dir outputs/revo3_right/traces/plots
```

生成：

```text
sim_joint_order_latent_cache7942_20f.joint_angles.png
sim_joint_order_latent_cache7942_20f.fingertip_forces.png
```

关节图默认以度为单位，并在每个关节的小图中同时显示实际角度与目标角度。追加
`--angle-unit rad` 可改用弧度；在有图形界面的环境中追加 `--show` 可保存后打开窗口。

## 8. 离线校验 Trace（可选）

下面的命令只读取、校验并打印 trace，不连接硬件：

```bash
scripts/sim2real.sh replay \
  --trace-npz outputs/revo3_right/traces/sim_joint_order_latent_cache7942_20f.npz \
  --checkpoint outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt \
  --profile deploy/revo3/config/revo3_right.yaml \
  --frames 20 --all-joints
```

不要添加 `--preflight` 或 `--execute`；这两个选项会进入真机只读检查或显式确认后的硬件回放
流程，不属于本示例的纯离线范围。

需要将 `policy_pos_rad` 按 20 Hz 回放到 Revo3 真机时，请按
[`docs/replay_trace/README.md`](../replay_trace/README.md) 的独立流程执行。
