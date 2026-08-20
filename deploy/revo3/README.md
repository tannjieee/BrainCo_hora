# Revo3 tactile HORA deployment

这个目录是当前 `cylinder_stage2.onnx` 的独立 Python 部署运行时。它复用了
RevoLab 的分层思路，但已按本工程的真实接口重写：

```text
obs[1,141] + proprio_hist[1,30,47] -> action[1,21]
```

每个 47 维帧严格为：

```text
[q_unscaled(21), current_target_rad(21), fingertip_force_N(5)]
```

运行时维护同一个按时间从旧到新排列的 30 帧缓冲；`obs` 使用最后 3 帧，
`proprio_hist` 使用全部 30 帧。ONNX 已包含两套 RunningMeanStd，部署侧不再归一化。

## 网络输入、输出与缩放对齐

当前已锁定的数学契约如下。metadata 现在保存 21 轴未缩放和乘 `0.9` 后的数值限位；部署
加载时会逐轴与 profile 比较，不再只比较一个 `scaled_by: 0.9` 标量。

| 项目 | 训练端 | 部署端 |
|---|---|---|
| 关节输入 | policy joint order，rad | SDK M0–M20 的绝对角度先 `deg→rad`，再重排到 policy order |
| `q_unscaled` | `(2q-upper-lower)/(upper-lower)` | 同式，使用 metadata 固化的 `0.9 × USD/URDF limits`；不额外 clip |
| target 通道 | 上一周期 `cur_targets`，rad | 上一周期积分目标，rad；不是 `q_unscaled` |
| 触觉通道 | `max(0, 0.1×norm(F)+N(0,0.01²))` | VisionTouch 先输出同顺序的 N 值，input builder 再按 metadata 确定性乘 `0.1`，不加噪声 |
| `obs` | `t-2,t-1,t` 三个 47 维帧按旧到新展平 | 相同，shape `[1,141]` |
| `proprio_hist` | `t-29...t` 按旧到新 | 相同，shape `[1,30,47]` |
| 归一化 | `running_mean_std` / `sa_mean_std` | 已烘焙在 ONNX 内，部署禁止二次归一化 |
| action | deterministic `mu`，clip `[-1,1]` | ONNX 输出后再次做有限值/范围保护 |
| action 缩放 | `target += action/24` rad，20 Hz | 相同；满幅每周期 `2.387324°` |

使用当前 checkpoint 的 PyTorch wrapper 与当前 ONNX，对 cache 第 7942 行的真实布局输入和
20 组随机 batch 做数值比较，最大绝对输出误差为 `1.64e-6`。因此 checkpoint→ONNX、张量
shape、RMS 和 action 输出已经数值对齐。

仍未完成的是两项 sim-to-real 标定，而不是 ONNX 导出：

1. `sim2real_joint_offset` 当前 21 项全零，代表假设 SDK 绝对角度与仿真关节坐标零点一致。
   SDK 的硬件零位不会被程序修改，但这个软件坐标假设仍需用已知机械姿态逐轴核对。
2. VisionTouch 当前 `gain=1`、`bias=0`。单位和手指顺序已对齐，但实机 Force6D 模长与
   Isaac object-filtered contact force 的尺度、零点和非目标接触仍需用已知载荷标定。

`first_delta_gate=PASS` 只证明首条命令相对当前测量值连续；因为 reset 时 target 初始化为
当前关节位置，它**不能**证明当前手势与某一 cache 姿态相同，也不能证明软件 offset 正确。

可用下面的完全离线命令打印逐轴 `policy index ↔ SDK motor`、训练限位、命令限位、offset、
第 7942 行位置和实际 `q_unscaled`：

```bash
/home/tan/miniconda3/envs/revo3/bin/python \
  deploy/revo3/scripts/validate_policy.py \
  --onnx outputs/revo3_right/onnx/cylinder_stage2.onnx \
  --metadata outputs/revo3_right/onnx/cylinder_stage2.deploy_meta.yaml \
  --profile deploy/revo3/config/revo3_right.yaml \
  --steps 1 \
  --joint-pos-npy cache/revo3_right_grasp_cylinder.npy \
  --joint-pos-row 7942 \
  --contact-forces 0 0 0 0 0 \
  --print-alignment
```

如果要把历史缓冲的真实时间轴也纳入 checkpoint↔ONNX 对比，可在能访问 NVIDIA GPU 的
本机终端运行 10 帧 Isaac Lab 金标准测试：

```bash
~/IsaacLab/isaaclab.sh -p tools/dump_runtime_actions.py \
  --task cylinder \
  --algo ProprioAdapt \
  --checkpoint outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt \
  --onnx outputs/revo3_right/onnx/cylinder_stage2.onnx \
  --num_envs 1 --episodes 1 --max_frames 10 --headless \
  --output_dir /tmp/hora_alignment_dump
```

最终一行的 `Simulator raw obs/history -> checkpoint vs ONNX max_abs_error` 应接近 `1e-6`。
该命令只运行仿真，不连接真机。

## 导出策略目标或仿真实测角并在真机做短时 replay

关节顺序诊断使用仿真 trace 里的 `policy_target_rad[T,21]`：它是 policy order、rad 单位的
绝对位置目标。不要把 `action` 或 `.raw_action.txt` 直接发给电机；action 只是
`[-1,1]` 的增量，必须依赖上一帧 target 做 `target += action/24`。也不要直接按 21 列发送
`.target.txt`，因为 policy order 与 SDK 的 M0–M20 order 不同。

先固定一个 grasp-cache 行并只导出 20 帧，以便得到可复现的策略增量。后面的
`--anchor-current` 关节映射模式会把这些增量锚定到真机实时姿态，因此不要求手工把真机摆成
cache 的绝对姿态：

```bash
scripts/sim2real.sh sim-trace \
  --task cylinder --algo ProprioAdapt \
  --checkpoint outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt \
  --onnx outputs/revo3_right/onnx/cylinder_stage2.onnx \
  --cache-row 7942 --max_frames 20 \
  --trace-npz outputs/revo3_right/traces/sim_joint_order_latent_cache7942_20f.npz \
  --output_dir outputs/revo3_right/action_dump/joint_order_latent_cache7942
```

然后离线校验 artifact，并打印每个 `Pidx -> SDK Midx` 的第一帧、范围和最大逐帧变化。下面
选中 `P05=right_index_MCP_joint`，按当前 profile 应映射到 `M13`：

```bash
scripts/sim2real.sh replay \
  --trace-npz outputs/revo3_right/traces/sim_joint_order_latent_cache7942_20f.npz \
  --checkpoint outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt \
  --profile deploy/revo3/config/revo3_right.yaml \
  --frames 10 --joint P05
```

这一步默认完全离线。加载器会核对 trace schema/source、checkpoint SHA256、21 维 joint
order、20 Hz、`action_scale=1/24`、数组有限值、连续 step/time，以及每帧
`target_before + action/24 -> clip -> policy_target` 的数学关系；terminal/reset 行不会进入
硬件 replay。

若要检查仿真中的无噪声实测关节角 `policy_pos_rad[T,21]`，显式增加
`--trajectory-source measured`。它仍按 metadata 中的 policy order 读取，再由 profile 重排到
SDK M0–M20；不会把 NPZ 的列顺序直接发给电机。例如下面只做 200 帧离线检查：

```bash
scripts/sim2real.sh replay \
  --trace-npz outputs/revo3_right/traces/sim_joint_order_latent_cache7942_200f.npz \
  --profile deploy/revo3/config/revo3_right.yaml \
  --trajectory-source measured --frames 200 --all-joints
```

当前 profile 已绑定 USB adapter
`usb-Prolific_Technology_Inc._USB-Serial_Controller_APACb111216-if00-port0`、5 Mbps 和
Modbus slave 127，因此正常宿主机终端不再执行全端口自动扫描。若在隐藏宿主机 `/dev` 的
沙箱/容器中运行，仍需把该设备节点映射进去；sysfs 能看到 `ttyUSB0` 并不代表容器内存在
可打开的 `/dev/ttyUSB0`。

普通 measured replay 仍使用 2°/s 插值以及 source-step、10° 行程和 60 秒计划门限。若要按
NPZ 的原始 20 Hz 直接回放 measured 端点，使用 `--recorded-rate`。该模式取消上述诊断门限及
0.05° inward margin，但保留设备静态/实时硬限位、逐电机电流、在线、通信、跟踪误差和除
显式忽略 Stall 外的 fault 检查。

当前真机张开姿态到轨迹第 0 帧的最大差值约 72.1°，不能作为第一条阶跃命令发送。
`--preposition-to-first` 会先按 `--max-speed-deg-s` 从 fresh 真机姿态插值到第 0 帧，再开始
原始 20 Hz 时钟。2026-08-12 的只读 preflight 结果为 920 tick / 46.0 秒，其中约 36 秒预定位，
后 10 秒为 200 帧 recorded-rate replay：

```bash
scripts/sim2real.sh replay \
  --trace-npz outputs/revo3_right/traces/sim_joint_order_latent_cache7942_200f.npz \
  --checkpoint outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt \
  --profile deploy/revo3/config/revo3_right.yaml \
  --trajectory-source measured --recorded-rate --preposition-to-first \
  --max-speed-deg-s 2 --frames 200 --all-joints \
  --port /dev/ttyUSB0 --preflight
```

21 轴 motion 还要求 `--execute`、`--confirm-full-hand`、`--confirm-recorded-rate`、
`--confirm-preposition` 和普通的物理确认。`--ignore-all-stall` 只屏蔽 Stall `0x100`；掉线、
其他 fault、硬限位、电流、通信和 release 逻辑仍然有效。

下一步只读连接真机，检查设备身份、21 路在线/fault/电流、静态与实时限位，并打印当前
`M13` 位置到 current-anchor 第一目标的差值；不会调用 MIT 写接口：

```bash
scripts/sim2real.sh replay \
  --trace-npz outputs/revo3_right/traces/sim_joint_order_latent_cache7942_20f.npz \
  --checkpoint outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt \
  --profile deploy/revo3/config/revo3_right.yaml \
  --frames 1 --joint P05 --anchor-current \
  --allow-passive-limit-outlier --preflight
```

`--anchor-current` 使用所选起始行的 `target_before_policy_rad` 作为仿真基准，并在真正执行前
再次读取真机姿态：`live_target[t] = fresh_live_start + (trace_target[t] -
trace_target_before[first])`。它保留第一帧策略动作，但不追赶 cache 的绝对角度。当前示例
P05 第一帧是约 `+2.387°`，因此 preflight 会打印“真机当前 M13 + 2.387°”的目标。

只有 `first_delta_gate=PASS` 且操作者确认打印的 P/M 映射、物理手指与运动方向后，才做单轴
低增益 replay。工具把每个原始 target 保留为精确路径端点，在端点之间按 20 Hz、默认/最高
2°/s 插值；因此电流、Stall、在线状态、跟踪误差和被动轴在整个慢放期间仍每 50 ms 检查，
末条命令后还会额外做一次 health read。只有选中轴获得非零 Kp/Kd，其余轴保持零增益；
current-anchor 模式的被动移动门限为 1°（绝对模式为 5°）。该诊断路径把电流限制为每轴最多
500 mA、选中轴合计最多 1000 mA，不继承
policy motion 的 10 A 阈值。只有显式 `--ignore-all-stall` 可以忽略 Stall `0x100`，其他检查
不变。第一次只执行一帧：

```bash
scripts/sim2real.sh replay \
  --trace-npz outputs/revo3_right/traces/sim_joint_order_latent_cache7942_20f.npz \
  --checkpoint outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt \
  --profile deploy/revo3/config/revo3_right.yaml \
  --frames 1 --joint P05 --anchor-current \
  --allow-passive-limit-outlier \
  --kp 0.2 --kd 0.05 --max-speed-deg-s 2 \
  --execute --allow-unverified-calibration \
  --confirm-fixed --confirm-clear-path --confirm-estop \
  --confirm-release --confirm-mapping --confirm-large-excursion \
  --confirm-current-anchor
```

current-anchor 模式只允许选择一个关节，相对 fresh 起点的最大 command 偏移硬限制为 3°，
实测偏移超过 3.5°会立即 release；超过 1°还必须追加 `--confirm-large-excursion`。首条命令
必须在 fresh read 所在的 50 ms 周期内发出，否则不下发。目标超出 profile 与设备实时限位的
向内交集会直接拒绝，绝不
silent clip。未选中轴保持 `Kp=Kd=effort=0`，并继续接受在线、fault、电流和 1°被动位移监测。
如果绝对编码器读数越过设备上报限位，必须显式给出 `--allow-passive-limit-outlier`，且最多只
接受 5°；对应的被动 command slot 会夹到设备实时限位内，但仍是零增益、零电流。这个例外
绝不适用于选中轴。单次插值计划最多 60 秒。任意异常或正常结束都会请求零力 release，因此
必须提前固定手、清空运动路径并准备硬件急停；release 后灵巧手会变软。

不带 `--anchor-current` 时仍是绝对 sim-target replay；该模式才要求真机预先接近相同 cache
姿态，并保留原有 5°首跳与 10°总行程门限。

建议一次只测一个 policy joint，并记录“预期 joint、打印的 SDK motor、实际运动部位与方向”。
`--joint` 也接受 `M13`、`P05` 或完整 joint name，执行时最多可重复四次；`--all-joints`
执行仅限显式选择 measured 轨迹并增加 `--confirm-full-hand`，同时仍受全部数值安全门限约束。
尤其要逐轴核对拇指
M16–M20：profile 的仿真关节命名与 SDK 文档的物理/差动关节名称不是一一同名，配置内部
round-trip 通过不能替代目视实测。

该工具是开环 target playback：它不读取触觉、不执行 ONNX，也不会根据真机状态重新计算
action。因此它只用于短时低增益的关节映射诊断，不能用来判断策略的真实闭环效果。

## 自动化关节顺序测试 session

`joint-order` 会从 trace 的实际 applied delta（`policy_target - target_before`）自动为 21 个
policy joint 选择单帧测试动作，并把进度、artifact SHA 和人工观察持久化到 JSON session。
计划和状态查询完全离线：

```bash
scripts/sim2real.sh joint-order init \
  --trace-npz outputs/revo3_right/traces/sim_joint_order_latent_cache7942_20f.npz \
  --checkpoint outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt \
  --profile deploy/revo3/config/revo3_right.yaml \
  --session outputs/revo3_right/joint_order/session_cache7942.json
```

当前 20 帧 trace 在默认 `1.0°–2.5°` 工作窗内为全部 21 轴找到了候选：20 轴的主候选约
`2.387°`，P04/M19 `right_thumb_CMP_joint` 为 `1.688°`。`status` 会显示主 row、正反向备选
和下一个仍为 `planned` 的 Pidx：

```bash
scripts/sim2real.sh joint-order status \
  --session outputs/revo3_right/joint_order/session_cache7942.json
```

每次 `probe` 最多处理一个显式指定的关节。它会先运行一次独立的只读 preflight 并关闭连接，
然后从 `/dev/tty` 要求输入绑定设备 SN、Pidx、Midx、row、带符号位移、override 和随机 nonce 的精确挑战
短语。挑战通过后才重新连接，使用 fresh pose 执行一帧 current-anchor 低增益动作；执行完成、
post-health、零力 release 和 close 全部返回后，才询问目视结果：

```bash
scripts/sim2real.sh joint-order probe \
  --session outputs/revo3_right/joint_order/session_cache7942.json \
  --joint P00 \
  --allow-unverified-calibration \
  --allow-passive-limit-outlier
```

release/close 返回后程序会丢弃运动前已缓存的 TTY 输入，并生成一个与 ARM 不同的新 nonce。
目视结果只接受带该新 nonce 的 `MATCH`、`OPPOSITE`、`WRONG_JOINT`、`NO_MOTION`、
`MULTIPLE` 或 `UNCERTAIN`。只有 `MATCH` 会把该轴记为 `passed`；任何其他结果都会把 session
置为 `blocked`，绝不会自动进入下一轴。若主候选因真机实时限位不可行，查看 `status` 中的
备选 row，审核方向后显式追加 `--row N` 重新 preflight。

该套件刻意没有 `--all`、`--yes`、管道输入或无人值守模式。每个轴都是新的进程和完整的
open/preflight/challenge/open/execute/release/close 生命周期；一次性 approval 不能复用于下一个
关节。session 只记录测试证据，不会自动修改 `sdk_joint_order`、offset、profile calibration 或
硬件零位。approval 前后都会复核 trace/checkpoint/profile SHA；三者变化会在再次打开硬件前
阻断。仓库内的 replay、jog、run-policy 和本套件在 SDK 层共享按设备 SN 加锁的 motion lock，
避免两个部署进程同时控制同一只手；官方 GUI 或仓库外 SDK 程序仍需人工关闭。收到
`SIGINT`/`SIGQUIT`/`SIGTSTP`/`SIGTERM`/`SIGHUP`（包括 Ctrl-C、Ctrl-\\、Ctrl-Z）时，
套件会先取消协程并等待 release/close，而不会带着 MIT 增益挂起；`SIGKILL` 无法软件清理，
硬件急停仍是最终保护。

## 已验证环境

统一使用：

```text
/home/tan/miniconda3/envs/revo3/bin/python
```

该环境现有 Python 3.10.20、NumPy 1.26.4、ONNX Runtime CPU 1.20.1、
PyYAML 6.0.3、`bc-revo3-sdk` 1.5.1 和 `pyvitaisdk4bc` 1.0.10
（import 名为 `pyvitaisdk`）。不要使用当前 base Python 3.14，也不要混用
`reference/brainco-revo3-sdk/python/.venv`。

当前 VisionTouch 运行时还要求五个与传感器 SN 对应的加密力模型存在于：

```text
reference/brainco-revo3-sdk/checkpoints/<SN>/<SN>.onnx.enc
```

`pyvitaisdk4bc` 是 BrainCo 提供的硬件依赖；若新环境中缺失，应从内部 wheel/SDK
发布包安装与当前验证版本一致的 1.0.10，不能用普通 ONNX 模型替代这些加密传感器模型。

如果之后希望安装命令行入口，可在 `revo3` 环境中执行：

```bash
/home/tan/miniconda3/envs/revo3/bin/python -m pip install --no-deps -e deploy/revo3
```

wheel 安装会把 profile 模板放到
`$CONDA_PREFIX/share/brainco-hora-revo3-deploy/config/revo3_right.yaml`；先复制到项目内的
可写位置并完成标定，再通过 `--profile` 显式传入。当前环境只有 CPU provider；若以后使用
`--provider cuda`，需自行把 `onnxruntime` 替换为匹配 CUDA 的 `onnxruntime-gpu`。

## 第一步：完全离线验证

此命令不连接灵巧手，也不发送任何控制命令：

```bash
/home/tan/miniconda3/envs/revo3/bin/python \
  deploy/revo3/scripts/validate_policy.py \
  --onnx outputs/revo3_right/onnx/cylinder_stage2.onnx \
  --metadata outputs/revo3_right/onnx/cylinder_stage2.deploy_meta.yaml \
  --profile deploy/revo3/config/revo3_right.yaml \
  --check-sdk \
  --joint-pos-npy cache/revo3_right_grasp_cylinder.npy
```

验证器会检查模型名称、shape、dtype、动态 batch、metadata、关节/触觉顺序、
`action_scale`、0.9 限位缩放、有限值和实际 ORT 推理，并输出模型 SHA256 与延迟；
`--check-sdk` 额外检查 SDK 1.5.1、`pyvitaisdk4bc` API 和五个本地 VTS 力模型，
但不会枚举或连接设备。

运行单元测试：

```bash
PYTHONPATH=deploy/revo3 /home/tan/miniconda3/envs/revo3/bin/python \
  -m unittest discover -s deploy/revo3/tests -v
```

## 第二步：真机无电机 dry-run

默认模式会连接真机、读取 21 路关节和五个 VisionTouch 指尖、维护 30 帧历史并执行
ONNX 推理，但不会发送任何电机命令。它会先核对右手、21-DoF hardware、设备 SN、
固件关节限位、五个 VTS SN/类型和力模型，然后在后台并行采集 Force6D：

```bash
/home/tan/miniconda3/envs/revo3/bin/python \
  deploy/revo3/scripts/run_policy.py \
  --onnx outputs/revo3_right/onnx/cylinder_stage2.onnx \
  --metadata outputs/revo3_right/onnx/cylinder_stage2.deploy_meta.yaml \
  --profile deploy/revo3/config/revo3_right.yaml \
  --steps 200
```

要把这 200 帧与 Isaac Lab 测试逐项比较，给同一条无电机命令追加一个尚不存在的输出路径：

```bash
  --trace-npz outputs/revo3_right/traces/real_dryrun_001.npz
```

trace 在控制循环中只写内存，并在零力 release（若需要）和 SDK close 之后原子保存；已有目标
文件会被拒绝覆盖。非 preflight trace 必须同时给有限的 `--steps`。NPZ 的 `metadata_json`
记录模型/profile 哈希、关节与触觉顺序、有效真机目标限位和结束原因；逐帧数组包含原始
`obs_raw` / `proprio_hist_raw`、ONNX 原始输出、部署 clip 后 action、限位前后 target、SDK 与
policy 坐标关节角、五路力、推理/读写时延、触觉样本年龄、电流、Stall 和命令发送状态。
`--preflight-only --trace-npz <new-file.npz>` 也受支持，并固定记录 `command_sent=false`。

这只 `Revo3UltraVisionTouch` 手不要追加 `--configure-tactile`：该参数只用于兼容
Pressure/Matrix 手的 Modbus 触觉模块，VisionTouch 路径会明确拒绝它。VTS 的
`calibrate_on_open: true` 会在每次 collector 启动时重置本次五个传感器的零点；启动命令前
必须让所有指尖悬空且不接触物体。该操作不会发送电机命令，也不会改写 Modbus 触觉模式。
如果需要让 trace 的第 0 帧在已经放入圆柱后开始，可保持指尖空载启动，并追加例如
`--policy-start-delay-s 15`。程序完成 SDK/VTS 打开与校验后会醒目提示；在这 15 秒内放入
圆柱，延迟结束后的第一次观测才会填充 30 帧策略历史。该选项只允许无电机 dry-run，范围为
0～120 秒，不能用于 motion 或 preflight。

profile 默认自动检测。已有真机日志曾记录 `/dev/ttyUSB0`、5 Mbps、slave `0x7F`；
设备 ID 历史上发生过变化，所以没有硬编码。自动检测失败时可显式追加：

```text
--port /dev/ttyUSB0 --baudrate 5000000 --slave-id 0x7F
```

同一时刻只能有一个 Modbus 主站使用串口。运行官方 GUI、教学工具或另一个部署进程时再启动
本程序，会出现连续 `Invalid CRC`、`BrokenPipe` 和 auto-detect 失败。先正常关闭占用
`/dev/ttyUSB0` 的程序，再运行 preflight；可用 `lsof /dev/ttyUSB0` 只读确认占用者。

2026-08-10 在当前右手上的实测结果：五传感器串行读取均值约 53.05 ms，不能满足
20 Hz；五线程并行 collector 的均值 22.12 ms、p95 33.62 ms、最大 41.42 ms。
集成后的 200 步无电机 dry-run 成功，打印样本中的主循环 `loop_ms` 为 0.69–2.50 ms；
最大 VTS 样本年龄为 45.6 ms，其中 1/200 个样本超过 45 ms。后台采集让推理循环无需
等待五个相机逐个返回。dry-run 会在结尾报告 `max_force_age_ms` 和 `stale_samples`，便于
量化抖动；运动模式不会放宽阈值，遇到超过 45 ms 的陈旧样本会在发命令前中止。因此当前
采样链路可用于观测/推理验证，但这次抖动结果还不支持直接解除电机闭环安全门。

## 触觉映射

设备身份探测已确认当前右手为 `Revo3UltraVisionTouch`，SN 为
`BCUVR1205J2600002`。真正的五路指尖数据来自独立 USB VisionTouch 传感器；设备同时
暴露的普通 Pressure summary 在空载实测中全为零，不作为当前策略输入。

当前 profile 中的策略顺序和 SN 如下：

| 策略通道 | VisionTouch SN | 类型 | 映射状态 |
| --- | --- | --- | --- |
| thumb | `WTUVL2198X260001A` | GFBCT | 实机按压确认 |
| index | `WTUVL3197X260010B` | GFBCI | 实机按压确认 |
| middle | `WTUVL3194X26000EB` | GFBCI | 实机按压确认 |
| ring | `WTUVL3195X26000EE` | GFBCI | 实机按压确认 |
| little | `WTUVL3197X2600106` | GFBCI | 实机按压确认 |

2026-08-10 逐指按压确认：旧采集顺序对应的物理手指为
`[thumb,little,ring,index,middle]`。profile 已据此重排为策略要求的
`[thumb,index,middle,ring,little]`，并设置 `mapping_verified: true`。可用下面的无电机
命令复核；依次按压 thumb、index、middle、ring、little 时，`force_N` 对应位置应显著上升：

```bash
/home/tan/miniconda3/envs/revo3/bin/python \
  deploy/revo3/scripts/run_policy.py \
  --onnx outputs/revo3_right/onnx/cylinder_stage2.onnx \
  --metadata outputs/revo3_right/onnx/cylinder_stage2.deploy_meta.yaml \
  --profile deploy/revo3/config/revo3_right.yaml \
  --steps 400 --print-every 1
```

运行时读取每个 Force6D 向量的前三个力分量并取模，得到策略需要的
`[thumb,index,middle,ring,little]` 牛顿值。这个模长与仿真中的三维接触合力定义一致，
但仍需通过已知静态载荷确认每指的尺度、零漂和接触方向。

Pressure/Matrix 支持仅作为其他 Revo3 触觉型号的兼容 fallback：Pressure 会把 42 路
ForceSummary 的五组 distal zones 从 mN 聚合到 N；Matrix 会读取五个 tip module 的
Force 输出。它们不用于当前 UltraVisionTouch 手，且 Matrix 运动路径在达到 20 Hz 前仍被
安全禁止。

## 第三步：无电机硬件 preflight

传感器映射确认后，先运行一次硬件 preflight。它会读取并校验 21 路电机在线、故障、
电流、当前位置、设备限位和五路触觉，再执行一次 ONNX 推理，列出策略第一帧的 21 路
目标和 measured-to-target 跳变量。它不会调用任何 MIT 控制 API：

```bash
/home/tan/miniconda3/envs/revo3/bin/python \
  deploy/revo3/scripts/run_policy.py \
  --onnx outputs/revo3_right/onnx/cylinder_stage2.onnx \
  --metadata outputs/revo3_right/onnx/cylinder_stage2.deploy_meta.yaml \
  --profile deploy/revo3/config/revo3_right.yaml \
  --preflight-only
```

`--preflight-only` 与 `--enable-motion`、`--configure-tactile` 互斥。输出中的
`first_delta_gate` 必须为 `PASS`，且每路位置、电流、限位和触觉必须合理；即使全部通过，
`calibration.status: unverified` 仍会继续阻止电机运动。

当前伸直姿态的实测 preflight 结果为：21 路电机全部在线、无 fault，观测耗时 1.95 ms、
触觉样本年龄 2.57 ms；伸直位编码器有 `-0.06°～-0.30°` 的轻微越零。运行时只对“测量值”
使用 0.5° 容差，任何命令目标仍严格限制在产品和设备限位内。第一帧最大目标跳变为
19.786°（M17），因此 `first_delta_gate=FAIL`，没有发送命令。

这个失败说明当前伸直手姿不在 Stage-2 的训练起点内。训练每次 reset 都直接从
`cache/revo3_right_grasp_cylinder.npy` 的圆柱抓取姿态开始，并将该姿态同时作为当前关节位置
和初始 target；它不是一套从完全伸直到抓取的策略。8192 个缓存姿态中，即使选择最接近
当前伸直手的样本，仍至少需要一个关节移动约 85.86°。因此不能通过删除 offset 或放宽
5° 首帧门来启动策略；必须先独立完成低速、受限的关节标定和预抓取，再在实际姿态接近
某个训练缓存姿态时启动 Stage-2。

## 第四步：受限单关节点动标定

`scripts/jog_joint.py` 是独立于 ONNX 策略的标定工具。它硬限制单次位移不超过 10°、
速度不超过 2°/s、绝对增益上限 `Kp <= 2.0`、`Kd <= 1.0`、频率不超过 20 Hz；超过
`Kp=0.3` 或 `Kd=0.1` 必须追加 `--confirm-high-gain`，超过 1° 还必须追加
`--confirm-large-jog`。可重复 `--joint` 同步测试最多四轴，多轴时还必须追加
`--confirm-multi-joint`。只有选中轴使用非零 Kp/Kd，其余轴为零力；工具还检查每轴 500 mA、
选中轴合计 1000 mA 和未选中轴 5° 被动移动包络。每周期检查 21 路在线、fault、Stall、
电流和位置，结束或任意异常都会请求零力 release。默认情况下 Stall 仍会中止；对于已确认
可反驱且允许堵转的标定，可显式添加 `--allow-selected-stall`，它只忽略选中轴的 Stall 位
`0x100`，不会忽略其他 fault、过流、掉线、通信超时或位置保护。编码器在机械零点附近的
实测容差可用 `--measured-limit-tolerance-deg` 调整，硬上限为 1°，且不会放宽命令目标限位。
执行时必须显式重复所有物理安全确认：

```bash
/home/tan/miniconda3/envs/revo3/bin/python \
  deploy/revo3/scripts/jog_joint.py \
  --profile deploy/revo3/config/revo3_right.yaml \
  --joint M13 --delta-deg 1.0 \
  --kp 0.2 --kd 0.05 --rate 20 --ramp-s 1.0 --hold-s 0.25 \
  --execute --confirm-fixed --confirm-empty --confirm-estop --confirm-release
```

当前 M13（食指 MCP）首次实测：编码器从 0.10° 变为 release 后的 0.62°，正向变化
约 0.52°；峰值已记录电流为 52 mA。当 ramp 目标约为 +0.5° 时固件报告 Stall，程序立即
停止且成功发送零力 release。随后只读 preflight 确认 Stall 已清除、M13 电流为 0 mA。
这说明故障/释放路径工作正常，但在操作者目视确认 M13 的物理运动方向之前，不提高 Kp、
不继续其他关节，也不把该结果标记为完成校准。

按 10° 测量建议进行的第二次实测仍保持 `Kp=0.2`，并把 ramp 延长到 5 秒以限制为
2°/s。零力状态下 M13 在测试前已自然移动到 12.36°；受控阶段跟随约 +2.2° 后再次触发
固件 Stall，release 后稳定读数为 15.28°，相对测试起点共变化约 +2.92°，没有强行达到
10°。已记录电流最高 86 mA，随后 Stall 清除且电流回到 0 mA。同期 M12（食指 MPR）从
约 0.07° 变为 1.71°，说明还需区分机械耦合、零力被动移动和真实交叉运动。未得到操作者
目视确认前，不应忽略 Stall 或继续提高增益。

四个非拇指 MCP 的同步 10° 测试使用 M1/M5/M9/M13、`Kp=0.2` 和 1°/s。起点角度分别为
`[-0.17°,1.99°,1.65°,0.63°]`；目标 ramp 到约 +0.5° 时，M9 和 M13 同时报告 Stall，
工具立即对四轴零力 release。release 后角度为 `[-0.17°,2.56°,2.16°,0.91°]`，相对起点
实际变化约 `[0.00°,0.57°,0.51°,0.28°]`；已记录电流为 `[0,57,81,69] mA`。
随后只读检查确认四轴电流均为 0、Stall 已清除。低增益没有完成 10°，因此不能把这次
安全中止解释为四指 10° 标定成功。

按操作者要求，随后以相同四轴 M1/M5/M9/M13、`Kp=2.0`、`Kd=0.25` 和 0.5°/s
重新尝试同步 10°。起点为 `[-0.18°,2.57°,2.16°,-0.42°]`；在第 10 个控制周期、
目标仅增加约 0.25° 时，M13 同时报出 `status=0x100` 和 `error=0x100`，工具立即中止并
成功发送零力 release。触发前记录的四轴最高电流不超过 `[78,0,0,90] mA`。随后只读
复查为 `[-0.17°,2.75°,2.20°,-0.45°]`，四轴电流均为 0，故障位已清除。该次默认严格
Stall 模式没有完成 10°。

操作者进一步确认该手可反驱、堵转常见并授权忽略所选四轴的 Stall 后，使用
`--allow-selected-stall` 再次测试。四轴起点为 `[0.13°,-0.11°,0.05°,21.25°]`；目标
到达 10° 时实测相对位移为 `[9.99°,9.78°,9.62°,9.72°]`，短暂停留后达到
`[10.01°,10.06°,9.93°,9.76°]`，随后完整回程至相对起点
`[0.34°,0.44°,0.34°,0.36°]` 并成功发送零力 release。日志采样到的最大绝对电流约
212 mA，低于 500 mA 单轴上限；后续默认严格模式的只读 preflight 确认四个选中 MCP
电流均为 0 且无 Stall/fault。因此四个非拇指 MCP 的同步正向 10° 点动已完成，但这只确认
本次方向和局部行程，不代表 21 路 offset/全行程或 ONNX policy 闭环已经标定。

本次实测命令的关键参数为：

```bash
--joint M1 --joint M5 --joint M9 --joint M13 \
--delta-deg 10 --kp 2.0 --kd 0.25 --rate 20 --ramp-s 20 \
--measured-limit-tolerance-deg 1 \
--confirm-large-jog --confirm-multi-joint --confirm-high-gain \
--allow-selected-stall
```

## 第五步：运动模式（已完成短时首测）

没有 `--enable-motion` 时程序始终不会发送电机命令。VisionTouch 五指映射已经确认，
但 21 路完整标定仍为 `calibration.status: unverified`，因此台架调试必须显式追加
`--allow-unverified-calibration`。从参考工程继承的 PIP/DIP `+0.3 rad` 软件 offset 会让伸直
实机在策略限位处产生约 17–20° 的假首跳；当前 profile 直接使用 SDK 返回的绝对位置，21 路
软件 offset 暂为全零。这只影响部署坐标换算，运行时从未调用电机置零/重设零位接口。
当时只读 preflight 的最大首命令差值为 3.127°，低于 5°门限。

2026-08-10 首次 ONNX 电机闭环使用 `Kp=0.2`、`Kd=0.05`：单周期测试成功，循环耗时
5.26 ms、触觉样本年龄 2.1 ms，并正常零力 release。随后尝试连续 5 周期，成功发送 2 帧，
第 3 次状态读取因 M13/M16 的 `status=error=0x100` 中止；仅对白名单 M13/M16 放行后，
又成功发送 1 帧，下一次读取在 M0 出现相同 `0x100` 并中止。两次异常路径都成功发送零力
release。已发送帧的循环耗时为 3.5–5.7 ms、触觉样本年龄不超过 18.7 ms，但每帧
`action_abs_max=1.0`，策略输出持续饱和。只读复查确认 fault 已清除；累计几帧已让多个关节
明显运动，例如 middle PIP 约 6.77°、middle MPR 约 -4.04°。因此已证实 ONNX 能真实驱动
整手，但尚未完成稳定的连续运行。

`--allow-stall MOTOR` 必须按电机逐项重复，例如 `--allow-stall M13 --allow-stall M16`；它只
忽略所列电机的 Stall 位 `0x100`，不能一次静默关闭全手保护。其他 fault、500 mA 电流、
在线、限位、跟踪误差、新鲜度和时序检查仍然有效。

随后按操作者要求增加 `--stall-grace-s 1.0`：任一电机的 Stall `0x100` 只有连续存在
超过 1.0 秒才中止，恰好 1.0 秒仍放行，状态清除会重置该电机计时；其他 error 位不会延迟。
计划以 `Kp=2.0`、`Kd=0.25` 运行 200 周期，但只成功发送第 1 帧，下一次状态读取即因
M0=514 mA、M4=-501 mA 超过既定 500 mA 上限而中止，并成功零力 release。这次中止不是
Stall grace 触发。后续默认严格 preflight 确认 M0/M4 电流回到 0 且 fault 已清除；当时姿态
包括 middle PIP=12.48°、index MPR=10.79°、thumb DIP=9.12°。因此 200 周期测试尚未完成，
在明确决定电流保护策略前不应把“错误延迟 1 秒”自动解释为也延迟过流保护。

本次命令的关键参数为：

```bash
--steps 200 --kp 2.0 --kd 0.25 --enable-motion \
--allow-unverified-calibration --stall-grace-s 1.0
```

操作者随后明确把 ONNX 网络运行的绝对电流阈值改为 10 A；profile 使用
`max_abs_current_ma: 10000`，独立 calibration jog 仍由 `jog_max_abs_current_ma: 500`
限制，避免把该授权扩散到点动工具。第一次 10 A 重测发送 1 帧后，策略要求 M0=10.02°，
超过设备实时上限 10.00°而被位置保护中止。运行器现已修复为连接后把训练/静态 target
限位与设备实时报告限位取交集，并向内保留 0.05°，因此 M0 最大 target 为 9.95°。

修复后再次执行同一 200 周期命令，成功连续运行到第 120 周期（约 6 秒）；M4 的 Stall
`0x100` 随后连续超过 1.0 秒，程序按约定在下一次读取时中止并成功零力 release，因此
**没有完成 200 周期**。本次没有触发 10 A 过流、设备限位、通信或周期保护；采样日志中
循环耗时约 2.48–5.81 ms，触觉年龄不超过 20.5 ms。网络动作始终饱和到 1.0，middle
指尖力曾达到约 12.11 N。使用仅限 read-only preflight 的 3°编码器容差复查后，M4
fault 已清除且电流为 0；M9/M11/M13 在零力状态分别读到约 -2.57°/-2.62°/-1.75°。
当前姿态重新启动时 M11 首目标差为 5.057°，略超 5°首帧门，故没有自动重启。

之后零力状态下 M0（little MPR）进一步反驱到 -21.17°。旧 profile 使用参考产品表
`[-14°,15°]`，导致 read-only preflight 在打印首目标前误报测量越界；当前已按绑定设备
`BCUVR1205J2600002` 的实时报告更新四个 MPR 静态测量包络：M0 `[-30°,10°]`、
M4 `[-25°,20°]`、M8 `[-20°,25°]`、M12 `[-10°,30°]`。训练 target 限位仍保持原值并
继续与实时设备限位取交集，因此该修正不会把策略 target 放宽到整个设备范围。

修正后的 preflight 可完整执行，但正确报告 `first_delta_gate=FAIL`：M0 绝对位置 -21.17°、
首目标 -11.11°，差 10.057°；M11 差 5.057°。这表示当前**物理姿态**不满足模型首帧连续性，
不表示需要修改电机零位。不得用当前读数反推或写入新零位；应选择一个已知的物理参考姿态
校验绝对坐标到训练坐标的固定软件映射，或者只移动实际关节姿态到模型可接受的启动姿态
（移动姿态不等于重新定零），再重复 preflight。不要删除或放宽首跳门。

操作者随后手工摆出近似圆柱抓取姿态。read-only preflight 显示多数屈曲关节已进入抓取
范围，但 M8（middle MPR）绝对位置 21.45°、首目标 11.11°，差 10.34°；M12（index MPR）
绝对位置 27.21°、首目标 12.09°，差 15.12°，因此仍为 `first_delta_gate=FAIL`，没有发送
电机命令。五路触觉仅约 `[0.003,0.002,0.002,0.004,0.006] N`，也未形成训练时的圆柱
接触载荷。不能用这套“近似姿态”反推或写入电机零位/软件 offset；应保持绝对位置直读，
实际调整 M8/M12 的侧摆姿态并放置圆柱形成接触后重复 preflight。粗略连续性范围为
M8 约 6.1–16.1°、M12 约 7.1–17.1°，但最终必须以重新推理得到的
`first_delta_gate=PASS` 为准。

### 圆柱抓取手动初始姿态

下表来自训练实际使用的 `cache/revo3_right_grasp_cylinder.npy` 第 7942 行；它是在最近一次
实机读数基础上，从 8192 个缓存样本中按 21 维角度 L2 距离选出的最近样本。数值已转换为
部署的 SDK M0–M20 顺序，单位为度。当前软件 offset 为全零，因此表中数值可直接与 SDK
绝对位置读数比较；它们只是手动摆姿态目标，**不得写成电机零位**。建议先调到目标 ±2°。

| Motor | policy joint | 目标绝对位置 (deg) |
|---|---|---:|
| M0 | `right_little_MPR_joint` | -8.075 |
| M1 | `right_little_MCP_joint` | 70.792 |
| M2 | `right_little_PIP_joint` | 22.029 |
| M3 | `right_little_DIP_joint` | 6.958 |
| M4 | `right_ring_MPR_joint` | -13.500 |
| M5 | `right_ring_MCP_joint` | 48.879 |
| M6 | `right_ring_PIP_joint` | 16.705 |
| M7 | `right_ring_DIP_joint` | 7.725 |
| M8 | `right_middle_MPR_joint` | 0.195 |
| M9 | `right_middle_MCP_joint` | 58.905 |
| M10 | `right_middle_PIP_joint` | 9.017 |
| M11 | `right_middle_DIP_joint` | 8.200 |
| M12 | `right_index_MPR_joint` | 10.987 |
| M13 | `right_index_MCP_joint` | 72.306 |
| M14 | `right_index_PIP_joint` | 21.314 |
| M15 | `right_index_DIP_joint` | 7.961 |
| M16 | `right_thumb_MCP_joint` | 21.836 |
| M17 | `right_thumb_PIP_joint` | 15.591 |
| M18 | `right_thumb_DIP_joint` | 7.779 |
| M19 | `right_thumb_CMP_joint` | 94.066 |
| M20 | `right_thumb_CMR_joint` | 64.429 |

该缓存行后 7 维物体状态为位置 `[0,-0.08,1.635]`、四元数 xyzw `[0,0,0,1]`，属于
Isaac 场景坐标，不能直接当作台架上的毫米位置使用。手动摆完后必须重新运行 read-only
preflight；只有 `first_delta_gate=PASS` 才进入电机闭环。

2026-08-10 按第 7942 行手动摆姿态后的实测 preflight 成功：21 轴最大首跳 2.387°，
`first_delta_gate=PASS`。随后按 `Kp=2.0`、`Kd=0.25`、10 A 阈值、1.0 s Stall grace
启动计划中的 200 周期闭环。程序运行到第 30 周期（约 1.5 s）后，M16 的 Stall `0x100`
连续超过 1.0 s，按约定中止并成功零力 release，因此本次仍未完成 200 周期。过程中未触发
过流、设备限位或通信保护；循环耗时约 3.27–4.77 ms。触觉已出现明显接触，采样峰值包括
thumb 约 7.11 N、middle 约 10.27 N、ring 约 4.07 N；动作仍持续饱和到 1.0。

释放后严格 fault 检查确认 M16 报警已清除且电流为 0，但实际姿态已经离开缓存起点：最新
preflight 最大差值为 M1 11.63°，另有 M0 11.15°、M12 8.72°，因此
`first_delta_gate=FAIL`，没有自动重启。若继续测试，需要先重新手动摆回表中姿态；是否延长
或取消 M16 的 1.0 s Stall 判定应作为单独参数决定，不能把一次 preflight PASS 当作持续
关闭保护的依据。

操作者随后明确授权本阶段完全不监测 `0x100`。运动命令可使用 `--ignore-all-stall`，它只
忽略全部 21 轴的 Stall 位；其他 error 位、10 A 阈值、在线、限位、跟踪、通信和时序保护
仍有效。该参数不能与 `--stall-grace-s` 或逐轴 `--allow-stall` 同时使用。授权后第一次
read-only 检查仍未发命令，因为零力姿态已漂移至 M0=-28.90°、M12=19.80°，对应首跳
17.79°/8.69°，`first_delta_gate=FAIL`；必须先重新摆回缓存姿态，不能用关闭 Stall 绕过
首帧连续性门。

也可以用下面的一条命令完成“缓慢到第 7942 行绝对姿态，然后立即闭环 200 周期”，无需
手动摆回。预定位按 SDK 返回的绝对关节位置插值，20 Hz 下每轴每帧最多 0.1°，即不超过
2°/s；它不会设置电机零位，也不会修改 profile offset。到位后最大误差超过 2.5°时拒绝
启动网络。预定位和网络共用同一连接，中间不会发送零力 release；完整 200 周期结束或异常
退出时仍会发送零力 release。

启动前固定手和圆柱、清空运动路径并准备急停。由于 VTS 会在打开连接时校准零点，五个
指尖在程序刚启动时必须无接触；圆柱可预先放在目标抓取位置，但不得压住指尖。

```bash
/home/tan/miniconda3/envs/revo3/bin/python \
  deploy/revo3/scripts/run_policy.py \
  --onnx outputs/revo3_right/onnx/cylinder_stage2.onnx \
  --metadata outputs/revo3_right/onnx/cylinder_stage2.deploy_meta.yaml \
  --profile deploy/revo3/config/revo3_right.yaml \
  --preposition-cache cache/revo3_right_grasp_cylinder.npy \
  --preposition-row 7942 \
  --preposition-speed-deg-s 2 \
  --confirm-preposition \
  --kp 2.0 --kd 0.25 \
  --steps 200 --print-every 10 \
  --enable-motion \
  --allow-unverified-calibration \
  --ignore-all-stall
```

该命令沿用 profile 中的 10 A 电流阈值，并按当前授权仅忽略全部电机的 Stall `0x100`；
其他 error、在线状态、绝对/实时限位、跟踪误差、通信超时和控制周期保护仍然有效。

只读诊断可按需放宽“实测位置”显示容差，而不会放宽命令限位：

```bash
--preflight-only --preflight-position-tolerance-deg 3
```

上真机前至少完成：

1. 逐关节低速确认 policy/SDK 顺序、方向、绝对位置坐标映射和拇指差动关节含义；不得写硬件零位。
2. 对照实际 USD/PhysX 核实 profile 的 21 路限位及 0.9 缩放。
3. 五个 VisionTouch SN 的物理映射已确认；再次复核逐指按压结果。
4. 用已知载荷验证五路 Force6D 的零点、尺度与稳定性，并复测后台样本年龄和抖动。
5. 在无物体、急停可用的条件下逐关节验证 offset、低增益 MIT 和小角度点动。
6. 当前全零 offset 已使伸直姿态 `first_delta_gate=PASS`，但 Stage-2 训练仍从 grasp-cache
   抓取姿态开始；首帧连续不代表伸直姿态位于模型训练分布内。
7. 当前设备 SN 已绑定为 `BCUVR1205J2600002`；复核无误后把
   `calibration.status` 改为 `verified`，之后才进行有保护的短时闭环测试。

运动模式还会检查设备身份、电机 error（忽略非故障 Running bit）、电流、跟踪误差、
21 个电机在线位、输入/输出 NaN、观测新鲜度、首次 measured-to-target 跳变、逐步目标、
静态产品限位、设备实时报告限位和控制周期。`q_unscaled` 始终沿用训练限位；动作目标则夹在
训练限位与 SDK 电机坐标限位的交集中。SDK 读、写、release 和 close 都有超时。只有本进程
可能已经发出控制帧时，退出或异常才会尝试发送零 Kp/Kd/电流的 MIT release；这会使手变软
并可能掉落物体，所以台架下方必须清空并始终准备硬件急停。release 若通信失败不能替代
硬件急停，清理失败也会返回非零状态。

`--allow-unverified-calibration` 只绕过标定状态，不会绕过 VisionTouch 映射、设备 SN、
电机在线/故障、电流、限位、新鲜度或时序安全门；它只用于明确的台架调试，不应作为
日常启动参数。

## 与两个参考工程的关系

- `reference/RevoLab/deploy/revo3` 使用旧的无触觉 42 维帧，产出 126/30x42 输入，
  不能运行当前 141/30x47 模型。
- 它还依赖旧 `bc-stark-sdk 1.4.5` API；本运行时已适配现有
  `bc-revo3-sdk 1.5.1`。
- `reference/brainco-revo3-sdk` 用于确认设备连接、电机顺序、触觉 API 与单位。其加密
  VTS ONNX 不是 HORA 策略模型，但它们是把五个相机信号转换为 Force6D 的必需传感器模型。
