# Revo3 measured trace 真机 replay 流程

本文记录如何把仿真 trace 中的实测关节角 `policy_pos_rad` 按原始 20 Hz 时序
回放到 Revo3 右手。当前已验证的输入为：

```text
outputs/revo3_right/traces/sim_joint_order_latent_cache7942_200f.npz
```

这是真机运动流程，会向 21 个电机发送 MIT 位置指令。每次执行前都应重新确认
手已牢固固定、运动路径无障碍、硬件急停可用，以及接受结束或异常后零力释放会让手变软。

## 1. Replay 的数据与关节顺序

`--trajectory-source measured` 选择 NPZ 中的 `policy_pos_rad[T,21]`，单位是弧度。程序先按
`metadata_json.joint_order` 解释 policy order，再按 profile 重排到 SDK `M0..M20` order；
不能将 NPZ 的 21 列直接当成 `M0..M20` 发送。

当前复核的映射为：

| SDK | Policy | Joint |
|---:|---:|---|
| M00 | P01 | `right_little_MPR_joint` |
| M01 | P06 | `right_little_MCP_joint` |
| M02 | P11 | `right_little_PIP_joint` |
| M03 | P16 | `right_little_DIP_joint` |
| M04 | P03 | `right_ring_MPR_joint` |
| M05 | P08 | `right_ring_MCP_joint` |
| M06 | P13 | `right_ring_PIP_joint` |
| M07 | P18 | `right_ring_DIP_joint` |
| M08 | P02 | `right_middle_MPR_joint` |
| M09 | P07 | `right_middle_MCP_joint` |
| M10 | P12 | `right_middle_PIP_joint` |
| M11 | P17 | `right_middle_DIP_joint` |
| M12 | P00 | `right_index_MPR_joint` |
| M13 | P05 | `right_index_MCP_joint` |
| M14 | P10 | `right_index_PIP_joint` |
| M15 | P15 | `right_index_DIP_joint` |
| M16 | P14 | `right_thumb_MCP_joint` |
| M17 | P19 | `right_thumb_PIP_joint` |
| M18 | P20 | `right_thumb_DIP_joint` |
| M19 | P04 | `right_thumb_CMP_joint` |
| M20 | P09 | `right_thumb_CMR_joint` |

## 2. 用固定抓握姿态标定每轴 offset

当真机和 URDF 的关节零位不一致时，对齐量定义为：

```text
SDK target[M] = sim pose[P→M] + sim2real_joint_offset[M]
```

因此，若某轴需要比当前命令再向 SDK 正方向转 2° 才与仿真一致，该轴就执行
`--add Mxx=+2`；若当前真机姿态已经比仿真更靠正方向，则使用负值。

首个标定基准使用 `cache row 7942` trace 的 `frame 0 / policy_pos_rad`。这是 episode
刚开始时的仿真实测抓握姿态，不是 action，也不是积分后的 `policy_target_rad`。

已生成的初始候选 profile 是：

```text
outputs/revo3_right/offset_calibration/cache7942_frame000_v01.yaml
```

它是正式 profile 的独立副本，21 轴 offset 仍为全零，且
`calibration.status` 保持 `unverified`。初始姿态可用下面的离线命令重新生成：

```bash
scripts/sim2real.sh offset-cal init \
  --profile deploy/revo3/config/revo3_right.yaml \
  --trace-npz outputs/revo3_right/traces/sim_joint_order_latent_cache7942_200f.npz \
  --checkpoint outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt \
  --frame 0 --trajectory-source measured \
  --output-profile outputs/revo3_right/offset_calibration/cache7942_frame000_v01.yaml
```

工具拒绝覆盖旧文件，每轮调整都使用新版本名。例如同时将 M16 增加 2°、
M13 减少 1°：

```bash
scripts/sim2real.sh offset-cal adjust \
  --profile outputs/revo3_right/offset_calibration/cache7942_frame000_v01.yaml \
  --add M16=+2 --add M13=-1 \
  --output-profile outputs/revo3_right/offset_calibration/cache7942_frame000_v02.yaml
```

`--add` 是在上一版上累加；`--set M16=2` 则将 M16 的候选 offset 直接设为 2°。
可用 `show` 查看某一版的 `sim + offset = candidate SDK target`：

```bash
scripts/sim2real.sh offset-cal show \
  --profile outputs/revo3_right/offset_calibration/cache7942_frame000_v02.yaml \
  --trace-npz outputs/revo3_right/traces/sim_joint_order_latent_cache7942_200f.npz \
  --checkpoint outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt \
  --frame 0 --trajectory-source measured
```

每个候选 profile 先做只读 preflight：

```bash
scripts/sim2real.sh replay \
  --trace-npz outputs/revo3_right/traces/sim_joint_order_latent_cache7942_200f.npz \
  --checkpoint outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt \
  --profile outputs/revo3_right/offset_calibration/cache7942_frame000_v01.yaml \
  --trajectory-source measured --recorded-rate --preposition-to-first \
  --start-frame 0 --frames 1 --all-joints --max-speed-deg-s 2 \
  --kp 0.2 --kd 0.05 --preflight --allow-unverified-calibration \
  --ignore-all-stall --measured-limit-tolerance-deg 5
```

然后以不超过 2°/s 的速度到达该姿态，并通电保持 30 秒供目视比对：

```bash
scripts/sim2real.sh replay \
  --trace-npz outputs/revo3_right/traces/sim_joint_order_latent_cache7942_200f.npz \
  --checkpoint outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt \
  --profile outputs/revo3_right/offset_calibration/cache7942_frame000_v01.yaml \
  --trajectory-source measured --recorded-rate --preposition-to-first \
  --start-frame 0 --frames 1 --all-joints --max-speed-deg-s 2 \
  --kp 0.2 --kd 0.05 --hold-final-s 30 \
  --execute --allow-unverified-calibration --ignore-all-stall \
  --measured-limit-tolerance-deg 5 \
  --confirm-fixed --confirm-clear-path --confirm-estop --confirm-release \
  --confirm-mapping --confirm-large-excursion --confirm-full-hand \
  --confirm-recorded-rate --confirm-preposition --confirm-hold
```

保持期间仍以 20 Hz 检查在线状态、电流、非 Stall fault、限位和跟踪误差；
30 秒到期、Ctrl-C 或异常退出后都会零力 release。保持时手仍在通电，不要用手强行扭动关节。

建议每轮只改 1–3 个能明确观察的轴，步长先用 2°，接近后改为 0.5°。记录
`候选 profile / M 编号 / 调整量 / 调整前后的物理现象`。姿态匹配后不要立即标记
`verified`：至少再用两个不同的抓握姿态验证。若同一轴所需 offset 随姿态明显变化，
说明问题可能是 URDF 轴向、关节传动比、连杆尺寸或差动/耦合定义，不能用一个常数零位偏移彻底解决。

## 3. 环境和串口

从仓库根目录执行：

```bash
cd /home/tan/hora/BrainCo
scripts/sim2real.sh env-check
```

部署环境默认使用：

```text
/home/tan/miniconda3/envs/revo3/bin/python
```

`deploy/revo3/config/revo3_right.yaml` 已绑定当前 USB adapter，不依赖 SDK 自动扫描：

```yaml
sdk:
  port: /dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller_APACb111216-if00-port0
  baudrate: 5000000
  slave_id: 127
  auto_detect: false
  serial_allowlist: [BCUVR1205J2600002]
```

如果出现 `SDK 自动发现失败: No available port found`，先在宿主机检查：

```bash
ls -l /dev/ttyUSB0
ls -l /dev/serial/by-id/
id
```

当前设备节点应为 `root:dialout`，运行用户应在 `dialout` 组。在容器或沙箱中，
`/sys/class/tty/ttyUSB0` 可见不代表 `/dev/ttyUSB0` 可打开；需要把宿主机设备节点映射进运行环境。

## 4. 软件回归测试

在打开真机串口之前运行：

```bash
PYTHONPATH=deploy/revo3 \
  /home/tan/miniconda3/envs/revo3/bin/python -m unittest \
  deploy.revo3.tests.test_runtime

bash scripts/test_sim2real.sh
```

2026-08-12 最新重测结果为 `63 tests OK` 和
`PASS: sim2real environment runner dispatch and safety checks`。

## 5. 离线检查 trace 和映射

以下命令只读取文件，不连接真机：

```bash
scripts/sim2real.sh replay \
  --trace-npz outputs/revo3_right/traces/sim_joint_order_latent_cache7942_200f.npz \
  --checkpoint outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt \
  --profile deploy/revo3/config/revo3_right.yaml \
  --trajectory-source measured --recorded-rate \
  --frames 200 --all-joints
```

检查输出中的 checkpoint SHA256、`rate=20Hz`、`trajectory_source=measured` 和上面的
`Pidx -> SDK Midx` 映射。

## 6. 真机只读 preflight

Preflight 会连接真机，核对设备序列号、21 路在线/fault/电流/限位和当前位姿，
然后生成预定位加 replay 计划。它不会调用 MIT 写接口：

```bash
scripts/sim2real.sh replay \
  --trace-npz outputs/revo3_right/traces/sim_joint_order_latent_cache7942_200f.npz \
  --checkpoint outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt \
  --profile deploy/revo3/config/revo3_right.yaml \
  --trajectory-source measured --recorded-rate --preposition-to-first \
  --max-speed-deg-s 2 --frames 200 --all-joints \
  --kp 0.2 --kd 0.05 --print-every 10 \
  --preflight --allow-unverified-calibration
```

成功时应看到：

```text
first_delta_gate=PASS
bounded_plan: mode=recorded_rate ...
preposition_to_first=enabled speed<=2deg/s then_trace_rate=20Hz
PREFLIGHT ONLY: hardware health/mapping checked; no command was sent.
```

预定位时间不是固定的，而是由每次 preflight 时的真机姿态与 trace 第 0 帧的最大差值
决定。首轮张开姿态需要约 36 秒；2026-08-12 重测时手位于上一轮末帧附近，
计划为 `418 ticks / 20.9 s`，其中预定位约 10.9 秒，replay 仍为 10 秒。

`--ignore-all-stall` 可用于 preflight 和 execute，两种模式都只忽略 Stall `0x100`；
其他 fault、电流、掉线、通信和限位检查不变。

## 7. 执行 21 轴 measured replay

在同一轮 preflight 通过且物理条件仍然满足后执行：

```bash
scripts/sim2real.sh replay \
  --trace-npz outputs/revo3_right/traces/sim_joint_order_latent_cache7942_200f.npz \
  --checkpoint outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt \
  --profile deploy/revo3/config/revo3_right.yaml \
  --trajectory-source measured --recorded-rate --preposition-to-first \
  --max-speed-deg-s 2 --frames 200 --all-joints \
  --kp 0.2 --kd 0.05 --print-every 10 \
  --execute --allow-unverified-calibration --ignore-all-stall \
  --confirm-fixed --confirm-clear-path --confirm-estop \
  --confirm-release --confirm-mapping --confirm-large-excursion \
  --confirm-full-hand --confirm-recorded-rate --confirm-preposition
```

两个运动阶段为：

1. `--preposition-to-first`：从 fresh 真机姿态以每轴最大 2°/s 插值到 trace 第 0 帧；
2. `--recorded-rate`：保留 200 个 measured endpoint，按 trace 的 20 Hz 时钟直接回放 10 秒。

单次真机执行最多允许 600 个 trace endpoint，也就是 profile 固定 20 Hz 下的 30 秒。
预定位 tick 单独生成，不占用这 600 帧。30 秒 Stage1 轨迹可使用：

```bash
scripts/sim2real.sh replay \
  --trace-npz outputs/revo3_right/traces/sim_stage1_last_cache7942_600f_v02.npz \
  --checkpoint outputs/revo3_right/run_cylinder_v2_continue/stage1_nn/last.pth \
  --profile deploy/revo3/config/revo3_right.yaml \
  --trajectory-source measured --recorded-rate --preposition-to-first \
  --max-speed-deg-s 8 --frames 600 --all-joints \
  --kp 1.0 --kd 0.05 --print-every 10 \
  --execute --allow-unverified-calibration --ignore-all-stall \
  --confirm-fixed --confirm-clear-path --confirm-estop \
  --confirm-release --confirm-mapping --confirm-large-excursion \
  --confirm-full-hand --confirm-recorded-rate --confirm-preposition
```

`--ignore-all-stall` 只忽略 SDK Stall bit `0x100`。掉线、通信异常、非 Stall fault、电流、
硬限位和跟踪误差检查仍保留。正常结束和异常退出都会请求零力 MIT release。

## 8. 成功标准和本次结果

成功执行应同时满足：

```text
TRACE REPLAY ENABLED: target_mode=recorded_rate ...
execute_first_delta_gate=PASS (fresh checked sample)
replay=200/200 ... source_row=199 step=199
post_send_health=PASS ...
Zero-force MIT release sent.
```

2026-08-12 重测结果：

- 计划：`418 ticks / 20.9 s`；
- 预定位：约 10.9 秒，速度上限 2°/s；
- 回放：`200/200`，20 Hz，10 秒；
- 末条指令后健康检查：`PASS`；
- 结束处理：零力 MIT release 已发送；
- 末帧最大绝对位置误差：约 `10.63°`，位于 `M16 / P14 / right_thumb_MCP_joint`。

本轮末帧 target（SDK M0..M20，度）：

```text
[-6.761, 70.554, 27.558, 1.037, -8.252, 52.488, 14.668,
  9.518, 12.330, 54.375, 14.466, 8.818, 15.000, 64.112,
 23.260,  6.808, 19.985, 19.399, 9.977, 98.842, 51.168]
```

末条指令后的 final measured（SDK M0..M20，度）：

```text
[-6.68, 71.36, 28.60, -0.06, -11.52, 52.44, 16.91,
  9.35,  8.53, 54.39, 13.77,  7.60, 15.16, 62.91,
 24.07,  6.80,  9.36, 18.70, 11.30, 98.68, 50.77]
```

M16 在 replay 中的跟随明显弱于其他轴。当前 profile 的跟踪误差门限为 25°，因此本次
不会中止；若需要改善轨迹跟随，优先单独检查 M16 的物理负载、差动关节含义、电机状态
和低增益设置，不应只以 `post_send_health=PASS` 判定跟随质量。

## 9. 常见中止

| 现象 | 含义/处理 |
|---|---|
| `No available port found` | 检查宿主机 `/dev`、`dialout` 组和 profile 的 by-id 路径；不要仅看 sysfs |
| `--ignore-all-stall is only valid with --preflight or --execute` | 离线检查不使用该参数；硬件 preflight/execute 可用 |
| `first_delta_gate` 不通过 | 不执行；重新读取真机姿态并使用 `--preposition-to-first` |
| 初始实测角略低于指令下限 | 新版预定位会将 fresh measured start 夹到可发送的指令包络；不会改写 trace endpoint |
| 途中 fault/过流/掉线/跟踪误差超限 | 程序中止并请求零力 release；记录首个异常电机后排查 |
| 日志中 release 行看似早于末帧行 | stdout/stderr 缓冲顺序可与实际 cleanup 顺序不同；以进程结果和健康行综合判断 |
