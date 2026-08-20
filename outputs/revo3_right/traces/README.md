# Trace 曲线绘制

本目录保存仿真或真机运行产生的 `.npz` policy trace。项目工具
`tools/plot_policy_trace.py` 可以从 trace 生成：

- 21 个 policy-order 关节的实际角度和目标角度曲线；
- 拇指、食指、中指、无名指和小指的力曲线。

## 绘制当前 200 帧 Trace

请在仓库根目录 `/home/tan/hora/BrainCo` 执行：

```bash
MPLCONFIGDIR=/tmp/brainco-matplotlib \
/home/tan/miniconda3/envs/revo3/bin/python tools/plot_policy_trace.py \
  outputs/revo3_right/traces/sim_joint_order_latent_cache7942_200f.npz \
  --output-dir outputs/revo3_right/traces/plots
```

生成的文件为：

```text
outputs/revo3_right/traces/plots/
├── sim_joint_order_latent_cache7942_200f.joint_angles.png
└── sim_joint_order_latent_cache7942_200f.fingertip_forces.png
```

关节角图中：

- 蓝色实线 `measured`：`policy_pos_rad`，仿真实际关节角；
- 橙色虚线 `target`：`policy_target_rad`，策略目标关节角；
- 默认显示单位为度。

指尖力图读取 `force_n`，单位为 N。五指名称和顺序来自 NPZ 内嵌的
`metadata_json.contact_order`。

当前文件包含 200 帧，policy rate 为 20 Hz，因此时间轴约为 0–9.95 秒。

## 绘制其他 Trace

将命令中的 NPZ 文件名替换为需要绘制的 trace：

```bash
MPLCONFIGDIR=/tmp/brainco-matplotlib \
/home/tan/miniconda3/envs/revo3/bin/python tools/plot_policy_trace.py \
  outputs/revo3_right/traces/<TRACE_NAME>.npz \
  --output-dir outputs/revo3_right/traces/plots
```

例如绘制 20 帧版本：

```bash
MPLCONFIGDIR=/tmp/brainco-matplotlib \
/home/tan/miniconda3/envs/revo3/bin/python tools/plot_policy_trace.py \
  outputs/revo3_right/traces/sim_joint_order_latent_cache7942_20f.npz \
  --output-dir outputs/revo3_right/traces/plots
```

## 常用选项

```text
--angle-unit deg     角度以度显示，默认值
--angle-unit rad     角度以弧度显示
--dpi 200            设置输出 PNG 分辨率
--show               保存图片后打开 Matplotlib 窗口
--output-dir PATH    指定图片输出目录
```

查看完整帮助：

```bash
/home/tan/miniconda3/envs/revo3/bin/python tools/plot_policy_trace.py --help
```

`MPLCONFIGDIR=/tmp/brainco-matplotlib` 用于把 Matplotlib 字体和配置缓存放到可写的临时目录，
不会改变 trace 数据或图片内容。
