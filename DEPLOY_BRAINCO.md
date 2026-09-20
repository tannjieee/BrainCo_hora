# BrainCo 服务器代码同步

目标：`tanjie@brainco:/home/tanjie/BrainCo_hora`。

```bash
ssh -J gpu-access@47.115.128.206 -p 22020 tanjie@127.0.0.1
cd /home/tanjie/BrainCo_hora
```

上传范围为 Python 代码、配置、脚本、测试、文档、机器人/物体资产和已有输入抓握缓存。排除 `output`、`outputs`、所有模型 checkpoint、本地冒烟缓存、`.git`、IDE 配置、凭据、字节码与仿真日志。`UPLOAD_SHA256SUMS` 可用于逐文件校验。

2026-09-20 检查时，该账号默认 Python 是 3.12.3，未安装 torch、isaaclab、isaacsim；项目上传不包含 conda/Isaac Sim 运行环境，也不启动训练。服务器没有 `/mnt/nas`，新训练的 outputs 默认写在本项目目录下。不要使用面向阿里云 NAS 的 `scripts/dsw.sh`。

## 服务器运行环境与监控（2026-09-20）

已在该账号下新增独立环境 `/home/tanjie/.venvs/hora`（Python 3.11），安装 PyTorch 2.7.0+cu128、Isaac Sim 5.1.0.0 和 Isaac Lab 0.54.4。Isaac Lab 核心源码与本地验证版本一致，位于 `/home/tanjie/IsaacLab`。不改系统 Python，也不使用其他账号的环境。

```bash
source /home/tanjie/.venvs/hora/bin/activate
cd /home/tanjie/BrainCo_hora
```

本次缓存采集的启动脚本、环境安装日志和采集日志保存在 `remote_runs/duck_collect_20260920/`。正式任务通过远端 tmux 运行，与 SSH 终端分离；完整参数记录在该目录的 `launch.json`，结束状态写入 `collect.exit`。运行时指定 `--device cuda:2`，关闭多 GPU 渲染，并设置 `TMPDIR=/home/tanjie/.cache/hora/tmp`，避免共享临时日志目录的权限冲突。启动脚本还检查完成报告与缓存校验和，防止仿真退出码掩盖启动失败。

本机可使用 `scripts/monitor_duck_remote.py` 通过已认证 SSH 控制连接镜像这些日志，默认每 30 秒刷新。监控器不保存密码；连接失效时会报告错误，需要重新建立控制连接。关闭本机监控终端不会停止远端采集。

训练所需的 `tensorboardX==2.6.4` 也已安装到独立环境。训练对照日志位于
`remote_runs/duck_compare_20260920/jobs/{baseline,long}/`，使用监控器的 `--kind training` 查看，
包括 `launch.json`、`train.log`、`train.exit`、`eval.log` 和 `eval.exit`。
这两组原计划在 GPU 2 上顺序运行，每组训练预算为 20,971,520 次环境交互。
后续按用户新指令停止该队列，保留基线第 50 轮 checkpoint，改为 16384 环境、4000 轮的长时序正式训练。
新任务日志位于 `remote_runs/duck_large_16384_4000_20260920/`，模型目录为
`outputs/revo3_right/run_rubber_duck_gait_long_16384_i4000_s42_20260920/`。
每 50 轮保存编号 checkpoint 和 `last.pth`；TensorBoard 指标逐轮记录。

最新调整：经过标称和随机化动力学的成对测试后，切换为 rollout 16、120 Hz 物理/PD、
6 个子步，策略仍为 20 Hz。新日志目录为 `remote_runs/duck_h16_120hz_16384_4000_20260920/`，
从备份的第 19 轮 checkpoint 续训到总计 4000 轮。结果和测试范围见 [FINGER_GAIT.md](FINGER_GAIT.md)。
回放这个新模型时显式传入 `--physics_hz 120`，以匹配训练动力学。

本机 SSH 控制连接和新任务的监控器已改为独立的临时用户服务
`brainco-monitor-connection.service`、`duck-h16-monitor-20260920.service`。
终端窗口只显示日志，关闭窗口不会停止后台监控。服务不保存密码；网络断开或本机重启后需要重新认证。

```bash
python3 scripts/monitor_duck_remote.py \
  --socket /tmp/brainco-collect.sock \
  --remote-run /home/tanjie/BrainCo_hora/remote_runs/duck_collect_20260920 \
  --local-dir outputs/remote_monitor/duck_collect_20260920
```

## 本地实际通过测试的环境

### TensorBoard 断线恢复（2026-09-20）

网页改为在本机 `127.0.0.1:16006` 运行，不再依赖 SSH 端口转发或远端 TensorBoard 进程。
`scripts/mirror_duck_events.py` 每 30 秒增量同步当前训练的事件文件到
`outputs/remote_monitor/duck_h16_120hz_16384_4000_20260920/stage1_tb/`。
同步中断保留最近数据；`tensorboard_sync.json` 记录最近成功同步时间和错误，
因此网页可用不等于曲线数据仍在实时更新。

```bash
bash scripts/duck_tensorboard.sh start
bash scripts/duck_tensorboard.sh status
bash scripts/duck_tensorboard.sh stop
```

`duck-tensorboard.service` 管理本机网页和同步进程，停止时释放本机 16006；远端不占用 6006/6007。
`brainco-monitor-connection.service` 使用 `scripts/brainco_ssh_connection.py`，
在两跳 SSH 上启用保活并在传输断开后重试。启动时通过权限 0600 的本机 Unix socket
接收认证信息，凭据仅留在进程内存用于重连，不写入脚本、日志或密码文件。
认证被拒绝时退出；进程/机器重启后需要重新提供认证信息。
这两个本机服务均不控制服务器的训练 tmux 会话。

| 组件 | 版本 |
|---|---|
| Python | 3.11 |
| PyTorch | 2.7.0+cu128 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab | 0.54.4，源码 commit `858234d06e6845d75420d2558e236526282e5da6` |
| NumPy | 1.26.0 |
| Gymnasium | 1.2.1 |
| OmegaConf | 2.3.1 |
| TensorBoard | 2.21.0 |

可设置 `ISAACLAB_PATH` 指向 Isaac Lab 启动脚本；本次远端任务使用 `PYTHON_BIN=/home/tanjie/.venvs/hora/bin/python`。上表记录本地验证版本，服务器已按兼容版本配置独立环境。

## 验证与训练

```bash
python -m unittest discover -s tests -p test_stage1_core.py -v
python -m unittest discover -s tests -p test_finger_gait.py -v
python -m unittest discover -s tests -p test_grasp_sampling.py -v
"$ISAACLAB_PATH" -p tests/check_stage1_integration.py

# 确认当前 GPU 使用情况后选择运行设备；上传时不占 GPU。
bash scripts/duck_gait.sh collect
bash scripts/duck_gait.sh train
```

新采集流程和奖励限制见 [FINGER_GAIT.md](FINGER_GAIT.md)。默认输出目录是 `outputs/revo3_right/run_rubber_duck_s1_gait_v2`。输入仍需重新采集并通过全部配额检查；随代码上传的旧 v2 缓存是种子数据，不能称为已完成的多样抓握库。
