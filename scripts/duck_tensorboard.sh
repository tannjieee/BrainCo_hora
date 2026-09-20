#!/usr/bin/env bash
# Local TensorBoard stays available while the remote SSH connection recovers.
set -euo pipefail
unit=duck-tensorboard.service
script=$(realpath "${BASH_SOURCE[0]}")
root=$(dirname "$(dirname "$script")")
events="$root/outputs/remote_monitor/duck_h16_120hz_16384_4000_20260920/stage1_tb"
tensorboard=/home/tan/miniconda3/envs/env_isaaclab/bin/tensorboard

case "${1:-start}" in
  start)
    if ! systemctl --user is-active --quiet "$unit"; then
      systemd-run --user --unit="$unit" --collect --service-type=exec \
        --property=KillMode=control-group --property=TimeoutStopSec=10 \
        --property=Restart=on-failure --property=RestartSec=5 \
        /bin/bash "$script" serve
    fi
    echo 'TensorBoard: http://127.0.0.1:16006 (open on this computer)'
    echo 'Remote data refreshes every 30s; disconnected sessions retain the last synced curves.'
    echo "Stop: bash $script stop"
    ;;
  stop)
    systemctl --user stop "$unit"
    ;;
  status)
    systemctl --user status "$unit" --no-pager
    [[ ! -f "$(dirname "$events")/tensorboard_sync.json" ]] || cat "$(dirname "$events")/tensorboard_sync.json"
    ;;
  serve)
    mkdir -p "$events"
    mirror_pid=''
    tb_pid=''
    cleanup() {
      trap - EXIT INT TERM HUP
      for pid in "$mirror_pid" "$tb_pid"; do
        if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; fi
      done
      wait 2>/dev/null || true
    }
    trap cleanup EXIT
    trap 'exit 0' INT TERM HUP
    /usr/bin/python3 -u "$root/scripts/mirror_duck_events.py" --destination "$events" &
    mirror_pid=$!
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$tensorboard" \
      --host 127.0.0.1 --port 16006 --load_fast=false --logdir "$events" &
    tb_pid=$!
    # Restart the whole service if either child exits, even with exit code zero.
    wait -n "$mirror_pid" "$tb_pid" && exit 1
    ;;
  *) echo "Usage: bash $script {start|stop|status}" >&2; exit 2 ;;
esac
