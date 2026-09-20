#!/usr/bin/env bash
# Explicit actions only: uploading this project never starts training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TERM="${TERM:-xterm}"
if [[ "$TERM" == dumb ]]; then export TERM=xterm; fi
export PYTHONUNBUFFERED=1
DUCK_CACHE=${DUCK_CACHE:-revo3_right_grasp_rubber_duck_gait_v2.npy}
DUCK_ENVS=${DUCK_ENVS:-2048}
DUCK_RUN=${DUCK_RUN:-run_rubber_duck_s1_gait_v2}
DUCK_SPEED=${DUCK_SPEED:-0.5}

run_python() {
    if [[ -n "${ISAACLAB_PATH:-}" ]]; then
        local launcher=$ISAACLAB_PATH
        [[ ! -d "$launcher" ]] || launcher="$launcher/isaaclab.sh"
        "$launcher" -p "$@"
    else
        "${PYTHON_BIN:-python}" "$@"
    fi
}

mode=${1:-help}
[[ $# -eq 0 ]] || shift
case "$mode" in
    collect)
        run_python gen_grasp.py --task rubber_duck --headless --num_envs "$DUCK_ENVS" \
            --seed_cache revo3_right_grasp_rubber_duck_v2.npy --noise_scale .15 \
            --cache_file "$DUCK_CACHE" --target_count 8192 --yaw_bins 8 \
            --free_fingers index middle ring little --min_contact_fingertips 3 \
            --min_live_contact_fingertips 3 --episode_length_s 5 --max_steps 20000 "$@"
        ;;
    train)
        # Require a completed collection with the exact recorded cache contents.
        python3 - "$DUCK_CACHE" <<'PY'
import hashlib, json, sys
from pathlib import Path
name = sys.argv[1]
if Path(name).name != name:
    raise SystemExit('DUCK_CACHE must be a filename under cache/')
cache = Path('cache', name)
report = json.loads(cache.with_suffix('.report.json').read_text())
if not report.get('complete') or report['accepted'] != report['targets']:
    raise SystemExit('Collection coverage is incomplete; inspect the report before training')
if hashlib.sha256(cache.read_bytes()).hexdigest() != report.get('cache_sha256'):
    raise SystemExit('Cache differs from its validated collection report')
PY
        run_python train.py --task rubber_duck --algo PPO --headless --finger_gait \
            --rotation_speed "$DUCK_SPEED" --fixed_train_gravity 9.81 \
            --cache_file "$DUCK_CACHE" --num_envs "$DUCK_ENVS" \
            --output_name "$DUCK_RUN" "$@"
        ;;
    eval)
        checkpoint=${1:?Usage: scripts/duck_gait.sh eval CHECKPOINT [options]}
        shift
        run_python train.py --task rubber_duck --algo PPO --headless --test --test_steps 400 \
            --finger_gait --rotation_speed "$DUCK_SPEED" --num_envs 128 \
            --cache_file "$DUCK_CACHE" --checkpoint "$checkpoint" \
            --output_name "${DUCK_RUN}_eval" "$@"
        ;;
    *)
        echo 'Usage: bash scripts/duck_gait.sh {collect|train|eval CHECKPOINT} [CLI options]'
        echo 'Activate an Isaac Lab environment or set ISAACLAB_PATH / PYTHON_BIN first.'
        echo 'Optional: DUCK_ENVS, DUCK_CACHE, DUCK_RUN, DUCK_SPEED. Collection and training are separate.'
        [[ "$mode" == help ]]
        ;;
esac
