#!/usr/bin/env bash
# Equal-transition screening runs. Use separate output names for every seed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
profile=${1:?Usage: duck_gait_compare.sh baseline|long|eval [arguments]}
shift
case "$profile" in
    baseline) horizon=16; gamma=0.99; seconds=20 ;;
    long) horizon=64; gamma=0.995; seconds=40 ;;
    eval)
        checkpoint=${1:?Provide a checkpoint}; shift
        exec bash scripts/duck_gait.sh eval "$checkpoint" \
            --episode_length_s 60 --test_steps 1200 --seed 12345 "$@"
        ;;
    *) echo "Unknown profile: $profile" >&2; exit 2 ;;
esac
export DUCK_RUN=${DUCK_RUN:-run_rubber_duck_gait_${profile}_s42}
# 20,971,520 is divisible by both 2048*16 and 2048*64.
bash scripts/duck_gait.sh train --seed 42 --horizon_length "$horizon" \
    --gamma "$gamma" --episode_length_s "$seconds" --minibatch_size 32768 \
    --max_agent_steps 20971520 "$@"
