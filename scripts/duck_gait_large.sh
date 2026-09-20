#!/usr/bin/env bash
# 16384 environments, 4000 PPO iterations; shorter rollouts for frequent updates.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export DUCK_ENVS=${DUCK_ENVS:-16384}
export DUCK_RUN=${DUCK_RUN:-run_rubber_duck_gait_h16_16384_i4000_s42}
export DUCK_SPEED=${DUCK_SPEED:-0.5}
bash scripts/duck_gait.sh train --seed 42 --horizon_length 16 --gamma 0.995 \
    --episode_length_s 40 --minibatch_size 32768 --max_iterations 4000 \
    --save_frequency 50 --physics_hz "${DUCK_PHYSICS_HZ:-240}" "$@"
