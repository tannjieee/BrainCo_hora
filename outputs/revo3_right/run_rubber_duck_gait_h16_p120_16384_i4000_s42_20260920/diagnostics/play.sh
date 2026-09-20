#!/usr/bin/env bash
set -euo pipefail
cd /home/tan/hora2/BrainCo_hora
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
exec /home/tan/miniconda3/envs/env_isaaclab/bin/python -u train.py \
  --task rubber_duck --algo PPO --test --num_envs 1 --seed 12345 \
  --physics_hz 120 --finger_gait --rotation_speed 0.5 --episode_length_s 40 \
  --cache_file revo3_right_grasp_rubber_duck_gait_v2.npy --device cuda:0 \
  --checkpoint outputs/revo3_right/run_rubber_duck_gait_h16_p120_16384_i4000_s42_20260920/stage1_nn/last.pth \
  --output_name run_rubber_duck_gait_h16_p120_16384_i4000_s42_20260920 --real-time \
  --camera_eye 0.45 -0.55 1.95 --camera_lookat 0 -0.06 1.62 "$@"
