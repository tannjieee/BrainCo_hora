#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
COLLECT_PYTHON="${COLLECT_PYTHON:-/home/tan/miniconda3/envs/env_isaaclab_bc/bin/python}"
COLLECT_NUM_ENVS="${COLLECT_NUM_ENVS:-8192}"
COLLECT_LOG_DIR="${COLLECT_LOG_DIR:-${PROJECT_ROOT}/logs/grasp_collection}"
GRASP_CACHE_VALIDATOR="${PROJECT_ROOT}/tools/validate_grasp_cache.py"

RADII_MM=(30 25 35 26 34 27 33 28 32 29 31)

mkdir -p "${COLLECT_LOG_DIR}"
cd "${PROJECT_ROOT}"

validate_cache() {
    local cache_path="$1"
    local expected_count="$2"
    "${COLLECT_PYTHON}" "${GRASP_CACHE_VALIDATOR}" \
        --cache "${cache_path}" \
        --expected-count "${expected_count}"
}

for radius_mm in "${RADII_MM[@]}"; do
    if [[ "${radius_mm}" -eq 30 ]]; then
        target_count=4096
    else
        target_count=512
    fi

    cache_path="${PROJECT_ROOT}/cache/revo3_right_grasp_cylinder_r${radius_mm}mm.npy"
    log_path="${COLLECT_LOG_DIR}/r${radius_mm}mm.log"

    if [[ -f "${cache_path}" ]]; then
        echo "[VERIFY] Existing final cache: ${cache_path}" | tee -a "${log_path}"
        validate_cache "${cache_path}" "${target_count}" 2>&1 | tee -a "${log_path}"
        echo "[SKIP] Valid final cache: ${cache_path}" | tee -a "${log_path}"
        continue
    fi

    echo "[START] radius=${radius_mm}mm target=${target_count} envs=${COLLECT_NUM_ENVS}"
    "${COLLECT_PYTHON}" "${PROJECT_ROOT}/gen_grasp.py" \
        --task cylinder \
        --cylinder_radius_mm "${radius_mm}" \
        --num_envs "${COLLECT_NUM_ENVS}" \
        --target_count "${target_count}" \
        --episode_length_s 4 \
        --checkpoint_interval 60 \
        --progress_interval 10 \
        --settle_steps 15 \
        --contact_window_steps 15 \
        --min_contact_ratio 0.60 \
        --min_contact_fingertips 4 \
        --min_live_contact_fingertips 2 \
        --contact_force_threshold 0.05 \
        --max_axis_tilt_deg 10 \
        --max_horizontal_drift_m 0.005 \
        --max_height_drift_m 0.015 \
        --gravity_mode sphere \
        --device cuda:0 \
        --headless \
        2>>"${log_path}" | tee -a "${log_path}"
    echo "[VERIFY] radius=${radius_mm}mm cache=${cache_path}" | tee -a "${log_path}"
    validate_cache "${cache_path}" "${target_count}" 2>&1 | tee -a "${log_path}"
    echo "[DONE] radius=${radius_mm}mm cache=${cache_path}" | tee -a "${log_path}"
done

echo "[COMPLETE] All requested cylinder caches are present."
