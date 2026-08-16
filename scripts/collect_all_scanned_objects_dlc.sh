#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${BRAINCO_HORA_DIR:-/workspace/BrainCo_hora}"
WORKER="${GRASP_WORKER:-${PROJECT_DIR}/scripts/collect_scanned_object_dlc.sh}"
OUTPUT_ROOT="${HORA_CACHE_OUTPUT_ROOT:-/mnt/nas/tanjie/BrainCo_hora_cache}"
NUM_ENVS="${NUM_ENVS:-32768}"
TARGET_COUNT="${TARGET_COUNT:-8192}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

TASKS=(
    great_dinos_triceratops
    perricone_eye_cream
    qabsorb_coq10
    sapota_planter
    toys_r_us_foobler
    wilton_sprinkles
)

if [[ ! -x "${WORKER}" ]]; then
    echo "Grasp worker not found or not executable: ${WORKER}" >&2
    exit 1
fi
if [[ ! -d /mnt/nas ]]; then
    echo "NAS is not mounted at /mnt/nas; refusing to collect ephemeral-only caches." >&2
    exit 1
fi
mkdir -p "${OUTPUT_ROOT}"

cd "${PROJECT_DIR}"
for task in "${TASKS[@]}"; do
    cache_file="revo3_right_grasp_${task}_six_axis.npy"
    dest_cache="${OUTPUT_ROOT%/}/${cache_file}"
    local_cache="${PROJECT_DIR}/cache/${cache_file}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        if [[ -f "${dest_cache}" && "${FORCE_OVERWRITE}" != "1" ]]; then
            echo "[DRY-RUN SKIP] ${task}: persistent cache exists"
        elif [[ -f "${local_cache}" && "${FORCE_OVERWRITE}" != "1" ]]; then
            echo "[DRY-RUN REUSE] ${task}: image cache will be copied to NAS"
        else
            echo "[DRY-RUN COLLECT] ${task}"
            TASK="${task}" CACHE_FILE="${cache_file}" NUM_ENVS="${NUM_ENVS}" TARGET_COUNT="${TARGET_COUNT}" HORA_CACHE_OUTPUT_ROOT="${OUTPUT_ROOT}" FORCE_OVERWRITE="${FORCE_OVERWRITE}" DRY_RUN=1 "${WORKER}" "$@"
        fi
        continue
    fi

    if [[ -f "${dest_cache}" && "${FORCE_OVERWRITE}" != "1" ]]; then
        echo "[SKIP] Persistent cache already exists: ${dest_cache}"
        continue
    fi

    if [[ -f "${local_cache}" && "${FORCE_OVERWRITE}" != "1" ]]; then
        dest_tmp="${dest_cache}.partial.$$"
        echo "[REUSE] Copying image cache to NAS: ${local_cache}"
        cp "${local_cache}" "${dest_tmp}"
        mv -f "${dest_tmp}" "${dest_cache}"
        sha256sum "${dest_cache}"
        continue
    fi

    echo "[COLLECT] ${task}"
    TASK="${task}" \
    CACHE_FILE="${cache_file}" \
    NUM_ENVS="${NUM_ENVS}" \
    TARGET_COUNT="${TARGET_COUNT}" \
    HORA_CACHE_OUTPUT_ROOT="${OUTPUT_ROOT}" \
    FORCE_OVERWRITE="${FORCE_OVERWRITE}" \
    "${WORKER}" "$@"
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[INFO] Dry-run validation completed; no collection was started."
else
    echo "[INFO] All requested scanned-object caches are available under: ${OUTPUT_ROOT}"
    find "${OUTPUT_ROOT}" -maxdepth 1 -type f -name 'revo3_right_grasp_*_six_axis.npy' -printf '%f %s bytes\n' | sort
fi
