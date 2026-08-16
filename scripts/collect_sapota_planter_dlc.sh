#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${BRAINCO_HORA_DIR:-/workspace/BrainCo_hora}"
ISAACLAB="${ISAACLAB_PATH:-/workspace/isaaclab}"
if [[ -d "${ISAACLAB}" ]]; then
    ISAACLAB="${ISAACLAB}/isaaclab.sh"
fi

TASK="${TASK:-sapota_planter}"
NUM_ENVS="${NUM_ENVS:-32768}"
TARGET_COUNT="${TARGET_COUNT:-8192}"
CACHE_FILE="${CACHE_FILE:-revo3_right_grasp_sapota_planter_six_axis.npy}"
DEVICE="${DEVICE:-cuda:0}"
HORA_CACHE_OUTPUT_ROOT="${HORA_CACHE_OUTPUT_ROOT:-}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "Project directory not found: ${PROJECT_DIR}" >&2
    exit 1
fi
if [[ ! -x "${ISAACLAB}" ]]; then
    echo "Isaac Lab launcher not found or not executable: ${ISAACLAB}" >&2
    exit 1
fi
if ! [[ "${NUM_ENVS}" =~ ^[0-9]+$ ]] || (( NUM_ENVS <= 0 )); then
    echo "NUM_ENVS must be a positive integer, got: ${NUM_ENVS}" >&2
    exit 1
fi
if ! [[ "${TARGET_COUNT}" =~ ^[0-9]+$ ]] || (( TARGET_COUNT <= 0 )); then
    echo "TARGET_COUNT must be a positive integer, got: ${TARGET_COUNT}" >&2
    exit 1
fi

cd "${PROJECT_DIR}"
LOCAL_CACHE="${PROJECT_DIR}/cache/${CACHE_FILE}"
DEST_CACHE=""
if [[ -n "${HORA_CACHE_OUTPUT_ROOT}" ]]; then
    mkdir -p "${HORA_CACHE_OUTPUT_ROOT}"
    DEST_CACHE="${HORA_CACHE_OUTPUT_ROOT%/}/${CACHE_FILE}"
else
    echo "[WARN] HORA_CACHE_OUTPUT_ROOT is unset; the cache will remain inside the DLC container." >&2
fi

if [[ "${FORCE_OVERWRITE}" != "1" ]]; then
    if [[ -e "${LOCAL_CACHE}" ]]; then
        echo "Local cache already exists: ${LOCAL_CACHE}; set FORCE_OVERWRITE=1 to replace it." >&2
        exit 1
    fi
    if [[ -n "${DEST_CACHE}" && -e "${DEST_CACHE}" ]]; then
        echo "Persistent cache already exists: ${DEST_CACHE}; set FORCE_OVERWRITE=1 to replace it." >&2
        exit 1
    fi
fi

COLLECT_CMD=(
    "${ISAACLAB}" -p gen_grasp.py
    --task "${TASK}"
    --num_envs "${NUM_ENVS}"
    --target_count "${TARGET_COUNT}"
    --cache_file "${CACHE_FILE}"
    --gravity_mode six_axis
    --device "${DEVICE}"
    --headless
)
COLLECT_CMD+=("$@")

printf '[INFO] Grasp collection command:'
printf ' %q' "${COLLECT_CMD[@]}"
printf '\n'
if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

"${ISAACLAB}" -p -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU is not visible"; print("GPU:", torch.cuda.get_device_name(0)); print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))'
"${COLLECT_CMD[@]}"

if [[ ! -f "${LOCAL_CACHE}" ]]; then
    echo "Collection exited without creating: ${LOCAL_CACHE}" >&2
    exit 1
fi
CACHE_TO_VALIDATE="${LOCAL_CACHE}" EXPECTED_COUNT="${TARGET_COUNT}" "${ISAACLAB}" -p -c 'import os, numpy as np; p=os.environ["CACHE_TO_VALIDATE"]; a=np.load(p, allow_pickle=False); assert a.ndim == 2 and a.shape[0] == int(os.environ["EXPECTED_COUNT"]), a.shape; assert np.isfinite(a).all(); print("cache:", p); print("shape:", a.shape, "dtype:", a.dtype, "finite: True")'
sha256sum "${LOCAL_CACHE}"

if [[ -n "${DEST_CACHE}" ]]; then
    DEST_TMP="${DEST_CACHE}.partial.$$"
    cp "${LOCAL_CACHE}" "${DEST_TMP}"
    mv -f "${DEST_TMP}" "${DEST_CACHE}"
    sha256sum "${DEST_CACHE}"
    echo "[INFO] Persistent cache saved: ${DEST_CACHE}"
fi
