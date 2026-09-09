#!/usr/bin/env bash
# Run the project in the IsaacLab DSW image, with all outputs on mounted NAS.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_LAUNCHER="${ISAACLAB_PATH:-/workspace/isaaclab/isaaclab.sh}"
NAS_OUTPUT_DIR="${HORA_DSW_OUTPUT_DIR:-/mnt/nas/BrainCo_hora/outputs}"

# DSW interactive shells commonly export ISAACLAB_PATH as the install
# directory. Directories can pass -x (search permission), but cannot be exec'd.
if [[ -d "$ISAACLAB_LAUNCHER" ]]; then
    ISAACLAB_LAUNCHER="${ISAACLAB_LAUNCHER%/}/isaaclab.sh"
fi
if [[ ! -f "$ISAACLAB_LAUNCHER" || ! -x "$ISAACLAB_LAUNCHER" ]]; then
    printf 'Invalid IsaacLab launcher: %s (ISAACLAB_PATH must name an IsaacLab directory or executable isaaclab.sh).\n' "$ISAACLAB_LAUNCHER" >&2
    exit 1
fi
if ! mountpoint -q /mnt/nas; then
    printf '/mnt/nas is not mounted; refusing to write training outputs to the container disk.\n' >&2
    exit 1
fi
if [[ ! -d "$NAS_OUTPUT_DIR" || ! -w "$NAS_OUTPUT_DIR" || ! -L "$PROJECT_DIR/outputs" ]]; then
    printf 'Missing writable NAS outputs or project outputs symlink. See DEPLOY_DSW.md.\n' >&2
    exit 1
fi
if [[ "$(readlink -f -- "$PROJECT_DIR/outputs")" != "$(readlink -f -- "$NAS_OUTPUT_DIR")" ]]; then
    printf 'Project outputs does not point to the configured NAS output directory.\n' >&2
    exit 1
fi

export ISAACLAB_PATH="$ISAACLAB_LAUNCHER"
export PYTHONUNBUFFERED=1
export TERM="${TERM:-xterm-256color}"
if [[ "$TERM" == dumb ]]; then export TERM=xterm-256color; fi
if [[ "$(id -u)" == 0 ]]; then export OMNI_KIT_ALLOW_ROOT=1; fi
cd -- "$PROJECT_DIR"

MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
    stage1) exec bash scripts/train_s1.sh "$@" --headless ;;
    stage2) exec bash scripts/train_s2.sh "$@" --headless ;;
    python) exec "$ISAACLAB_LAUNCHER" -p "$@" ;;
    *)
        printf 'Usage: bash scripts/dsw.sh {stage1|stage2|python} [arguments...]\n' >&2
        exit 2
        ;;
esac
