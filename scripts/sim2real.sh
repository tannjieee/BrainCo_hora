#!/usr/bin/env bash
# Run the simulator and Revo3 deployment tools with their own Python
# interpreters.  This script intentionally does not activate Conda and never
# evaluates a caller-provided shell string.

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(dirname -- "${SCRIPT_DIR}")"

SIM_PY="${SIM_PY:-/home/tan/miniconda3/envs/env_isaaclab/bin/python}"
REAL_PY="${REAL_PY:-/home/tan/miniconda3/envs/revo3/bin/python}"

readonly SCRIPT_DIR REPO_ROOT SIM_PY REAL_PY

die() {
    printf 'sim2real: %s\n' "$*" >&2
    exit 2
}

usage() {
    cat <<'EOF'
Usage: scripts/sim2real.sh COMMAND [ARG ...]

Use the Isaac Lab and Revo3 Conda environments without activating either one.

Commands:
  env-check       Check both Python executables and required modules.
  sim-trace       Record a single-env simulator trace (Isaac Lab Python).
  real-trace      Record a hardware dry-run trace (Revo3 Python, no motor commands).
  compare         Compare simulator and real .npz traces (Revo3 Python).
  replay          Inspect/preflight/replay a validated simulator target trace.
  joint-order     Plan/status/probe a one-joint-at-a-time mapping session.
  offset-cal      Create/show/adjust versioned sim-to-real offset profiles.
  export          Export a Stage-2 checkpoint to ONNX (Isaac Lab Python).
  real-validate   Validate ONNX/metadata/profile offline (Revo3 Python).
  help            Show this help.

Interpreter overrides (must be absolute executable paths):
  SIM_PY=/path/to/env_isaaclab/bin/python
  REAL_PY=/path/to/revo3/bin/python

Examples:
  scripts/sim2real.sh env-check

  scripts/sim2real.sh sim-trace \
    --task cylinder --algo ProprioAdapt \
    --checkpoint outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt \
    --onnx outputs/revo3_right/onnx/cylinder_stage2.onnx \
    --cache-row 7942 \
    --max_frames 200 \
    --trace-npz outputs/revo3_right/traces/sim_cylinder.npz

  scripts/sim2real.sh real-trace \
    --onnx outputs/revo3_right/onnx/cylinder_stage2.onnx \
    --metadata outputs/revo3_right/onnx/cylinder_stage2.deploy_meta.yaml \
    --profile deploy/revo3/config/revo3_right.yaml \
    --steps 200 \
    --policy-start-delay-s 15 \
    --trace-npz outputs/revo3_right/traces/real_dryrun_cylinder.npz

  scripts/sim2real.sh compare \
    outputs/revo3_right/traces/sim_cylinder.npz \
    outputs/revo3_right/traces/real_dryrun_cylinder.npz \
    --onnx outputs/revo3_right/onnx/cylinder_stage2.onnx

  # Offline by default: validates the checkpoint SHA and prints Pidx -> SDK Midx.
  scripts/sim2real.sh replay \
    --trace-npz outputs/revo3_right/traces/sim_joint_order_replay.npz \
    --checkpoint outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt \
    --profile deploy/revo3/config/revo3_right.yaml \
    --frames 10 --joint P05

  # Offline: generate a resumable 21-joint mapping plan; no hardware is opened.
  scripts/sim2real.sh joint-order init \
    --trace-npz outputs/revo3_right/traces/sim_joint_order_replay.npz \
    --checkpoint outputs/revo3_right/stage2_smoke2/stage2_nn/model_best_latent.ckpt \
    --profile deploy/revo3/config/revo3_right.yaml \
    --session outputs/revo3_right/joint_order/session.json

Run `scripts/sim2real.sh real-trace --help` for its deliberately restricted
hardware options. Other subcommands pass their arguments as an argv array to
the named Python tool; no shell evaluation is performed. The replay tool is
offline unless --preflight or the explicitly confirmed --execute mode is used.
EOF
}

real_trace_usage() {
    cat <<'EOF'
Usage: scripts/sim2real.sh real-trace OPTIONS

Connect to the Revo3, read observations, run ONNX inference, and save a trace.
This entry point is motor dry-run only: it has no option that can send a motor
command. A default of --steps 200 is added when --steps is omitted.

Required:
  --onnx PATH
  --metadata PATH
  --profile PATH
  --trace-npz PATH

Allowed optional arguments:
  --port PORT
  --baudrate N
  --slave-id N
  --rate HZ
  --steps N
  --print-every N
  --provider cpu|cuda
  --policy-start-delay-s SECONDS
  --preflight-only
  --preflight-position-tolerance-deg DEG

Arguments may also use --name=value. All other arguments are rejected before
the hardware runtime is started. Use the deployment CLI directly for a
separately reviewed and explicitly authorized motion procedure.
EOF
}

sim_trace_usage() {
    cat <<'EOF'
Usage: scripts/sim2real.sh sim-trace OPTIONS

Record one simulator episode through tools/dump_runtime_actions.py. This
wrapper always appends --num_envs 1 --episodes 1 --headless so the resulting
trace has an unambiguous frame sequence.

Required:
  --checkpoint PATH
  --trace-npz PATH

Common optional arguments:
  --task ball|cylinder
  --algo auto|PPO|ProprioAdapt
  --onnx PATH
  --max_frames N
  --episode-length-s SECONDS
  --seed N
  --cache_file NAME
  --cache-row N
  --usd PATH

Other Isaac Lab application arguments are passed as an argv array. No shell
evaluation is performed.
EOF
}

require_python() {
    local label="$1"
    local path="$2"

    [[ "${path}" = /* ]] || die "${label} must be an absolute path: ${path}"
    [[ -f "${path}" && -x "${path}" ]] || die "${label} is not an executable file: ${path}"
}

require_tool() {
    local path="$1"
    [[ -f "${path}" ]] || die "tool not found: ${path}"
}

run_tool() {
    local label="$1"
    local python_path="$2"
    local tool_path="$3"
    shift 3

    require_python "${label}" "${python_path}"
    require_tool "${tool_path}"
    cd -- "${REPO_ROOT}"
    exec "${python_path}" "${tool_path}" "$@"
}

has_option() {
    local wanted="$1"
    shift
    local argument
    for argument in "$@"; do
        case "${argument}" in
            "${wanted}"|"${wanted}"=*) return 0 ;;
        esac
    done
    return 1
}

require_option() {
    local option="$1"
    shift
    has_option "${option}" "$@" || die "missing required option: ${option}"
}

check_environment() {
    require_python "SIM_PY" "${SIM_PY}"
    require_python "REAL_PY" "${REAL_PY}"

    printf 'SIM_PY=%s\n' "${SIM_PY}"
    local status=0
    if ! "${SIM_PY}" -c '
import importlib.util
import sys
required = ("isaaclab", "numpy", "omegaconf", "torch")
missing = [name for name in required if importlib.util.find_spec(name) is None]
print(f"  Python {sys.version.split()[0]} ({sys.executable})")
print("  modules: " + ("OK: " + ", ".join(required) if not missing else "MISSING: " + ", ".join(missing)))
raise SystemExit(bool(missing))
'; then
        status=1
    fi

    printf 'REAL_PY=%s\n' "${REAL_PY}"
    if ! "${REAL_PY}" -c '
import importlib.util
import sys
required = ("bc_revo3_sdk", "numpy", "onnxruntime", "yaml")
optional = ("pyvitaisdk",)
missing = [name for name in required if importlib.util.find_spec(name) is None]
optional_missing = [name for name in optional if importlib.util.find_spec(name) is None]
print(f"  Python {sys.version.split()[0]} ({sys.executable})")
print("  modules: " + ("OK: " + ", ".join(required) if not missing else "MISSING: " + ", ".join(missing)))
if optional_missing:
    print("  optional modules missing: " + ", ".join(optional_missing))
raise SystemExit(bool(missing))
'; then
        status=1
    fi
    return "${status}"
}

run_sim_trace() {
    if [[ "$#" -eq 1 && ( "$1" = "-h" || "$1" = "--help" ) ]]; then
        sim_trace_usage
        return
    fi

    require_option --checkpoint "$@"
    require_option --trace-npz "$@"
    printf '%s\n' '[sim2real] sim-trace: forcing --num_envs 1 --episodes 1 --headless.' >&2
    run_tool "SIM_PY" "${SIM_PY}" "${REPO_ROOT}/tools/dump_runtime_actions.py" \
        "$@" --num_envs 1 --episodes 1 --headless
}

REAL_TRACE_ARGS=()
REAL_TRACE_SEEN_ONNX=0
REAL_TRACE_SEEN_METADATA=0
REAL_TRACE_SEEN_PROFILE=0
REAL_TRACE_SEEN_TRACE=0
REAL_TRACE_SEEN_STEPS=0

mark_real_trace_option() {
    local option="$1"
    case "${option}" in
        --onnx) REAL_TRACE_SEEN_ONNX=1 ;;
        --metadata) REAL_TRACE_SEEN_METADATA=1 ;;
        --profile) REAL_TRACE_SEEN_PROFILE=1 ;;
        --trace-npz) REAL_TRACE_SEEN_TRACE=1 ;;
        --steps) REAL_TRACE_SEEN_STEPS=1 ;;
    esac
}

parse_real_trace_args() {
    local option
    while [[ "$#" -gt 0 ]]; do
        option="$1"
        case "${option}" in
            --preflight-only)
                REAL_TRACE_ARGS+=("${option}")
                shift
                ;;
            --onnx|--metadata|--profile|--trace-npz|--port|--baudrate|--slave-id|--rate|--steps|--print-every|--provider|--policy-start-delay-s|--preflight-position-tolerance-deg)
                [[ "$#" -ge 2 ]] || die "${option} requires a value"
                [[ "$2" != --* ]] || die "${option} requires a value, got option: $2"
                REAL_TRACE_ARGS+=("${option}" "$2")
                mark_real_trace_option "${option}"
                shift 2
                ;;
            --onnx=*|--metadata=*|--profile=*|--trace-npz=*|--port=*|--baudrate=*|--slave-id=*|--rate=*|--steps=*|--print-every=*|--provider=*|--policy-start-delay-s=*|--preflight-position-tolerance-deg=*)
                REAL_TRACE_ARGS+=("${option}")
                mark_real_trace_option "${option%%=*}"
                shift
                ;;
            --enable-motion|--enable-motion=*)
                die "real-trace is motor dry-run only; motion arguments are not accepted"
                ;;
            *)
                die "real-trace argument is not on the dry-run allowlist: ${option}"
                ;;
        esac
    done

    [[ "${REAL_TRACE_SEEN_ONNX}" -eq 1 ]] || die 'real-trace requires --onnx'
    [[ "${REAL_TRACE_SEEN_METADATA}" -eq 1 ]] || die 'real-trace requires --metadata'
    [[ "${REAL_TRACE_SEEN_PROFILE}" -eq 1 ]] || die 'real-trace requires --profile'
    [[ "${REAL_TRACE_SEEN_TRACE}" -eq 1 ]] || die 'real-trace requires --trace-npz'
    if [[ "${REAL_TRACE_SEEN_STEPS}" -eq 0 ]]; then
        REAL_TRACE_ARGS+=(--steps 200)
    fi
}

run_real_trace() {
    if [[ "$#" -eq 1 && ( "$1" = "-h" || "$1" = "--help" ) ]]; then
        real_trace_usage
        return
    fi

    parse_real_trace_args "$@"
    printf '%s\n' '[sim2real] real-trace: MOTOR DRY-RUN; observations/inference only.' >&2
    run_tool "REAL_PY" "${REAL_PY}" "${REPO_ROOT}/deploy/revo3/scripts/run_policy.py" \
        "${REAL_TRACE_ARGS[@]}"
}

main() {
    local command="${1:-help}"
    if [[ "$#" -gt 0 ]]; then
        shift
    fi

    case "${command}" in
        help|-h|--help)
            usage
            ;;
        env-check)
            [[ "$#" -eq 0 ]] || die 'env-check does not accept arguments'
            check_environment
            ;;
        sim-trace)
            run_sim_trace "$@"
            ;;
        real-trace)
            run_real_trace "$@"
            ;;
        compare)
            if [[ "$#" -eq 0 ]]; then
                die 'compare requires SIM_TRACE REAL_TRACE (or --help)'
            fi
            run_tool "REAL_PY" "${REAL_PY}" "${REPO_ROOT}/tools/compare_policy_traces.py" "$@"
            ;;
        replay)
            if [[ "$#" -eq 0 ]]; then
                die 'replay requires replay_trace.py arguments (or --help)'
            fi
            run_tool "REAL_PY" "${REAL_PY}" "${REPO_ROOT}/deploy/revo3/scripts/replay_trace.py" "$@"
            ;;
        joint-order)
            if [[ "$#" -eq 0 ]]; then
                die 'joint-order requires init, status, or probe arguments (or --help)'
            fi
            run_tool "REAL_PY" "${REAL_PY}" "${REPO_ROOT}/deploy/revo3/scripts/joint_order.py" "$@"
            ;;
        offset-cal)
            if [[ "$#" -eq 0 ]]; then
                die 'offset-cal requires init, show, or adjust arguments (or --help)'
            fi
            run_tool "REAL_PY" "${REAL_PY}" "${REPO_ROOT}/deploy/revo3/scripts/offset_calibration.py" "$@"
            ;;
        export)
            if [[ "$#" -eq 0 ]]; then
                die 'export requires tools/export_onnx.py arguments (or --help)'
            fi
            run_tool "SIM_PY" "${SIM_PY}" "${REPO_ROOT}/tools/export_onnx.py" "$@"
            ;;
        real-validate)
            if [[ "$#" -eq 0 ]]; then
                die 'real-validate requires validate_policy.py arguments (or --help)'
            fi
            run_tool "REAL_PY" "${REAL_PY}" "${REPO_ROOT}/deploy/revo3/scripts/validate_policy.py" "$@"
            ;;
        *)
            die "unknown command: ${command} (run 'scripts/sim2real.sh help')"
            ;;
    esac
}

main "$@"
