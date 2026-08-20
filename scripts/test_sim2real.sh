#!/usr/bin/env bash
# Lightweight dispatch and safety tests for scripts/sim2real.sh. No hardware or
# simulator is opened: /bin/echo stands in for each Python interpreter.

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
RUNNER="${SCRIPT_DIR}/sim2real.sh"

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

assert_contains() {
    local value="$1"
    local expected="$2"
    case "${value}" in
        *"${expected}"*) ;;
        *) fail "expected output to contain: ${expected}" ;;
    esac
}

assert_not_contains() {
    local value="$1"
    local unexpected="$2"
    case "${value}" in
        *"${unexpected}"*) fail "output unexpectedly contained: ${unexpected}" ;;
        *) ;;
    esac
}

help_output="$("${RUNNER}" --help)"
assert_contains "${help_output}" 'sim-trace'
assert_contains "${help_output}" 'real-trace'
assert_contains "${help_output}" 'replay'
assert_contains "${help_output}" 'joint-order'

sim_help_output="$("${RUNNER}" sim-trace --help)"
assert_contains "${sim_help_output}" '--checkpoint PATH'
assert_contains "${sim_help_output}" '--num_envs 1 --episodes 1 --headless'
assert_contains "${sim_help_output}" '--cache-row N'

sim_output="$(
    SIM_PY=/bin/echo "${RUNNER}" sim-trace \
        --checkpoint checkpoint.ckpt \
        --trace-npz sim.npz \
        --max_frames 3 2>&1
)"
assert_contains "${sim_output}" '/tools/dump_runtime_actions.py'
assert_contains "${sim_output}" '--num_envs 1 --episodes 1 --headless'

real_output="$(
    REAL_PY=/bin/echo "${RUNNER}" real-trace \
        --onnx policy.onnx \
        --metadata policy.yaml \
        --profile hand.yaml \
        --trace-npz real.npz \
        --policy-start-delay-s 15 2>&1
)"
assert_contains "${real_output}" '/deploy/revo3/scripts/run_policy.py'
assert_contains "${real_output}" '--steps 200'
assert_contains "${real_output}" '--policy-start-delay-s 15'
assert_not_contains "${real_output}" '--enable-motion'

real_equals_output="$(
    REAL_PY=/bin/echo "${RUNNER}" real-trace \
        --onnx=policy.onnx \
        --metadata=policy.yaml \
        --profile=hand.yaml \
        --trace-npz=real.npz \
        --steps=7 2>&1
)"
assert_contains "${real_equals_output}" '--steps=7'
assert_not_contains "${real_equals_output}" '--steps 200'

set +e
blocked_output="$(
    REAL_PY=/bin/echo "${RUNNER}" real-trace \
        --onnx policy.onnx \
        --metadata policy.yaml \
        --profile hand.yaml \
        --trace-npz real.npz \
        --enable-motion 2>&1
)"
blocked_status=$?
set -e
[[ "${blocked_status}" -ne 0 ]] || fail 'motion argument was accepted'
assert_contains "${blocked_output}" 'motor dry-run only'
assert_not_contains "${blocked_output}" '/deploy/revo3/scripts/run_policy.py'

set +e
unknown_output="$(
    REAL_PY=/bin/echo "${RUNNER}" real-trace \
        --onnx policy.onnx \
        --metadata policy.yaml \
        --profile hand.yaml \
        --trace-npz real.npz \
        --future-unsafe-option 2>&1
)"
unknown_status=$?
set -e
[[ "${unknown_status}" -ne 0 ]] || fail 'unknown real-trace argument was accepted'
assert_contains "${unknown_output}" 'not on the dry-run allowlist'

compare_output="$(REAL_PY=/bin/echo "${RUNNER}" compare sim.npz real.npz --max-steps 2)"
assert_contains "${compare_output}" '/tools/compare_policy_traces.py sim.npz real.npz --max-steps 2'

replay_output="$(
    REAL_PY=/bin/echo "${RUNNER}" replay \
        --trace-npz sim.npz \
        --profile hand.yaml \
        --frames 5 \
        --joint P05
)"
assert_contains "${replay_output}" '/deploy/revo3/scripts/replay_trace.py'
assert_contains "${replay_output}" '--trace-npz sim.npz'
assert_not_contains "${replay_output}" '--execute'

joint_order_output="$(
    REAL_PY=/bin/echo "${RUNNER}" joint-order status \
        --session mapping.json
)"
assert_contains "${joint_order_output}" '/deploy/revo3/scripts/joint_order.py'
assert_contains "${joint_order_output}" 'status --session mapping.json'
assert_not_contains "${joint_order_output}" '--execute'

validate_output="$(REAL_PY=/bin/echo "${RUNNER}" real-validate --help)"
assert_contains "${validate_output}" '/deploy/revo3/scripts/validate_policy.py --help'

export_output="$(SIM_PY=/bin/echo "${RUNNER}" export --help)"
assert_contains "${export_output}" '/tools/export_onnx.py --help'

set +e
relative_output="$(SIM_PY=relative/python "${RUNNER}" sim-trace --checkpoint x --trace-npz y 2>&1)"
relative_status=$?
set -e
[[ "${relative_status}" -ne 0 ]] || fail 'relative interpreter path was accepted'
assert_contains "${relative_output}" 'must be an absolute path'

printf '%s\n' 'PASS: sim2real environment runner dispatch and safety checks'
