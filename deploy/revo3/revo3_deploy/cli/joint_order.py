from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager, nullcontext
import fcntl
import getpass
import os
import secrets
import signal
import sys
import termios
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np

from revo3_deploy.cli.replay_trace import (
    async_main as replay_async_main,
    build_parser as replay_build_parser,
)
from revo3_deploy.joint_order_session import (
    create_session,
    load_session,
    resolve_session_joint,
    save_session,
    select_candidate,
    utc_now,
    validate_session_plan,
    verify_session_artifacts,
)
from revo3_deploy.replay_trace import ReplayTrace
from revo3_deploy.robot_profile import Revo3Profile


Prompt = Callable[[str], str]
VERDICTS = {
    "MATCH": "passed",
    "OPPOSITE": "wrong_direction",
    "WRONG_JOINT": "wrong_joint",
    "NO_MOTION": "no_motion",
    "MULTIPLE": "multiple_joints",
    "UNCERTAIN": "uncertain",
}


@contextmanager
def _session_lock(session_path: str | Path):
    path = Path(session_path).expanduser().resolve()
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another joint-order process holds the session lock: {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class TtyPrompter:
    def __init__(self) -> None:
        try:
            self._fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
        except OSError as exc:
            raise RuntimeError(
                "Joint-order motion requires a controlling TTY; piped or unattended "
                "execution is refused."
            ) from exc
        if not os.isatty(self._fd):
            os.close(self._fd)
            raise RuntimeError("Joint-order motion requires a real controlling TTY.")

    async def ask(self, message: str) -> str:
        pending = memoryview(message.encode("utf-8"))
        while pending:
            pending = pending[os.write(self._fd, pending):]
        value = bytearray()
        loop = asyncio.get_running_loop()
        result: asyncio.Future[str] = loop.create_future()

        def read_one_byte() -> None:
            if result.done():
                return
            try:
                chunk = os.read(self._fd, 1)
                if not chunk:
                    raise EOFError("Controlling TTY closed; no approval was granted.")
                if chunk == b"\n":
                    result.set_result(value.decode("utf-8", errors="strict"))
                    return
                if chunk != b"\r":
                    value.extend(chunk)
                if len(value) > 4096:
                    raise ValueError("TTY response exceeded the 4096-byte safety limit.")
            except BaseException as exc:
                result.set_exception(exc)

        loop.add_reader(self._fd, read_one_byte)
        try:
            return await result
        finally:
            loop.remove_reader(self._fd)

    def close(self) -> None:
        os.close(self._fd)

    def discard_pending_input(self) -> None:
        """Drop text pasted before the post-motion observation challenge."""
        termios.tcflush(self._fd, termios.TCIFLUSH)

    def __enter__(self) -> "TtyPrompter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create and run a fail-closed, one-joint-per-process Revo3 joint-order "
            "test session. Session initialization and status are fully offline."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("init", help="Build the 21-joint offline plan.")
    initialize.add_argument("--trace-npz", required=True)
    initialize.add_argument("--checkpoint", required=True)
    initialize.add_argument("--profile", required=True)
    initialize.add_argument("--session", required=True)
    initialize.add_argument("--min-delta-deg", type=float, default=1.0)
    initialize.add_argument("--max-delta-deg", type=float, default=2.5)

    status = subparsers.add_parser("status", help="Show session progress offline.")
    status.add_argument("--session", required=True)

    probe = subparsers.add_parser(
        "probe",
        help=(
            "Preflight and, after a one-time TTY challenge, execute exactly one "
            "current-anchored joint probe."
        ),
    )
    probe.add_argument("--session", required=True)
    probe.add_argument("--joint", required=True)
    probe.add_argument(
        "--row",
        type=int,
        default=None,
        help="Use a reviewed alternate candidate row from `status`.",
    )
    probe.add_argument("--port", default=None)
    probe.add_argument("--baudrate", type=int, default=None)
    probe.add_argument("--slave-id", type=lambda value: int(value, 0), default=None)
    probe.add_argument("--operator", default=getpass.getuser())
    probe.add_argument("--allow-unverified-calibration", action="store_true")
    probe.add_argument("--allow-passive-limit-outlier", action="store_true")
    return parser


def _load_runtime(session: dict) -> tuple[Revo3Profile, ReplayTrace]:
    verify_session_artifacts(session)
    artifacts = session["artifacts"]
    profile = Revo3Profile.load(artifacts["profile"]["path"])
    trace = ReplayTrace.load(
        artifacts["trace"]["path"],
        profile,
        checkpoint_path=artifacts["checkpoint"]["path"],
    )
    if trace.metadata["checkpoint_sha256"] != session["trace_checkpoint_sha256"]:
        raise ValueError("Session trace checkpoint SHA256 provenance changed.")
    expected_device = session.get("device_expected") or {}
    expected_serials = [str(value) for value in profile.sdk.get("serial_allowlist") or ()]
    if len(expected_serials) != 1 or not expected_serials[0].strip():
        raise ValueError(
            "Joint-order session profile must bind exactly one physical hand serial."
        )
    if expected_device.get("hand") != profile.hand or list(
        expected_device.get("serial_allowlist") or ()
    ) != expected_serials:
        raise ValueError("Session expected device identity was modified.")
    validate_session_plan(session, trace, profile)
    return profile, trace


def _print_status(session: dict) -> None:
    print(
        f"session={session['session_id']} state={session['state']} "
        f"updated={session['updated_at']}"
    )
    print("state       Pidx SDK joint                        primary_row delta_deg alternate")
    next_joint: dict | None = None
    for joint in session["joints"]:
        candidates = joint.get("candidates") or []
        if candidates:
            primary = candidates[0]
            row_text = str(primary["row"])
            delta_text = f"{float(primary['delta_deg']):+.3f}"
            alternate = ",".join(
                f"r{item['row']}:{float(item['delta_deg']):+.3f}"
                for item in candidates[1:]
            ) or "-"
        else:
            row_text = "-"
            delta_text = "-"
            alternate = "-"
        print(
            f"{joint['state']:<11} P{joint['policy_index']:02d} "
            f"M{joint['sdk_index']:02d} {joint['joint_name']:<28} "
            f"{row_text:>11} {delta_text:>9} {alternate}"
        )
        if next_joint is None and joint["state"] == "planned":
            next_joint = joint
    if next_joint is not None:
        print(
            "next_explicit_probe="
            f"--joint P{next_joint['policy_index']:02d}"
        )
    elif all(joint["state"] == "passed" for joint in session["joints"]):
        print("ALL JOINTS RECORDED AS MATCH")


def _replay_namespace(
    *,
    session: dict,
    joint: dict,
    candidate: dict,
    args: argparse.Namespace,
    execute: bool,
) -> argparse.Namespace:
    artifacts = session["artifacts"]
    argv = [
        "--trace-npz",
        artifacts["trace"]["path"],
        "--checkpoint",
        artifacts["checkpoint"]["path"],
        "--profile",
        artifacts["profile"]["path"],
        "--start-frame",
        str(candidate["row"]),
        "--frames",
        "1",
        "--joint",
        f"P{joint['policy_index']:02d}",
        "--anchor-current",
        "--kp",
        "0.2",
        "--kd",
        "0.05",
        "--max-speed-deg-s",
        "2",
        "--expected-trace-sha256",
        artifacts["trace"]["sha256"],
        "--expected-checkpoint-sha256",
        artifacts["checkpoint"]["sha256"],
        "--expected-profile-sha256",
        artifacts["profile"]["sha256"],
    ]
    if args.port is not None:
        argv.extend(("--port", args.port))
    if args.baudrate is not None:
        argv.extend(("--baudrate", str(args.baudrate)))
    if args.slave_id is not None:
        argv.extend(("--slave-id", str(args.slave_id)))
    if args.allow_passive_limit_outlier:
        argv.append("--allow-passive-limit-outlier")
    if execute:
        argv.extend(
            (
                "--execute",
                "--confirm-fixed",
                "--confirm-clear-path",
                "--confirm-estop",
                "--confirm-release",
                "--confirm-mapping",
                "--confirm-large-excursion",
                "--confirm-current-anchor",
            )
        )
        if args.allow_unverified_calibration:
            argv.append("--allow-unverified-calibration")
    else:
        argv.append("--preflight")
    return replay_build_parser().parse_args(argv)


def _approval_text(
    session: dict,
    joint: dict,
    candidate: dict,
    nonce: str,
    args: argparse.Namespace,
) -> str:
    overrides: list[str] = []
    if args.allow_unverified_calibration:
        overrides.append("UNVERIFIED_CALIBRATION")
    if args.allow_passive_limit_outlier:
        overrides.append("PASSIVE_LIMIT_OUTLIER")
    override_text = "+".join(overrides) if overrides else "NO_OVERRIDES"
    serial = session["device_expected"]["serial_allowlist"][0]
    return (
        f"ARM SN={serial} P{joint['policy_index']:02d} M{joint['sdk_index']:02d} "
        f"ROW{int(candidate['row']):03d} "
        f"{float(candidate['delta_deg']):+.3f}deg {override_text} {nonce}"
    )


def _new_nonce(*, different_from: str | None = None) -> str:
    for _ in range(16):
        nonce = secrets.token_hex(2).upper()
        if nonce != different_from:
            return nonce
    raise RuntimeError("Could not generate a distinct one-time challenge nonce.")


async def _ask_prompt(prompt_value: TtyPrompter | Prompt, message: str) -> str:
    if isinstance(prompt_value, TtyPrompter):
        return await prompt_value.ask(message)
    return prompt_value(message)


async def _probe(
    args: argparse.Namespace,
    *,
    prompt: Prompt | None = None,
    replay_runner: Callable[[argparse.Namespace], object] = replay_async_main,
) -> int:
    session_path = Path(args.session).expanduser().resolve()
    session = load_session(session_path)
    if session["state"] != "active":
        raise RuntimeError(f"Session is not active: {session['state']}.")
    profile, trace = _load_runtime(session)
    joint = resolve_session_joint(session, args.joint)
    if joint["state"] != "planned":
        raise RuntimeError(
            f"P{joint['policy_index']:02d} is {joint['state']}, not planned; "
            "automatic re-probing is refused."
        )
    candidate = select_candidate(joint, args.row)
    row = int(candidate["row"])
    actual_delta = float(
        np.rad2deg(
            trace.policy_target_rad[row, joint["policy_index"]]
            - trace.target_before_policy_rad[row, joint["policy_index"]]
        )
    )
    if abs(actual_delta - float(candidate["delta_deg"])) > 1.0e-4:
        raise ValueError("Session candidate no longer matches the validated trace.")
    if profile.calibration_status != "verified" and not args.allow_unverified_calibration:
        raise RuntimeError(
            "Profile calibration is unverified; this one-joint probe requires "
            "--allow-unverified-calibration, which will be bound into its TTY challenge."
        )

    prompt_context = nullcontext(prompt)
    if prompt is None:
        prompt_context = TtyPrompter()
    with prompt_context as prompt_value:
        attempt = {
            "attempt_id": str(uuid4()),
            "started_at": utc_now(),
            "operator": str(args.operator),
            "candidate": dict(candidate),
            "preflight": {"status": "started"},
            "execution": {"status": "not_started"},
            "observation": {"verdict": "pending"},
        }
        joint["attempts"].append(attempt)
        save_session(session_path, session)
        try:
            await replay_runner(
                _replay_namespace(
                    session=session,
                    joint=joint,
                    candidate=candidate,
                    args=args,
                    execute=False,
                )
            )
        except BaseException as exc:
            attempt["preflight"] = {
                "status": "failed",
                "ended_at": utc_now(),
                "error": repr(exc),
            }
            save_session(session_path, session)
            raise
        attempt["preflight"] = {"status": "passed", "ended_at": utc_now()}
        approval_nonce = _new_nonce()
        approval = _approval_text(session, joint, candidate, approval_nonce, args)
        print(
            "\nONE-JOINT MOTION GATE\n"
            "Confirm for this joint only: hand fixture secure; motion path clear; "
            "hardware E-stop ready; zero-force release will make the hand soft; "
            "printed P->M mapping, signed displacement, current-anchor semantics, and "
            "listed overrides reviewed.\n"
            f"Type exactly:\n{approval}\n"
        )
        response = await _ask_prompt(prompt_value, "approval> ")
        if response != approval:
            attempt["approval"] = {
                "status": "declined_or_mismatch",
                "at": utc_now(),
            }
            save_session(session_path, session)
            print("Approval mismatch: no motion command was sent.", file=sys.stderr)
            return 2

        attempt["approval"] = {
            "status": "consumed",
            "nonce": approval_nonce,
            "challenge": approval,
            "at": utc_now(),
        }
        # Review can take an arbitrary amount of time. Revalidate the exact
        # bytes behind the displayed P->M mapping before arming; replay also
        # enforces these same hashes around its own artifact load.
        try:
            _load_runtime(session)
        except BaseException as exc:
            attempt["execution"] = {
                "status": "artifact_revalidation_failed",
                "ended_at": utc_now(),
                "error": repr(exc),
            }
            joint["state"] = "blocked"
            session["state"] = "blocked"
            save_session(session_path, session)
            raise
        attempt["execution"] = {"status": "armed", "at": utc_now()}
        joint["state"] = "armed"
        session["state"] = "armed"
        save_session(session_path, session)
        try:
            await replay_runner(
                _replay_namespace(
                    session=session,
                    joint=joint,
                    candidate=candidate,
                    args=args,
                    execute=True,
                )
            )
        except BaseException as exc:
            attempt["execution"] = {
                "status": "failed_or_interrupted",
                "ended_at": utc_now(),
                "error": repr(exc),
            }
            joint["state"] = "blocked"
            session["state"] = "blocked"
            save_session(session_path, session)
            raise

        attempt["execution"] = {
            "status": "passed_released_closed",
            "ended_at": utc_now(),
        }
        joint["state"] = "observation_pending"
        session["state"] = "observation_pending"
        save_session(session_path, session)
        try:
            if isinstance(prompt_value, TtyPrompter):
                prompt_value.discard_pending_input()
            observation_nonce = _new_nonce(different_from=approval_nonce)
        except BaseException as exc:
            attempt["observation"] = {
                "verdict": "uncertain",
                "at": utc_now(),
                "error": repr(exc),
            }
            joint["state"] = "blocked"
            session["state"] = "blocked"
            save_session(session_path, session)
            raise
        options = " | ".join(
            f"{name} {observation_nonce}" for name in VERDICTS
        )
        print(
            "Motion ended and replay cleanup returned. Record the physical observation.\n"
            f"Type exactly one of:\n{options}"
        )
        try:
            verdict_response = await _ask_prompt(prompt_value, "observation> ")
        except (EOFError, KeyboardInterrupt) as exc:
            attempt["observation"] = {
                "verdict": "uncertain",
                "at": utc_now(),
                "error": repr(exc),
            }
            joint["state"] = "blocked"
            session["state"] = "blocked"
            save_session(session_path, session)
            raise
        verdict_name = next(
            (
                name
                for name in VERDICTS
                if verdict_response == f"{name} {observation_nonce}"
            ),
            None,
        )
        verdict = VERDICTS.get(verdict_name or "", "uncertain")
        attempt["observation"] = {
            "verdict": verdict,
            "response_valid": verdict_name is not None,
            "nonce": observation_nonce,
            "at": utc_now(),
        }
        if verdict == "passed":
            joint["state"] = "passed"
            if all(item["state"] == "passed" for item in session["joints"]):
                session["state"] = "complete"
            else:
                session["state"] = "active"
            save_session(session_path, session)
            print(f"Recorded MATCH for P{joint['policy_index']:02d}/M{joint['sdk_index']:02d}.")
            next_joint = next(
                (item for item in session["joints"] if item["state"] == "planned"),
                None,
            )
            if next_joint is not None:
                print(
                    "next_explicit_probe="
                    f"--joint P{next_joint['policy_index']:02d}"
                )
            return 0
        joint["state"] = "blocked"
        session["state"] = "blocked"
        save_session(session_path, session)
        print(
            f"Recorded {verdict}; session stopped before any next joint.",
            file=sys.stderr,
        )
        return 3


async def async_main(
    args: argparse.Namespace,
    *,
    prompt: Prompt | None = None,
    replay_runner: Callable[[argparse.Namespace], object] = replay_async_main,
) -> int:
    if args.command == "init":
        profile = Revo3Profile.load(args.profile)
        trace = ReplayTrace.load(
            args.trace_npz,
            profile,
            checkpoint_path=args.checkpoint,
        )
        session = create_session(
            path=args.session,
            trace_path=args.trace_npz,
            checkpoint_path=args.checkpoint,
            profile_path=args.profile,
            trace=trace,
            profile=profile,
            min_delta_deg=args.min_delta_deg,
            max_delta_deg=args.max_delta_deg,
        )
        _print_status(session)
        return 0
    if args.command == "status":
        session = load_session(args.session)
        _load_runtime(session)
        _print_status(session)
        return 0
    if args.command == "probe":
        with _session_lock(args.session):
            return await _probe(args, prompt=prompt, replay_runner=replay_runner)
    raise AssertionError(f"Unhandled joint-order command: {args.command}")


async def _run_with_signal_cleanup(
    args: argparse.Namespace,
    *,
    runner: Callable[[argparse.Namespace], object] = async_main,
) -> int:
    """Turn terminal/process signals into cancellation and wait for release."""
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(runner(args))
    received_signal: int | None = None
    installed: list[signal.Signals] = []

    def request_cleanup(signum: signal.Signals) -> None:
        nonlocal received_signal
        if received_signal is None:
            received_signal = int(signum)
            print(
                f"Received {signum.name}; cancelling motion and waiting for release/close.",
                file=sys.stderr,
            )
            task.cancel()

    # SIGTSTP (Ctrl-Z) must not suspend a process while an MIT target is held;
    # SIGQUIT (Ctrl-\\) must not bypass replay's cleanup either. Handling SIGINT
    # here also keeps repeated Ctrl-C from interrupting the first cleanup pass.
    for signum in (
        signal.SIGINT,
        signal.SIGQUIT,
        signal.SIGTSTP,
        signal.SIGTERM,
        signal.SIGHUP,
    ):
        try:
            loop.add_signal_handler(signum, request_cleanup, signum)
        except (NotImplementedError, RuntimeError):
            continue
        installed.append(signum)
    try:
        return await task
    except asyncio.CancelledError:
        if received_signal is not None:
            return 128 + received_signal
        raise
    finally:
        for signum in installed:
            loop.remove_signal_handler(signum)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_run_with_signal_cleanup(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
