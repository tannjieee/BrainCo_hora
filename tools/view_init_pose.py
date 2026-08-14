"""Visualize and interactively edit the initial hand/object pose.

Modes:
  --physics off (default): render only, freeze sim. Check visual pose.
  --physics on: step zero actions, print obj_z/hand_z every 20 steps. Test passive stability.
  --edit_joints: show a live editor for all hand joints in frozen mode.

Task selection uses the same object registry as grasp collection and training.

Gotcha — joint pose override: after env.reset(), writes cfg.init_state.joint_pos directly
  to sim via write_joint_state_to_sim. This is needed because USD may have baked-in default
  joint positions that differ from assets.py. The env's init_joint_pos is built from the
  same source, but the manual override ensures the render matches exactly.
"""

import argparse
import asyncio
import copy
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher
from hora.object_registry import OBJECT_MANIFEST_PATH, OBJECT_TASK_NAMES

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="ball",
    choices=OBJECT_TASK_NAMES,
    help="Object task",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--cache", action="store_true", help="Load grasp cache by task instead of assets.py init pose.")
parser.add_argument("--cache_file", type=str, default="", help="Override cache filename under cache/; implies --cache.")
parser.add_argument(
    "--sequential_cache",
    action="store_true",
    help="Map environment i to cache row i (modulo cache size) for deterministic batch validation.",
)
parser.add_argument("--usd", type=str, default="", help="Override hand USD path.")
parser.add_argument(
    "--physics",
    action="store_true",
    help="Step physics with zero actions, useful for checking if assets.py init pose is stable.",
)
parser.add_argument(
    "--edit_joints",
    action="store_true",
    help="Open a live 21-joint editor. Available in the default frozen mode.",
)
parser.add_argument(
    "--joint_step",
    type=float,
    default=0.01,
    help="Fine adjustment step for the joint editor, in radians (default: 0.01).",
)
parser.add_argument("--steps", type=int, default=0, help="Stop after this many physics steps; 0 runs until closed.")
parser.add_argument("--gravity", type=float, default=None, help="Override downward gravity magnitude in m/s².")
parser.add_argument("--settle_steps", type=int, default=20, help="Steps excluded from stable-phase tilt reporting.")
parser.add_argument(
    "--hide_axes",
    action="store_true",
    help="Hide the RGB object-local coordinate frame shown during inspection.",
)
parser.add_argument(
    "--screenshot",
    type=Path,
    default=None,
    help="Capture the initialized viewport to this PNG and exit.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.steps < 0:
    parser.error("--steps must be greater than or equal to 0")
if args.gravity is not None and args.gravity < 0:
    parser.error("--gravity must be greater than or equal to 0")
if args.settle_steps < 0:
    parser.error("--settle_steps must be greater than or equal to 0")
if args.joint_step <= 0:
    parser.error("--joint_step must be greater than zero")
if args.edit_joints and args.physics:
    parser.error("--edit_joints is for frozen pose editing and cannot be combined with --physics")
if args.edit_joints and args.headless:
    parser.error("--edit_joints requires a GUI; remove --headless")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.utils.math import quat_apply
from pxr import Usd, UsdGeom

from hora.tasks.isaaclab import Revo3HandHoraEnv, Revo3HandHoraEnvCfg
from hora.tasks.isaaclab.assets import configure_env_for_object_task

env_cfg = Revo3HandHoraEnvCfg()

object_spec = configure_env_for_object_task(env_cfg, args.task)
# Start close enough to inspect fingertip/object intersections without having
# to navigate from Isaac Sim's generic scene-wide camera pose.
env_cfg.viewer.eye = (0.45, -0.55, 1.82)
env_cfg.viewer.lookat = (0.0, -0.06, 1.60)
use_cache = args.cache or bool(args.cache_file)
if args.sequential_cache and not use_cache:
    parser.error("--sequential_cache requires --cache or --cache_file")
if use_cache:
    if args.cache_file:
        cache_path = f"cache/{args.cache_file.removesuffix('.npy')}"
    else:
        cache_path = object_spec.cache_stem
    env_cfg.grasp_cache_path = cache_path
else:
    cache_path = "none"
    env_cfg.grasp_cache_path = "__nonexistent__"  # force fallback to init_joint_pos

if args.usd:
    usd_path = os.path.abspath(args.usd)
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"--usd path not found: {usd_path}")
    env_cfg.robot_cfg = copy.deepcopy(env_cfg.robot_cfg)
    if env_cfg.robot_cfg.spawn is None or not hasattr(env_cfg.robot_cfg.spawn, "usd_path"):
        raise RuntimeError("env_cfg.robot_cfg.spawn has no usd_path to override.")
    env_cfg.robot_cfg.spawn.usd_path = usd_path

env_cfg.scene.num_envs = args.num_envs
env_cfg.grasp_cache_sequential = args.sequential_cache
# Keep interactive inspection from timing out and changing to a different
# cached row. Automated checks use --steps to provide their own finite horizon.
env_cfg.episode_length_s = 9999.0
env_cfg.randomize_mass = False
env_cfg.randomize_com = False
env_cfg.randomize_friction = False
env_cfg.randomize_pd_gains = False
env_cfg.gravity_curriculum = False
env_cfg.force_scale = 0.0
env_cfg.random_force_prob_scalar = 0.0
env_cfg.debug_show_axes = not args.hide_axes
if args.gravity is not None:
    env_cfg.sim.gravity = (0.0, 0.0, -args.gravity)

print(f"[VIEW] Task: {args.task}")
print(f"[VIEW] Object: {object_spec.display_name} (scale={object_spec.scale:g})")
print(f"[VIEW] Hand seed: {object_spec.hand_pose} + {len(object_spec.hand_joint_pos_rad)} joint overrides")
print(f"[VIEW] Object position: {object_spec.object_init_pos_m} m")
print(f"[VIEW] Object quaternion (wxyz): {object_spec.object_init_quat_wxyz}")
print(f"[VIEW] Object local rotation axis: {object_spec.rotation_axis_local}")
print(f"[VIEW] Target world rotation axis: {object_spec.target_axis_world}")
print(
    f"[VIEW] Axis alignment: {object_spec.enforce_axis_alignment}, "
    f"bidirectional={object_spec.axis_bidirectional}, "
    f"tolerance={object_spec.axis_tilt_tolerance_deg:g} deg"
)
print(f"[VIEW] Cache: {cache_path if use_cache else 'none (assets.py init pose)'}")
print(f"[VIEW] Gravity: {abs(env_cfg.sim.gravity[2]):g} m/s² downward")
if args.usd:
    print(f"[VIEW] Hand USD override: {os.path.abspath(args.usd)}")

print("[VIEW] Creating environment...", flush=True)
env = Revo3HandHoraEnv(env_cfg, render_mode=None if args.headless else "human")
print("[VIEW] Environment created; resetting...", flush=True)
env.reset()
print("[VIEW] Reset complete.", flush=True)

# Verify the composed Stage, not merely the registry metadata. UsdFileCfg.scale
# authors this root scale when spawning the manifest-selected asset.
object_prim = env.sim.stage.GetPrimAtPath("/World/envs/env_0/object")
object_scale_ops = [
    op.Get()
    for op in UsdGeom.Xformable(object_prim).GetOrderedXformOps()
    if op.GetOpType() == UsdGeom.XformOp.TypeScale
]
bbox_range = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
).ComputeWorldBound(object_prim).ComputeAlignedRange()
bbox_size = bbox_range.GetSize()
print(f"[VIEW] Stage root scale op(s): {object_scale_ops}", flush=True)
print(
    "[VIEW] Stage world AABB size: "
    + " x ".join(f"{float(value) * 1000.0:.1f}" for value in bbox_size)
    + " mm",
    flush=True,
)

# Override USD baked-in defaults with assets.py init_state.joint_pos
_init_joint_pos = env_cfg.robot_cfg.init_state.joint_pos
if _init_joint_pos and not use_cache:
    dof_pos = torch.zeros((env.num_envs, env.num_hand_dofs), device=env.device)
    for joint_name, joint_val in _init_joint_pos.items():
        if joint_name in env.hand.joint_names:
            idx = env.hand.joint_names.index(joint_name)
            dof_pos[:, idx] = float(joint_val)
    env.hand.write_joint_state_to_sim(dof_pos, torch.zeros_like(dof_pos))
    env.hand.set_joint_position_target(dof_pos)

# Print actual joint positions to verify assets.py changes took effect.
print("[VIEW] Reading joint state...", flush=True)
joint_names = list(env.hand.joint_names)
print(f"[VIEW] Found {len(joint_names)} joints.", flush=True)
joint_pos = env.hand.data.joint_pos[0].detach().cpu().numpy()
print("[VIEW] Joint state copied to CPU.", flush=True)
print(
    "[VIEW] Actual joint positions after reset (rad): "
    + ", ".join(
        f"{name}={float(pos):+.4f}" for name, pos in zip(joint_names, joint_pos)
    ),
    flush=True,
)


joint_editor = None
editor_values = None
editor_initial_values = None
editor_dirty = False

# Human-readable order used for terminal output and manifest serialization.
# The articulation's internal DOF order is interleaved by joint level, which is
# efficient for simulation but awkward when hand-editing a grasp pose.
JOINT_OUTPUT_ORDER = (
    "right_thumb_CMP_joint",
    "right_thumb_CMR_joint",
    "right_thumb_MCP_joint",
    "right_thumb_PIP_joint",
    "right_thumb_DIP_joint",
    "right_index_MPR_joint",
    "right_index_MCP_joint",
    "right_index_PIP_joint",
    "right_index_DIP_joint",
    "right_middle_MPR_joint",
    "right_middle_MCP_joint",
    "right_middle_PIP_joint",
    "right_middle_DIP_joint",
    "right_ring_MPR_joint",
    "right_ring_MCP_joint",
    "right_ring_PIP_joint",
    "right_ring_DIP_joint",
    "right_little_MPR_joint",
    "right_little_MCP_joint",
    "right_little_PIP_joint",
    "right_little_DIP_joint",
)


def _joint_values_dict(values) -> dict[str, float]:
    """Return a manifest mapping grouped by finger without changing values."""
    values_by_name = {
        name: round(float(value), 6) for name, value in zip(joint_names, values)
    }
    missing = set(values_by_name).difference(JOINT_OUTPUT_ORDER)
    if missing:
        raise RuntimeError(f"JOINT_OUTPUT_ORDER is missing joints: {sorted(missing)}")
    return {name: values_by_name[name] for name in JOINT_OUTPUT_ORDER}


def _print_edited_joint_values() -> None:
    print(
        "\n[JOINT EDITOR] Current manifest hand_joint_pos_rad:\n"
        + json.dumps(_joint_values_dict(editor_values), indent=2),
        flush=True,
    )


def _save_edited_joint_values() -> None:
    """Update only this task's joint seed, re-reading the manifest at click time."""
    if object_spec.kind != "usd":
        message = "Built-in ball/cylinder tasks have no manifest entry to save."
        print(f"[JOINT EDITOR] {message}", flush=True)
        if joint_editor is not None:
            joint_editor["status"].text = message
        return

    try:
        manifest = json.loads(OBJECT_MANIFEST_PATH.read_text(encoding="utf-8"))
        grasp_seed = manifest[args.task]["grasp_seed"]
        grasp_seed["hand_pose_profile"] = "custom"
        grasp_seed["hand_joint_pos_rad"] = _joint_values_dict(editor_values)
        temporary_path = OBJECT_MANIFEST_PATH.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, OBJECT_MANIFEST_PATH)
    except (KeyError, OSError, TypeError, ValueError) as error:
        message = f"Save failed: {error}"
        print(f"[JOINT EDITOR] {message}", flush=True)
        if joint_editor is not None:
            joint_editor["status"].text = message
        return


    message = f"Saved {args.task}.grasp_seed.hand_joint_pos_rad"
    print(f"[JOINT EDITOR] {message} -> {OBJECT_MANIFEST_PATH}", flush=True)
    if joint_editor is not None:
        joint_editor["status"].text = message


def _build_joint_editor():
    """Build an omni.ui panel whose models feed the frozen render loop."""
    import omni.ui as ui

    global editor_values, editor_initial_values, editor_dirty
    editor_values = [float(value) for value in joint_pos]
    editor_initial_values = list(editor_values)
    editor_dirty = False

    lower_limits = env.hand_dof_lower_limits[0].detach().cpu().tolist()
    upper_limits = env.hand_dof_upper_limits[0].detach().cpu().tolist()
    models = [None] * len(joint_names)

    def set_joint_value(index: int, model) -> None:
        global editor_dirty
        # omni.ui.AbstractValueModel exposes ``as_float`` as a property in
        # Isaac Sim 5.1 (calling it raises: TypeError: 'float' object is not callable).
        editor_values[index] = float(model.as_float)
        editor_dirty = True

    def reset_values() -> None:
        global editor_dirty
        for model, value in zip(models, editor_initial_values):
            model.set_value(value)
        editor_dirty = True
        editor["status"].text = "Reset to the pose loaded when the editor opened."

    window = ui.Window("Revo3 initial joint pose", width=660, height=860)
    editor = {"window": window, "models": models, "status": None}
    with window.frame:
        with ui.VStack(spacing=5):
            ui.Label(
                f"Task: {args.task}    |    values: rad    |    all {env.num_envs} envs",
                height=24,
            )
            ui.Label(
                "Drag a slider or type a value. The frozen hand updates immediately.",
                height=22,
            )
            with ui.HStack(height=32, spacing=5):
                ui.Button("Reset", clicked_fn=reset_values)
                ui.Button("Print JSON", clicked_fn=_print_edited_joint_values)
                ui.Button("Save to manifest.json", clicked_fn=_save_edited_joint_values)
            status = ui.Label("Not saved. Closing Isaac Sim prints the final JSON.", height=24)
            editor["status"] = status
            with ui.ScrollingFrame():
                with ui.VStack(spacing=3):
                    for finger in ("thumb", "index", "middle", "ring", "little"):
                        ui.Separator(height=5)
                        ui.Label(finger.capitalize(), height=22)
                        for index, name in enumerate(joint_names):
                            if f"right_{finger}_" not in name:
                                continue

                            lower = float(lower_limits[index])
                            upper = float(upper_limits[index])
                            model = ui.SimpleFloatModel(editor_values[index])
                            models[index] = model
                            model.add_value_changed_fn(
                                lambda changed_model, joint_index=index: set_joint_value(
                                    joint_index, changed_model
                                )
                            )
                            short_name = name.removeprefix("right_").removesuffix("_joint")
                            with ui.HStack(height=26, spacing=5):
                                ui.Label(short_name, width=150)
                                ui.FloatSlider(
                                    model=model,
                                    min=lower,
                                    max=upper,
                                    step=args.joint_step,
                                )
                                ui.FloatDrag(
                                    model=model,
                                    min=lower,
                                    max=upper,
                                    step=args.joint_step,
                                    width=90,
                                )
                                ui.Label(f"[{lower:+.2f}, {upper:+.2f}]", width=105)
    return editor


if args.edit_joints:
    joint_editor = _build_joint_editor()
    print(
        "[JOINT EDITOR] Opened. Adjust values in the panel; "
        "use 'Save to manifest.json' to persist this task only.",
        flush=True,
    )
print(
    "[VIEW] Manifest hand_joint_pos_rad template:\n"
    + json.dumps(_joint_values_dict(joint_pos), indent=2),
    flush=True,
)

if args.screenshot is not None:
    import omni.kit.renderer_capture
    import omni.kit.viewport.utility as viewport_utils

    screenshot_path = args.screenshot.expanduser().resolve()
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)

    async def capture_viewport() -> bool:
        viewport = viewport_utils.get_active_viewport()
        if viewport is None:
            raise RuntimeError("No active viewport is available for screenshot capture")
        await viewport_utils.next_viewport_frame_async(viewport)
        capture = viewport_utils.capture_viewport_to_file(
            viewport, file_path=str(screenshot_path)
        )
        return bool(await capture.wait_for_result(completion_frames=30))

    capture_task = asyncio.ensure_future(capture_viewport())
    while simulation_app.is_running() and not capture_task.done():
        simulation_app.update()
    if not capture_task.done() or not capture_task.result():
        raise RuntimeError(f"Failed to capture viewport to {screenshot_path}")
    omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
    print(f"[VIEW] Screenshot saved: {screenshot_path}", flush=True)
    env.close()
    simulation_app.close()
    raise SystemExit(0)

zero_actions = torch.zeros((env.num_envs, env_cfg.action_space), device=env.device)
initial_obj_pos = env.object.data.root_pos_w.clone()
initial_obj_z = initial_obj_pos[:, 2]


def object_axis_tilt_deg() -> torch.Tensor:
    """Angle between the configured object-local and target-world axes."""
    quat = env.object.data.root_quat_w
    axis_world = quat_apply(quat, env.object_rotation_axis_local)
    alignment = (axis_world * env.target_rotation_axis_world).sum(-1)
    if env_cfg.object_axis_bidirectional:
        alignment = torch.abs(alignment)
        alignment = torch.clamp(alignment, 0.0, 1.0)
    else:
        alignment = torch.clamp(alignment, -1.0, 1.0)
    return torch.rad2deg(torch.acos(alignment))

if not args.physics:
    print("\n[VIEW] Frozen render mode.")
    print("  Showing assets.py hand init pose + assets.py object init pos.")
    if args.edit_joints:
        print("  Joint editor is live; adjustments are applied to every displayed environment.")
    print("  Add --physics to step zero actions and test passive stability.\n")
    env.sim._physics_context.enabled = False  # freeze physics, render only
    while simulation_app.is_running():
        if editor_dirty:
            edited_pos = torch.tensor(
                editor_values,
                dtype=env.hand.data.joint_pos.dtype,
                device=env.device,
            ).unsqueeze(0).repeat(env.num_envs, 1)
            env.hand.write_joint_state_to_sim(edited_pos, torch.zeros_like(edited_pos))
            env.hand.set_joint_position_target(edited_pos)
            editor_dirty = False
        env.sim.render()
else:
    print("\n[PHYSICS] Stepping with zero actions.", flush=True)
    print("  Testing whether the selected initial pose can hold the object without policy action.", flush=True)
    print("  obj_z printed every 20 steps. Hand z printed for reference.\n", flush=True)
    step = 0
    termination_count = 0
    timeout_count = 0
    max_abs_z_drift = 0.0
    max_horizontal_drift = 0.0
    max_stable_horizontal_drift = 0.0
    max_axis_tilt_deg = 0.0
    max_stable_axis_tilt_deg = 0.0
    while simulation_app.is_running():
        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step(zero_actions)
        step += 1
        termination_count += int(terminated.sum().item())
        timeout_count += int(truncated.sum().item())
        obj_z = env.object.data.root_pos_w[:, 2]
        max_abs_z_drift = max(
            max_abs_z_drift, float(torch.max(torch.abs(obj_z - initial_obj_z)).item())
        )
        horizontal_drift = torch.norm(
            env.object.data.root_pos_w[:, :2] - initial_obj_pos[:, :2], dim=-1
        )
        max_horizontal_drift = max(
            max_horizontal_drift, float(horizontal_drift.max().item())
        )
        axis_tilt_deg = object_axis_tilt_deg()
        max_axis_tilt_deg = max(max_axis_tilt_deg, float(axis_tilt_deg.max().item()))
        if step > args.settle_steps:
            max_stable_horizontal_drift = max(
                max_stable_horizontal_drift, float(horizontal_drift.max().item())
            )
            max_stable_axis_tilt_deg = max(
                max_stable_axis_tilt_deg, float(axis_tilt_deg.max().item())
            )
        if step % 20 == 0:
            hand_z = env.hand.data.root_pos_w[:, 2]
            print(
                f"  step={step:4d}  "
                f"obj_z={obj_z[0]:.4f}  "
                f"obj_z_range=[{obj_z.min():.4f}, {obj_z.max():.4f}]  "
                f"xy_drift_max={1000.0 * horizontal_drift.max():.2f}mm  "
                f"tilt_range=[{axis_tilt_deg.min():.2f}, {axis_tilt_deg.max():.2f}]deg  "
                f"hand_z={hand_z[0]:.4f}  "
                f"diff={obj_z[0] - hand_z[0]:.4f}"
            )
        if args.steps and step >= args.steps:
            print(
                f"\n[RESULT] steps={step} terminations={termination_count} timeouts={timeout_count} "
                f"max_abs_z_drift={max_abs_z_drift:.6f}m "
                f"max_xy_drift={1000.0 * max_horizontal_drift:.2f}mm "
                f"max_stable_xy_drift={1000.0 * max_stable_horizontal_drift:.2f}mm "
                f"max_axis_tilt={max_axis_tilt_deg:.2f}deg "
                f"max_stable_axis_tilt={max_stable_axis_tilt_deg:.2f}deg",
                flush=True,
            )
            break

if args.edit_joints:
    _print_edited_joint_values()

env.close()
simulation_app.close()
