"""Live hand/object editing for the frozen initial-pose viewer.

Import after AppLauncher has started Isaac Sim. USD edits stay in the session
layer; only an explicit Save writes the selected task's manifest entry.
"""

import json
import math
import os
from pathlib import Path
import tempfile

import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz
from pxr import Gf, Sdf, Usd, UsdGeom

from hora.object_registry import OBJECT_MANIFEST_PATH


JOINT_OUTPUT_ORDER = (
    "right_thumb_CMP_joint", "right_thumb_CMR_joint", "right_thumb_MCP_joint",
    "right_thumb_PIP_joint", "right_thumb_DIP_joint",
    "right_index_MPR_joint", "right_index_MCP_joint", "right_index_PIP_joint", "right_index_DIP_joint",
    "right_middle_MPR_joint", "right_middle_MCP_joint", "right_middle_PIP_joint", "right_middle_DIP_joint",
    "right_ring_MPR_joint", "right_ring_MCP_joint", "right_ring_PIP_joint", "right_ring_DIP_joint",
    "right_little_MPR_joint", "right_little_MCP_joint", "right_little_PIP_joint", "right_little_DIP_joint",
)


def joint_values_dict(names, values) -> dict[str, float]:
    """Group the articulation's interleaved DOFs by finger for the manifest."""
    by_name = {name: round(float(value), 6) for name, value in zip(names, values)}
    missing = set(by_name).difference(JOINT_OUTPUT_ORDER)
    if missing:
        raise ValueError(f"JOINT_OUTPUT_ORDER is missing joints: {sorted(missing)}")
    if not all(math.isfinite(value) for value in by_name.values()):
        raise ValueError("Joint positions must be finite")
    return {name: by_name[name] for name in JOINT_OUTPUT_ORDER if name in by_name}


class InitialPoseEditor:
    def __init__(
        self, env, object_spec, joint_values, *, joint_step=0.01,
        position_step=0.001, rotation_step=1.0, scale_step=0.01,
        manifest_path=OBJECT_MANIFEST_PATH,
    ):
        self.env = env
        self.spec = object_spec
        self.manifest_path = Path(manifest_path)
        self.joint_values = [float(value) for value in joint_values]
        self._initial_joints = list(self.joint_values)
        self._initial_object_pose = env.object.data.root_state_w[:, :7].clone()
        self.position_m = (
            self._initial_object_pose[0, :3] - env.scene.env_origins[0]
        ).cpu().tolist()
        self.quaternion = self._initial_object_pose[0, 3:7].cpu().tolist()
        quat = torch.tensor([self.quaternion], dtype=torch.float64)
        quat /= torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        self.quaternion = quat[0].tolist()
        self.rotation_deg = [math.degrees(float(angle[0])) for angle in euler_xyz_from_quat(quat)]
        self.scale = float(object_spec.scale)
        self._initial_position = list(self.position_m)
        self._initial_quaternion = list(self.quaternion)
        self._initial_rotation = list(self.rotation_deg)
        self._joint_dirty = self._pose_dirty = self._scale_dirty = False
        self._restore_initial_object = False
        self._syncing = False
        self._lower = env.hand_dof_lower_limits[0].cpu().tolist()
        self._upper = env.hand_dof_upper_limits[0].cpu().tolist()

        # PhysX Fabric caches rigid-body root scales during initialization, so
        # changing the root's USD scale alone does not resize the frozen render.
        # Prepend a session-only scale to each geometry branch instead. Placing
        # it BEFORE the authored transforms also scales recentering translations,
        # giving the same result as the manifest's root scale on the next launch.
        self._scale_ops = []
        stage = env.sim.stage
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for path in env.object.root_physx_view.prim_paths:
                for child in stage.GetPrimAtPath(path).GetChildren():
                    xform = UsdGeom.Xformable(child)
                    if not xform:
                        continue
                    op_name = "xformOp:scale:poseEditor"
                    ops = [op for op in xform.GetOrderedXformOps() if op.GetOpName() != op_name]
                    reset_stack = xform.GetResetXformStack()
                    # Cloned environments may already inherit this op from env_0.
                    attr = child.GetAttribute(op_name)
                    op = UsdGeom.XformOp(attr) if attr else xform.AddScaleOp(opSuffix="poseEditor")
                    op.Set(Gf.Vec3d(1.0))
                    xform.SetXformOpOrder([op, *ops], reset_stack)
                    self._scale_ops.append(op)
        if not self._scale_ops:
            raise ValueError("Object has no transformable geometry branches to scale")

        self._build_ui(joint_step, position_step, rotation_step, scale_step)

    def _status(self, message):
        self.status.text = message

    def _accept_value(self, model, previous, *, minimum=None, maximum=None):
        value = float(model.as_float)  # property, not a callable in Isaac Sim 5.1
        if not math.isfinite(value) or (minimum is not None and value < minimum) or (
            maximum is not None and value > maximum
        ):
            self._syncing = True
            try:
                model.set_value(previous)
            finally:
                self._syncing = False
            self._status("Invalid value: enter a finite number within the control's limits.")
            return None
        self._status("Unsaved changes. Click Save to persist hand, object pose and scale.")
        return value

    def _set_joint(self, index, model):
        if self._syncing:
            return
        value = self._accept_value(model, self.joint_values[index], minimum=self._lower[index], maximum=self._upper[index])
        if value is not None:
            self.joint_values[index] = value
            self._joint_dirty = True

    def _set_position(self, index, model):
        if self._syncing:
            return
        value = self._accept_value(model, self.position_m[index] * 1000.0)
        if value is not None:
            self.position_m[index] = value / 1000.0
            self._pose_dirty = True
            self._restore_initial_object = False

    def _set_rotation(self, index, model):
        if self._syncing:
            return
        value = self._accept_value(model, self.rotation_deg[index])
        if value is not None:
            self.rotation_deg[index] = value
            angles = torch.deg2rad(torch.tensor(self.rotation_deg, dtype=torch.float64))
            self.quaternion = quat_from_euler_xyz(*angles).tolist()
            self._pose_dirty = True
            self._restore_initial_object = False
            self._update_object_labels()

    def _set_scale(self, model):
        if self._syncing:
            return
        value = self._accept_value(model, self.scale, minimum=0.0001, maximum=1000.0)
        if value is not None:
            self.scale = value
            self._scale_dirty = True
            self._update_object_labels()

    def _update_object_labels(self):
        size = [value * self.scale * 1000 for value in self.spec.source_size_m]
        self.size_label.text = "Local size (mm): " + " x ".join(f"{value:.1f}" for value in size)
        self.quat_label.text = "Quaternion (wxyz): " + ", ".join(f"{value:+.5f}" for value in self.quaternion)

    def reset_object(self):
        self.position_m = list(self._initial_position)
        self.quaternion = list(self._initial_quaternion)
        self.rotation_deg = list(self._initial_rotation)
        self.scale = float(self.spec.scale)
        self._syncing = True
        try:
            for model, value in zip(self.position_models, self.position_m):
                model.set_value(value * 1000.0)
            for model, value in zip(self.rotation_models, self.rotation_deg):
                model.set_value(value)
            self.scale_model.set_value(self.scale)
        finally:
            self._syncing = False
        self._pose_dirty = self._scale_dirty = True
        self._restore_initial_object = True
        self._update_object_labels()
        self._status("Object reset to its pose and size when this editor opened (not saved).")

    def reset(self):
        self.reset_object()
        self.joint_values = list(self._initial_joints)
        self._syncing = True
        try:
            for model, value in zip(self.joint_models, self.joint_values):
                model.set_value(value)
        finally:
            self._syncing = False
        self._joint_dirty = True
        self._status("Hand and object reset to the values loaded when this editor opened (not saved).")

    def apply(self):
        """Apply pending UI edits before rendering, without advancing physics."""
        if self._scale_dirty:
            stage = self.env.sim.stage
            ratio = self.scale / self.spec.scale
            with Usd.EditContext(stage, stage.GetSessionLayer()), Sdf.ChangeBlock():
                for op in self._scale_ops:
                    op.Set(Gf.Vec3d(ratio))
            self._scale_dirty = False
        if self._pose_dirty:
            if self._restore_initial_object:
                poses = self._initial_object_pose.clone()
            else:
                poses = torch.tensor(
                    self.position_m + self.quaternion,
                    dtype=self._initial_object_pose.dtype, device=self.env.device,
                ).repeat(self.env.num_envs, 1)
                poses[:, :3] += self.env.scene.env_origins
            self.env.object.write_root_pose_to_sim(poses)
            self.env.object.write_root_velocity_to_sim(torch.zeros((self.env.num_envs, 6), device=self.env.device))
            axes = getattr(self.env, "_axes_visualizer", None)
            if axes is not None:
                axes.visualize(translations=poses[:, :3], orientations=poses[:, 3:7])
            self._pose_dirty = self._restore_initial_object = False
        if self._joint_dirty:
            joints = torch.tensor(
                self.joint_values, dtype=self.env.hand.data.joint_pos.dtype, device=self.env.device,
            ).repeat(self.env.num_envs, 1)
            self.env.hand.write_joint_state_to_sim(joints, torch.zeros_like(joints))
            self.env.hand.set_joint_position_target(joints)
            self._joint_dirty = False

    def manifest_values(self):
        return {
            "scale": self.scale,
            "grasp_seed": {
                "hand_pose_profile": "custom",
                "hand_joint_pos_rad": joint_values_dict(self.env.hand.joint_names, self.joint_values),
                "object_pos_m": [round(value, 6) for value in self.position_m],
                "object_quat_wxyz": list(self.quaternion),
            },
        }

    def print_values(self):
        print("\n[POSE EDITOR] Current manifest fields:\n" + json.dumps(self.manifest_values(), indent=2, allow_nan=False), flush=True)

    def save(self):
        """Re-read at click time, preserve other entries, then atomically replace."""
        if self.spec.kind != "usd":
            self._status("Built-in ball/cylinder: preview and Print JSON only; no manifest entry to save.")
            return False
        temporary_path = None
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            values = self.manifest_values()
            item = manifest[self.spec.name]
            item["scale"] = values["scale"]
            item["grasp_seed"].update(values["grasp_seed"])
            payload = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.manifest_path.parent,
                prefix=f".{self.manifest_path.stem}-", suffix=".json.tmp", delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(payload)
            temporary_path.chmod(self.manifest_path.stat().st_mode & 0o777)
            os.replace(temporary_path, self.manifest_path)
        except (KeyError, OSError, TypeError, ValueError, AttributeError) as error:
            self._status(f"Save failed: {error}")
            print(f"[POSE EDITOR] Save failed: {error}", flush=True)
            return False
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        self._status(f"Saved {self.spec.name}: hand joints, object position, rotation and scale.")
        print(f"[POSE EDITOR] Saved {self.spec.name} -> {self.manifest_path}", flush=True)
        return True

    def _build_ui(self, joint_step, position_step, rotation_step, scale_step):
        import omni.ui as ui

        self.joint_models = [None] * len(self.joint_values)
        self.position_models = []
        self.rotation_models = []
        self.window = ui.Window("Revo3 initial hand / object pose", width=720, height=900)
        with self.window.frame, ui.VStack(spacing=5):
            ui.Label(f"Task: {self.spec.name}    |    frozen preview    |    all {self.env.num_envs} envs", height=24)
            with ui.HStack(height=32, spacing=5):
                ui.Button("Reset all", clicked_fn=self.reset)
                ui.Button("Print JSON", clicked_fn=self.print_values)
                ui.Button("Save to manifest.json", clicked_fn=self.save)
            self.status = ui.Label("Not saved. Closing Isaac Sim prints the final JSON.", height=32, word_wrap=True)
            with ui.ScrollingFrame(), ui.VStack(spacing=5):
                with ui.CollapsableFrame("Object pose and size", collapsed=False), ui.VStack(spacing=4):
                    ui.Label("Position: mm in the environment frame (not relative to the hand).", height=22)
                    for index, axis in enumerate("XYZ"):
                        model = ui.SimpleFloatModel(self.position_m[index] * 1000.0)
                        self.position_models.append(model)
                        model.add_value_changed_fn(lambda m, i=index: self._set_position(i, m))
                        with ui.HStack(height=26, spacing=5):
                            ui.Label(f"Position {axis}", width=150)
                            ui.FloatDrag(model=model, min=-1e6, max=1e6, step=position_step * 1000, format="%.3f")
                            ui.Label("mm", width=50)
                    ui.Label("Rotation: degrees, fixed XYZ axes; R = Rz(yaw) Ry(pitch) Rx(roll).", height=22)
                    for index, name in enumerate(("Roll X", "Pitch Y", "Yaw Z")):
                        model = ui.SimpleFloatModel(self.rotation_deg[index])
                        self.rotation_models.append(model)
                        model.add_value_changed_fn(lambda m, i=index: self._set_rotation(i, m))
                        with ui.HStack(height=26, spacing=5):
                            ui.Label(name, width=150)
                            ui.FloatSlider(model=model, min=-180, max=180, step=rotation_step)
                            ui.FloatDrag(model=model, min=-180, max=180, step=rotation_step, format="%.2f", width=90)
                            ui.Label("deg", width=50)
                    self.scale_model = ui.SimpleFloatModel(self.scale)
                    self.scale_model.add_value_changed_fn(self._set_scale)
                    with ui.HStack(height=30, spacing=5):
                        ui.Label("Uniform scale", width=150)
                        ui.FloatDrag(model=self.scale_model, min=0.0001, max=1000, step=scale_step, format="%.4f")
                        ui.Button("Smaller / 1.1", width=105, clicked_fn=lambda: self.scale_model.set_value(self.scale / 1.1))
                        ui.Button("Larger x 1.1", width=105, clicked_fn=lambda: self.scale_model.set_value(self.scale * 1.1))
                    self.size_label = ui.Label("", height=22)
                    self.quat_label = ui.Label("", height=22)
                    self._update_object_labels()
                    ui.Button("Reset object pose and size", height=28, clicked_fn=self.reset_object)
                for finger in ("thumb", "index", "middle", "ring", "little"):
                    with ui.CollapsableFrame(f"{finger.capitalize()} joints (rad)", collapsed=False), ui.VStack(spacing=3):
                        for index, name in enumerate(self.env.hand.joint_names):
                            if f"right_{finger}_" not in name:
                                continue
                            model = ui.SimpleFloatModel(self.joint_values[index])
                            self.joint_models[index] = model
                            model.add_value_changed_fn(lambda m, i=index: self._set_joint(i, m))
                            lower, upper = self._lower[index], self._upper[index]
                            with ui.HStack(height=26, spacing=5):
                                ui.Label(name.removeprefix("right_").removesuffix("_joint"), width=150)
                                ui.FloatSlider(model=model, min=lower, max=upper, step=joint_step)
                                ui.FloatDrag(model=model, min=lower, max=upper, step=joint_step, width=90)
                                ui.Label(f"[{lower:+.2f}, {upper:+.2f}]", width=105)
