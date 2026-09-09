"""Isaac Sim integration check; never saves into the real object manifest.

Run: ~/IsaacLab/isaaclab.sh -p tests/test_initial_pose_editor.py
"""

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--screenshot", type=Path, help="Optional full Isaac Sim window capture for UI inspection.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from pxr import Usd, UsdGeom

from hora import object_registry
from hora.tasks.isaaclab import Revo3HandHoraEnv, Revo3HandHoraEnvCfg
from hora.tasks.isaaclab.assets import configure_env_for_object_task, get_object_cfg
from tools.initial_pose_editor import InitialPoseEditor


class InitialPoseEditorTest(unittest.TestCase):
    def test_live_edits_reset_and_manifest_round_trip(self):
        cfg = Revo3HandHoraEnvCfg()
        spec = configure_env_for_object_task(cfg, "strawberry")
        cfg.scene.num_envs = 2
        cfg.grasp_cache_path = "__nonexistent__"
        cfg.debug_show_axes = True
        cfg.viewer.eye = (0.45, -0.55, 1.82)
        cfg.viewer.lookat = (0.0, -0.06, 1.60)
        env = Revo3HandHoraEnv(cfg)
        editor = None
        try:
            env.reset()
            env.sim._physics_context.enabled = False
            initial_pose = env.object.data.root_state_w[:, :7].clone()
            # Mimic different cache rows, so reset must restore each env's pose.
            initial_pose[1, 0] += 0.015
            env.object.write_root_pose_to_sim(initial_pose)
            initial_joints = env.hand.data.joint_pos[0].cpu().tolist()
            original_manifest = object_registry.OBJECT_MANIFEST_PATH.read_bytes()
            with tempfile.TemporaryDirectory(prefix="pose-editor-test-") as directory:
                manifest_path = Path(directory) / "manifest.json"
                manifest_path.write_bytes(original_manifest)
                editor = InitialPoseEditor(env, spec, initial_joints, manifest_path=manifest_path)
                self.assertEqual(len(editor.joint_models), 21)
                for _ in range(3):
                    env.sim.render()

                if args.screenshot is not None:
                    import omni.kit.renderer_capture

                    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                    capture = omni.kit.renderer_capture.acquire_renderer_capture_interface()
                    capture.capture_next_frame_swapchain(str(args.screenshot))
                    for _ in range(5):
                        env.sim.render()
                    capture.wait_async_capture()

                import omni.usd
                import usdrt

                fabric_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())

                def rendered_scales():
                    result = {}
                    for path in env.object.root_physx_view.prim_paths:
                        root = env.sim.stage.GetPrimAtPath(path)
                        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                            if not prim.IsA(UsdGeom.Mesh):
                                continue
                            rt_prim = fabric_stage.GetPrimAtPath(str(prim.GetPath()))
                            attr = usdrt.Rt.Xformable(rt_prim).GetFabricHierarchyWorldMatrixAttr()
                            if not attr.IsValid():
                                continue
                            matrix = attr.Get()
                            result[str(prim.GetPath())] = [
                                math.sqrt(sum(matrix[row][col] ** 2 for col in range(3)))
                                for row in range(3)
                            ]
                    self.assertGreaterEqual(len(result), env.num_envs)
                    return result

                # Measure actual USD geometry, not just the scale UI value.
                def bounds():
                    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
                    return [
                        cache.ComputeWorldBound(env.sim.stage.GetPrimAtPath(path)).ComputeAlignedRange().GetSize()
                        for path in env.object.root_physx_view.prim_paths
                    ]

                before = bounds()
                rendered_before = rendered_scales()
                editor.scale_model.set_value(spec.scale * 1.5)
                editor.apply()
                for _ in range(3):
                    env.sim.render()
                after = bounds()
                for old_size, new_size in zip(before, after):
                    for old, new in zip(old_size, new_size):
                        self.assertAlmostEqual(new / old, 1.5, places=5)

                # Check the actual rendered meshes, including instance proxies.
                for path, new_scale in rendered_scales().items():
                    for old, new in zip(rendered_before[path], new_scale):
                        self.assertAlmostEqual(new / old, 1.5, places=5)
                print("[TEST] Both USD bounds and rendered mesh scales increased by 1.5x.", flush=True)

                target_pos = (0.02, -0.06, 1.67)
                for model, value in zip(editor.position_models, target_pos):
                    model.set_value(value * 1000)
                editor.rotation_models[2].set_value(90.0)
                joint_index = 0
                target_joint = float((env.hand_dof_lower_limits[0, joint_index] + env.hand_dof_upper_limits[0, joint_index]) / 2)
                editor.joint_models[joint_index].set_value(target_joint)
                editor.apply()
                for _ in range(3):
                    env.sim.render()
                local_pos = env.object.data.root_pos_w - env.scene.env_origins
                torch.testing.assert_close(local_pos, torch.tensor(target_pos, device=env.device).repeat(2, 1), atol=1e-6, rtol=0)
                quat = torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], device=env.device).repeat(2, 1)
                torch.testing.assert_close(env.object.data.root_quat_w, quat, atol=1e-6, rtol=0)
                for index, path in enumerate(env.object.root_physx_view.prim_paths):
                    rt_prim = fabric_stage.GetPrimAtPath(path)
                    matrix = usdrt.Rt.Xformable(rt_prim).GetFabricHierarchyWorldMatrixAttr().Get()
                    for actual, expected in zip(matrix.ExtractTranslation(), env.object.data.root_pos_w[index].cpu().tolist()):
                        self.assertAlmostEqual(actual, expected, places=5)
                for path, new_scale in rendered_scales().items():
                    for old, new in zip(rendered_before[path], new_scale):
                        self.assertAlmostEqual(new / old, 1.5, places=5)
                self.assertAlmostEqual(float(env.hand.data.joint_pos[0, joint_index]), target_joint, places=5)
                axis_prim = env.sim.stage.GetPrimAtPath("/Visuals/ObjectAxes")
                axis_positions = axis_prim.GetAttribute("positions").Get()
                for actual, expected in zip(axis_positions, env.object.data.root_pos_w.cpu().tolist()):
                    for a, b in zip(actual, expected):
                        self.assertAlmostEqual(a, b, places=5)

                # Invalid typed values must neither reach physics nor be saved.
                previous = editor.scale
                editor.scale_model.set_value(-1)
                self.assertEqual(editor.scale, previous)
                editor.position_models[0].set_value(float("nan"))
                self.assertAlmostEqual(editor.position_m[0], target_pos[0], places=6)
                json.dumps(editor.manifest_values(), allow_nan=False)

                # Include external changes made after the panel opened.
                external = json.loads(manifest_path.read_text())
                external["strawberry"]["editor_test_note"] = "keep me"
                manifest_path.write_text(json.dumps(external))
                self.assertTrue(editor.save())
                saved = json.loads(manifest_path.read_text())
                self.assertEqual(saved["strawberry"]["editor_test_note"], "keep me")
                for name in external:
                    if name != "strawberry":
                        self.assertEqual(saved[name], external[name])
                for name, value in external["strawberry"].items():
                    if name not in {"scale", "grasp_seed"}:
                        self.assertEqual(saved["strawberry"][name], value)

                with patch.object(object_registry, "OBJECT_MANIFEST_PATH", manifest_path):
                    reloaded = object_registry._load_scanned_objects()["strawberry"]
                self.assertAlmostEqual(reloaded.scale, 1.5 * spec.scale, places=5)
                for actual, expected in zip(reloaded.object_init_pos_m, target_pos):
                    self.assertAlmostEqual(actual, expected, places=6)
                with patch.dict(object_registry.OBJECT_TASK_SPECS, strawberry=reloaded):
                    spawn_cfg = get_object_cfg("strawberry")
                self.assertEqual(spawn_cfg.spawn.scale, (reloaded.scale,) * 3)
                self.assertEqual(spawn_cfg.init_state.pos, reloaded.object_init_pos_m)
                self.assertEqual(spawn_cfg.init_state.rot, reloaded.object_init_quat_wxyz)

                # Reset restores object scale and each loaded pose, not defaults.
                editor.reset()
                editor.apply()
                for _ in range(3):
                    env.sim.render()
                torch.testing.assert_close(env.object.data.root_state_w[:, :7], initial_pose)
                self.assertEqual(editor.joint_values, initial_joints)
                for old_size, reset_size in zip(before, bounds()):
                    for old, new in zip(old_size, reset_size):
                        self.assertAlmostEqual(old, new, places=6)
                for path, reset_scale in rendered_scales().items():
                    for old, new in zip(rendered_before[path], reset_scale):
                        self.assertAlmostEqual(old, new, places=6)
                self.assertEqual(object_registry.OBJECT_MANIFEST_PATH.read_bytes(), original_manifest)
                print("[TEST] Passed: UI callbacks, live scale, pose, axes, joints, reset, save and reload.", flush=True)
        finally:
            if editor is not None:
                editor.window.destroy()
            env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        result = unittest.main(argv=[sys.argv[0]], exit=False)
        exit_code = int(not result.result.wasSuccessful())
    finally:
        # Kit's fast shutdown exits the process, so set its code before closing.
        import omni.kit.app

        omni.kit.app.get_app().post_quit(exit_code)
        app.close()
    raise SystemExit(exit_code)
