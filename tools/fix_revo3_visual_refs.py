"""Remove stale visual references emitted for visual-less Revo3 links."""

from pathlib import Path

from isaacsim import SimulationApp


LINKS_WITHOUT_VISUALS = (
    "world",
    "right_palm",
    "right_thumb_tip_Link",
    "right_index_tip_Link",
    "right_middle_tip_Link",
    "right_ring_tip_Link",
    "right_little_tip_Link",
)


def main() -> None:
    simulation_app = SimulationApp({"headless": True})

    try:
        from pxr import Sdf

        usd_path = (
            Path(__file__).resolve().parents[1]
            / "assets/usd/configuration/revo3_right_base.usd"
        )
        layer = Sdf.Layer.FindOrOpen(str(usd_path))
        if layer is None:
            raise RuntimeError(f"Could not open USD layer: {usd_path}")

        removed_paths: list[str] = []
        for link_name in LINKS_WITHOUT_VISUALS:
            visual_target = f"/visuals/{link_name}"
            if layer.GetPrimAtPath(visual_target) is not None:
                raise RuntimeError(
                    f"Refusing to remove reference because its target exists: {visual_target}"
                )

            prim_path = f"/revo3_single/{link_name}/visuals"
            prim_spec = layer.GetPrimAtPath(prim_path)
            if prim_spec is None:
                raise RuntimeError(f"Expected prim is missing: {prim_path}")

            if prim_spec.HasInfo("references"):
                prim_spec.ClearInfo("references")
                removed_paths.append(prim_path)

        if not layer.Save():
            raise RuntimeError(f"Failed to save USD layer: {usd_path}")

        print(f"Saved {usd_path}", flush=True)
        print(f"Removed {len(removed_paths)} stale references:", flush=True)
        for prim_path in removed_paths:
            print(f"  {prim_path}", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
