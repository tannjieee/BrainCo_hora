"""Bake Revo3 collider offsets into the source prims used by instances."""

from pathlib import Path

from isaacsim import SimulationApp


CONTACT_OFFSET = 0.002
REST_OFFSET = 0.0


def main() -> None:
    simulation_app = SimulationApp({"headless": True})

    try:
        from pxr import PhysxSchema, Usd

        usd_path = (
            Path(__file__).resolve().parents[1]
            / "assets/usd/configuration/revo3_right_physics.usd"
        )
        stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
        if stage is None:
            raise RuntimeError(f"Could not open USD stage: {usd_path}")

        stage.SetEditTarget(stage.GetRootLayer())
        # Applying an API schema recomposes the stage, so freeze the traversal
        # result before editing to avoid skipping prims mid-iteration.
        colliders = [
            prim
            for prim in stage.TraverseAll()
            if str(prim.GetPath()).startswith("/colliders/")
            and "PhysicsCollisionAPI" in prim.GetAppliedSchemas()
        ]
        collider_paths: list[str] = []
        for prim in colliders:
            prim_path = str(prim.GetPath())
            if prim.IsInstanceProxy():
                raise RuntimeError(f"Refusing to edit instance proxy: {prim_path}")

            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision_api.CreateContactOffsetAttr(CONTACT_OFFSET).Set(CONTACT_OFFSET)
            collision_api.CreateRestOffsetAttr(REST_OFFSET).Set(REST_OFFSET)
            collider_paths.append(prim_path)

        if not collider_paths:
            raise RuntimeError(f"No source colliders found in {usd_path}")
        if not stage.GetRootLayer().Save():
            raise RuntimeError(f"Failed to save USD layer: {usd_path}")

        print(f"Saved {usd_path}", flush=True)
        print(
            f"Authored contactOffset={CONTACT_OFFSET} and restOffset={REST_OFFSET} "
            f"on {len(collider_paths)} source colliders",
            flush=True,
        )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
