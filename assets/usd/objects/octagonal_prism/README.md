# Regular octagonal prism

Locally generated analytic mesh; no external model dependencies.
Eight rectangular side faces and two octagonal caps, centered at the origin,
long axis +Z. Circumradius 30 mm, across flats 55.433 mm, height 70 mm.
16 shared vertices, outward face winding, flat normals, no subdivision.
Collision uses convex decomposition of the same already-convex mesh, matching
the existing asset workflow and supporting live pose-editor scaling. Direct
convexHull cooking invalidated PhysX tensor views when preview transforms changed.
Nominal mass 0.1 kg; static/dynamic friction 1.0, restitution 0.

Task: `octagonal_prism`. The manifest stores scale and the editable seed.
The initial 21-joint pose is copied from the built-in cylinder as an unvalidated
editing starting point. No grasp cache has been collected for this object.

Open the editor from the project root:

```bash
/home/tan/miniconda3/envs/env_isaaclab/bin/python tools/view_init_pose.py \
  --task octagonal_prism --num_envs 1 --edit_pose
```

Use **Save to manifest.json** after editing. The editor changes object position,
rotation, uniform scale and all 21 joints. This does not collect grasp samples.
