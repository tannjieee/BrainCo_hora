"""Asset configs for Revo3 right hand in-hand rotation."""
from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg

_REVO3_USD = os.path.join(os.path.dirname(__file__), "../../../assets/usd/revo3_right.usd")

# Object initial positions in env-local coordinates.
OBJECT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
# Hand initial pose (-25 deg around world X-axis)
HAND_INIT_POS = (0.0, 0.0, 1.5)
HAND_INIT_ROT = (0.59636781, 0.37992820, -0.37992820, 0.59636781)

CYLINDER_INIT_POS = (0.000, -0.08, 1.635)
BALL_INIT_POS = (0.000, -0.08, 1.65)

REVO3_HAND_CYLINDER_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/hand",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_REVO3_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=HAND_INIT_POS,
        rot=HAND_INIT_ROT,
        joint_pos={
            "right_thumb_CMP_joint":  1.65, "right_thumb_CMR_joint":  1.123107,
            "right_thumb_MCP_joint":  0.35, "right_thumb_PIP_joint":  0.20,
            "right_thumb_DIP_joint":  0.00,
            "right_index_MPR_joint":  0.25, "right_index_MCP_joint":  1.20,
            "right_index_PIP_joint":  0.30, "right_index_DIP_joint":  0.00,
            "right_middle_MPR_joint": 0.00, "right_middle_MCP_joint": 0.95,
            "right_middle_PIP_joint": 0.20, "right_middle_DIP_joint": 0.00,
            "right_ring_MPR_joint":  -0.20, "right_ring_MCP_joint":   0.95,
            "right_ring_PIP_joint":   0.20, "right_ring_DIP_joint":   0.00,
            "right_little_MPR_joint": -0.25, "right_little_MCP_joint": 1.20,
            "right_little_PIP_joint": 0.30, "right_little_DIP_joint": 0.00,
        },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["right_.*"],
            effort_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.01,
            armature=0.001,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

REVO3_HAND_BALL_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/hand",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_REVO3_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=HAND_INIT_POS,
        rot=HAND_INIT_ROT,
        joint_pos={
            "right_thumb_CMP_joint":  1.65, "right_thumb_CMR_joint":  1.35,
            "right_thumb_MCP_joint":  0.50, "right_thumb_PIP_joint":  0.00,
            "right_thumb_DIP_joint":  0.00,
            "right_index_MPR_joint": -0.17, "right_index_MCP_joint":  1.40,
            "right_index_PIP_joint":  0.00, "right_index_DIP_joint":  0.00,
            "right_middle_MPR_joint": 0.00, "right_middle_MCP_joint": 1.10,
            "right_middle_PIP_joint": 0.00, "right_middle_DIP_joint": 0.00,
            "right_ring_MPR_joint":   0.20, "right_ring_MCP_joint":   1.10,
            "right_ring_PIP_joint":   0.00, "right_ring_DIP_joint":   0.00,
            "right_little_MPR_joint": 0.12, "right_little_MCP_joint": 1.40,
            "right_little_PIP_joint": 0.05, "right_little_DIP_joint": 0.00,
        },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["right_.*"],
            effort_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.01,
            armature=0.001,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

_COMMON_RIGID = sim_utils.RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    # Scene gravity is zero.  Gravity is applied as a per-environment
    # equivalent force so every object can use an independent direction.
    disable_gravity=True,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=0,
    sleep_threshold=0.005,
    stabilization_threshold=0.0025,
    max_depenetration_velocity=1000.0,
)
_COMMON_MASS = sim_utils.MassPropertiesCfg(mass=0.10)
_COMMON_COLLISION = sim_utils.CollisionPropertiesCfg(
    collision_enabled=True, contact_offset=0.002, rest_offset=0.0)
_COMMON_MATERIAL = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)


BALL_OBJECT_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/object",
    spawn=sim_utils.SphereCfg(
        radius=0.030,
        rigid_props=_COMMON_RIGID,
        mass_props=_COMMON_MASS,
        collision_props=_COMMON_COLLISION,
        physics_material=_COMMON_MATERIAL,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=BALL_INIT_POS, rot=OBJECT_INIT_ROT),
)

CYLINDER_OBJECT_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/object",
    spawn=sim_utils.CylinderCfg(
        radius=0.03, height=0.070,
        rigid_props=_COMMON_RIGID,
        mass_props=_COMMON_MASS,
        collision_props=_COMMON_COLLISION,
        physics_material=_COMMON_MATERIAL,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=CYLINDER_INIT_POS, rot=OBJECT_INIT_ROT),
)


CYLINDER_RADIUS_MM = tuple(range(25, 36))
"""Supported cylinder-radius bins in millimetres."""

# One 50-environment cycle realizes the requested distribution exactly:
# nominal 30 mm occupies 20/50 = 40%, every other bin occupies 3/50 = 6%.
# The shuffled-looking deterministic order avoids placing all nominal objects
# in one contiguous block while keeping geometry/cache/radius metadata aligned.
CYLINDER_RADIUS_SLOT_MM = (
    # First 34 slots: 30 mm x14, every other radius x2.  This makes
    # num_envs=16384 exact: 6554 nominal and 983 for every other bin.
    30, 29, 30, 31, 28, 30, 32, 27, 30, 33,
    26, 30, 34, 25, 30, 35, 30, 30, 29, 31,
    30, 28, 32, 30, 27, 33, 30, 26, 34, 30,
    25, 35, 30, 30,
    # Remaining 16 slots: 30 mm x6, every other radius x1.
    29, 30, 31, 28, 30, 32, 27, 30, 33, 26,
    30, 34, 25, 30, 35, 30,
)

if len(CYLINDER_RADIUS_SLOT_MM) != 50:
    raise RuntimeError("CYLINDER_RADIUS_SLOT_MM must contain exactly 50 slots")
if set(CYLINDER_RADIUS_SLOT_MM) != set(CYLINDER_RADIUS_MM):
    raise RuntimeError("CYLINDER_RADIUS_SLOT_MM must cover every supported radius bin")
for _radius_mm in CYLINDER_RADIUS_MM:
    _expected_slots = 20 if _radius_mm == 30 else 3
    if CYLINDER_RADIUS_SLOT_MM.count(_radius_mm) != _expected_slots:
        raise RuntimeError(
            f"Radius {_radius_mm} mm must occupy {_expected_slots}/50 slots"
        )


def make_cylinder_object_cfg(
    radius_mm: int | None = None,
    *,
    use_radius_distribution: bool = True,
) -> RigidObjectCfg:
    """Return a cylinder config for one radius or the deterministic 40/6% mixture.

    The returned object owns a new spawn config.  ``configclass.copy()`` is a
    shallow dataclass replacement, so mutating ``cfg.spawn.radius`` after a
    plain copy would also mutate :data:`CYLINDER_OBJECT_CFG`.
    """
    base_spawn = CYLINDER_OBJECT_CFG.spawn
    if radius_mm is not None:
        radius_mm = int(radius_mm)
        if radius_mm not in CYLINDER_RADIUS_MM:
            raise ValueError(
                f"radius_mm must be one of {CYLINDER_RADIUS_MM}, got {radius_mm}"
            )
        return CYLINDER_OBJECT_CFG.replace(
            spawn=base_spawn.replace(radius=radius_mm / 1000.0)
        )

    if not use_radius_distribution:
        return CYLINDER_OBJECT_CFG.replace(spawn=base_spawn.replace(radius=0.030))

    cylinder_cfgs = [
        base_spawn.replace(radius=slot_radius_mm / 1000.0)
        for slot_radius_mm in CYLINDER_RADIUS_SLOT_MM
    ]
    return CYLINDER_OBJECT_CFG.replace(
        spawn=MultiAssetSpawnerCfg(
            assets_cfg=cylinder_cfgs,
            # Slot order, geometry readback, privileged radius and cache
            # selection must remain deterministic and mutually aligned.
            random_choice=False,
        )
    )
