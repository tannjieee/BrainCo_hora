"""Asset configs for Revo3 right hand in-hand rotation."""
from __future__ import annotations

import copy
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg

from hora.object_registry import ObjectTaskSpec, get_object_task_spec

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
        # Collider contact/rest offsets are authored in revo3_right_physics.usd.
        # Runtime overrides cannot traverse the hand's instanceable collider prims.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
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
        # Collider contact/rest offsets are authored in revo3_right_physics.usd.
        # Runtime overrides cannot traverse the hand's instanceable collider prims.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
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
    disable_gravity=False,
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


def _usd_object_cfg(spec: ObjectTaskSpec) -> RigidObjectCfg:
    """Build a centered, dynamic rigid-object config for a packaged USD asset."""
    if not spec.usd_path or not os.path.isfile(spec.usd_path):
        raise FileNotFoundError(f"USD asset for task {spec.name!r} was not found: {spec.usd_path}")
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=spec.usd_path,
            scale=(spec.scale, spec.scale, spec.scale),
            rigid_props=_COMMON_RIGID,
            mass_props=_COMMON_MASS,
            collision_props=_COMMON_COLLISION,
            semantic_tags=[("class", spec.name)],
        ),
        # training.usda recenters every scanned mesh around its rigid-body root,
        # so all objects share the same in-hand center as the reference cylinder.
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=spec.object_init_pos_m,
            rot=spec.object_init_quat_wxyz,
        ),
    )


def get_robot_cfg(task: str) -> ArticulationCfg:
    """Return an independent hand config with a grasp pose suited to the task."""
    spec = get_object_task_spec(task)
    if spec.hand_pose == "ball":
        template = REVO3_HAND_BALL_CFG
    elif spec.hand_pose in {"cylinder", "custom"}:
        template = REVO3_HAND_CYLINDER_CFG
    else:
        raise ValueError(f"Unsupported hand pose profile for {task!r}: {spec.hand_pose!r}")
    robot_cfg = copy.deepcopy(template)
    seed_joint_pos = robot_cfg.init_state.joint_pos
    for joint_name, joint_value in spec.hand_joint_pos_rad:
        if joint_name not in seed_joint_pos:
            raise ValueError(f"Unknown seed joint for {task!r}: {joint_name!r}")
        seed_joint_pos[joint_name] = joint_value
    return robot_cfg


def get_object_cfg(task: str) -> RigidObjectCfg:
    """Return an independent object config for a built-in or packaged asset."""
    spec = get_object_task_spec(task)
    if spec.kind == "sphere":
        object_cfg = copy.deepcopy(BALL_OBJECT_CFG)
    elif spec.kind == "cylinder":
        object_cfg = copy.deepcopy(CYLINDER_OBJECT_CFG)
    elif spec.kind == "usd":
        object_cfg = _usd_object_cfg(spec)
    else:
        raise ValueError(f"Unsupported object kind for {task!r}: {spec.kind}")
    object_cfg.init_state.pos = spec.object_init_pos_m
    object_cfg.init_state.rot = spec.object_init_quat_wxyz
    return object_cfg


def configure_env_for_object_task(env_cfg, task: str) -> ObjectTaskSpec:
    """Apply the single registry entry shared by collection and training."""
    spec = get_object_task_spec(task)
    env_cfg.object_task = spec.name
    env_cfg.object_display_name = spec.display_name
    env_cfg.object_scale = spec.scale
    env_cfg.object_size_m = spec.scaled_size_m
    env_cfg.object_usd_path = spec.usd_path or ""
    env_cfg.robot_cfg = get_robot_cfg(task)
    env_cfg.object_cfg = get_object_cfg(task)
    env_cfg.grasp_cache_path = spec.cache_stem
    env_cfg.object_rotation_axis_local = spec.rotation_axis_local
    env_cfg.target_rotation_axis_world = spec.target_axis_world
    env_cfg.object_axis_bidirectional = spec.axis_bidirectional
    env_cfg.enforce_object_axis_alignment = spec.enforce_axis_alignment
    env_cfg.object_axis_tilt_tolerance = math.radians(spec.axis_tilt_tolerance_deg)
    env_cfg.grasp_fingertip_near_threshold = spec.fingertip_near_threshold_m
    return spec
