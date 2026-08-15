"""Environment config for Revo3 right hand in-hand rotation."""
from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .assets import make_cylinder_object_cfg

_REVO3_USD = os.path.join(os.path.dirname(__file__), "../../../assets/usd/revo3_right.usd")


@configclass
class Revo3HandHoraEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    action_space = 21
    observation_space = 141  # 3 frames x 47 dims (21 joint_pos + 21 targets + 5 fingertip force magnitudes)
    prop_hist_len = 30
    # [object_pos_delta(3), friction(1), mass(1), com(3), gravity_direction_world(3),
    #  normalized_radius(1), cylinder_axis_world(3), object_angular_velocity(3),
    #  object_linear_velocity(3)]
    priv_info_dim = 21
    state_space = 0
    asymmetric_obs = False
    decimation = 12
    clip_obs = 5.0
    clip_actions = 1.0
    action_scale = 1 / 24
    torque_control = True
    # Per-joint-type PD gains — identified from real hardware dynamics (Dynamic_identication/controller_para/parameter.yaml)
    pgain_dict: dict = {
        "thumb_CMP":     16.4,   # thumb base CMP: high stiffness
        "thumb_CMR":      0.7,   # thumb base CMR: low stiffness
        "thumb_flexion":  1.2,   # thumb MCP + PIP: moderate
        "DIP":            8.0,   # all 5 DIP joints: high stiffness
        "MPR":            0.7,   # 4 finger spread (MPR): low stiffness
        "MCP":            0.6,   # 4 finger MCP: low stiffness
        "PIP":            0.8,   # 4 finger PIP: low stiffness
    }
    dgain_dict: dict = {
        "thumb_CMP":     0.23,   # thumb base CMP
        "thumb_CMR":     0.02,   # thumb base CMR
        "thumb_flexion": 0.09,   # thumb MCP + PIP
        "DIP":           0.10,   # all 5 DIP
        "MPR":           0.04,   # 4 finger MPR
        "MCP":           0.014,  # 4 finger MCP: very low damping
        "PIP":           0.027,  # 4 finger PIP: low damping (covers middle/ring/little, index 0.0014 is noise floor)
    }


    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240, render_interval=2,
        # Per-environment gravity is applied as an equivalent world-frame
        # force at the object COM; scene gravity must remain disabled.
        gravity=(0.0, 0.0, 0.0),
        physx=PhysxCfg(
            solver_type=1, max_position_iteration_count=8, max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=8388608, gpu_max_rigid_patch_count=5 * 2**18,
        ),
    )

    hand_init_pose = ((0.0, 0.0, 1.5), (0.59636781, 0.37992820, -0.37992820, 0.59636781))
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/hand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_REVO3_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0, max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8, solver_velocity_iteration_count=0,
                sleep_threshold=0.005, stabilization_threshold=0.0005, fix_root_link=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.002, rest_offset=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=hand_init_pose[0], rot=hand_init_pose[1],
        ),
        actuators={
            "fingers": IdealPDActuatorCfg(
                joint_names_expr=["right_.*"], stiffness=None, damping=None,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    actuated_joint_names = [
        "right_thumb_CMP_joint", "right_thumb_CMR_joint",
        "right_thumb_MCP_joint", "right_thumb_PIP_joint", "right_thumb_DIP_joint",
        "right_index_MPR_joint", "right_index_MCP_joint",
        "right_index_PIP_joint", "right_index_DIP_joint",
        "right_middle_MPR_joint", "right_middle_MCP_joint",
        "right_middle_PIP_joint", "right_middle_DIP_joint",
        "right_ring_MPR_joint", "right_ring_MCP_joint",
        "right_ring_PIP_joint", "right_ring_DIP_joint",
        "right_little_MPR_joint", "right_little_MCP_joint",
        "right_little_PIP_joint", "right_little_DIP_joint",
    ]
    fingertip_body_names = [
        "right_thumb_DIP_Link", "right_index_DIP_Link",
        "right_middle_DIP_Link", "right_ring_DIP_Link", "right_little_DIP_Link",
    ]
    elastomer_body_names = [
        "right_thumb_DIP_Link", "right_index_DIP_Link",
        "right_middle_DIP_Link", "right_ring_DIP_Link", "right_little_DIP_Link",
    ]
    contact_sensor = []

    # Keep the standalone config consistent with train.py: direct users of
    # Revo3HandHoraEnvCfg also receive the 11-radius deterministic mixture.
    object_cfg: RigidObjectCfg = make_cylinder_object_cfg(use_radius_distribution=True)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16384, env_spacing=0.75, replicate_physics=False)

    drop_radial_distance = 0.020
    drop_axial_distance = 0.020
    reset_angle_diff = 45 / 180 * math.pi
    reset_random_quat = False

    rot_axis = (0, 0, 1)
    angvel_clip_min = -0.5
    angvel_clip_max = 0.5
    rotate_reward_scale = 2.5
    cylinder_tilt_tolerance = 10 / 180 * math.pi
    cylinder_tilt_penalty_scale = -0.25
    off_axis_angvel_penalty_scale = -0.2
    xy_drift_tolerance = 0.005
    xy_drift_penalty_scale = -0.15
    z_drift_tolerance = 0.005
    z_drift_penalty_scale = -0.25
    drop_penalty_scale = -5.0
    pos_diff_penalty_scale = -0.4
    torque_penalty_scale = -0.1
    work_penalty_scale = -0.5

    grasp_cache_path = 'cache/revo3_right_grasp_cylinder'
    grasp_cache_sequential = False
    # Production training must fail fast if a radius-specific cache is absent.
    # Collection/visualization tools explicitly disable this and use the
    # configured default pose while they create or inspect a cache.
    strict_grasp_caches = True
    randomize_cylinder_radius = True
    cylinder_radius_bins_mm: tuple = tuple(range(25, 36))
    cylinder_radius_nominal_mm = 30
    cylinder_radius_normalization_half_range_mm = 5

    joint_noise_scale = 0.02
    # Encoder zero-offset error. One offset is sampled per environment/joint at
    # reset and remains fixed for the whole episode; it only affects the joint
    # position seen by the policy, not the simulated joint state or controller.
    randomize_joint_zero = True
    joint_zero_offset_lower = -0.02  # rad
    joint_zero_offset_upper = 0.02   # rad
    enable_tactile = True
    enable_contact_in_obs = True   # Tactile Stage1/Stage2/deployment share the same contact channels.
    binary_contact = False
    enable_contact_pos = False
    disable_tactile_ids = []
    # Contact force is sampled once per 0.05 s policy step (20 Hz).  No
    # additional multi-step force window is applied in the training env.
    contact_threshold = 0.05
    contact_latency = 0.0
    contact_sensor_noise = 0.01
    # Fingertip-force magnitudes are normalized before entering the policy:
    # force_obs = force_magnitude_newtons * contact_force_scale + N(0, std).
    # The additive noise is sampled independently every policy step in the
    # normalized observation space.
    contact_force_scale = 0.1
    randomize_contact_force = True
    contact_force_noise_std = 0.05
    dof_limits_scale = 0.9

    randomize_pd_gains = True
    randomize_p_gain_scale_lower = 0.5
    randomize_p_gain_scale_upper = 2
    randomize_d_gain_scale_lower = 0.5
    randomize_d_gain_scale_upper = 2
    randomize_friction = True
    randomize_friction_scale_lower = 0.5
    randomize_friction_scale_upper = 2.0
    elastomer_base_friction = 0.8
    metal_base_friction = 0.1
    object_base_friction = 0.5
    randomize_com = True
    randomize_com_lower = -0.01
    randomize_com_upper = 0.01
    randomize_mass = True
    randomize_mass_lower = 0.05
    randomize_mass_upper = 0.20

    force_scale = 2
    random_force_prob_scalar = 0.25
    force_decay = 0.9
    force_decay_interval = 0.08

    randomize_gravity_direction = True
    gravity_magnitude = 9.81
    drop_reset_rate_window = 200          # 10 seconds at 20 Hz
    drop_stable_reset_rate = 0.003
    debug_show_axes = False

    def __post_init__(self):
        super().__post_init__()
        expected_observation_space = 3 * (2 * self.action_space + len(self.fingertip_body_names))
        if self.observation_space != expected_observation_space:
            raise ValueError(
                f"observation_space must be {expected_observation_space} for "
                f"{self.action_space} joints and {len(self.fingertip_body_names)} fingertip force magnitudes"
            )
        if self.joint_zero_offset_lower > self.joint_zero_offset_upper:
            raise ValueError("joint_zero_offset_lower must be <= joint_zero_offset_upper")
        if self.contact_force_scale <= 0.0:
            raise ValueError("contact_force_scale must be positive")
        if self.contact_force_noise_std < 0.0:
            raise ValueError("contact_force_noise_std must be non-negative")
        if self.randomize_mass and self.randomize_mass_lower < 0.05:
            raise ValueError("randomize_mass_lower must be at least 0.05 kg")
        if self.randomize_mass_lower > self.randomize_mass_upper:
            raise ValueError("randomize_mass_lower must be <= randomize_mass_upper")
        if self.gravity_magnitude <= 0.0:
            raise ValueError("gravity_magnitude must be positive")
        if self.drop_reset_rate_window <= 0:
            raise ValueError("drop_reset_rate_window must be positive")
        if not 0.0 <= self.drop_stable_reset_rate <= 1.0:
            raise ValueError("drop_stable_reset_rate must lie in [0, 1]")
        if self.drop_radial_distance <= 0.0 or self.drop_axial_distance <= 0.0:
            raise ValueError("drop distances must be positive")
        if self.cylinder_radius_normalization_half_range_mm <= 0:
            raise ValueError("cylinder_radius_normalization_half_range_mm must be positive")
        if tuple(float(v) for v in self.sim.gravity) != (0.0, 0.0, 0.0):
            raise ValueError("sim.gravity must be zero when using per-environment gravity")
        if self.randomize_cylinder_radius:
            if self.scene.replicate_physics:
                raise ValueError("radius-randomized geometry requires scene.replicate_physics=False")
            bins = tuple(int(v) for v in self.cylinder_radius_bins_mm)
            if bins != tuple(range(25, 36)):
                raise ValueError("cylinder_radius_bins_mm must be the 11 integer bins from 25 to 35 mm")
        for name in self.elastomer_body_names:
            self.contact_sensor.append(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/hand/{name}",
                history_length=3,
                track_contact_points=True,
                filter_prim_paths_expr=["/World/envs/env_.*/object"],
            ))
