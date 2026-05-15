# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UniformNoise
from isaaclab.sim import DomeLightCfg, MdlFileCfg, RigidBodyMaterialCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg

from BipeD.tasks.manager_based.biped import mdp


##
# Scene definition
##

@configclass
class BDLipSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    # ground = AssetBaseCfg(
    #     prim_path="/World/ground",
    #     spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    # )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.5,
            dynamic_friction=1.2,
            restitution=1.0,
        ),
        visual_material=MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            + "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"),
    )

    robot: ArticulationCfg = MISSING # type: ignore

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=4, track_air_time=True, update_period=0.0
    )


################
# MDP settings #
################


@configclass
class CommandsLipCfg(CommandsCfg):
    """Command specifications for the MDP."""

    gait_command = mdp.UniformGaitCommandCfg(
        resampling_time_range=(5.0, 5.0),  # Fixed resampling time of 5 seconds
        debug_vis=False,  # No debug visualization needed
        ranges=mdp.UniformGaitCommandCfg.Ranges(
            frequencies=(1.5, 2.5),  # Gait frequency range [Hz]
            offsets=(0.5, 0.5),  # Phase offset range [0-1]
            durations=(0.5, 0.5),  # Contact duration range [0-1]
        ),
    )
    
    lip_step_command = mdp.LipStepCommandCfg(
        asset_name="robot",
        update_period=0.1,
        resampling_time_range=(1e6, 1e6),
    )

    base_height_command = mdp.BaseHeightCommandCfg(
        resampling_time_range=(1e6, 1e6),
        ranges=mdp.BaseHeightCommandCfg.Ranges(height=(0.30, 0.30)),
    )

    def __post_init__(self):
        self.base_velocity.asset_name = "robot"
        self.base_velocity.heading_command = True
        self.base_velocity.debug_vis = True
        self.base_velocity.heading_control_stiffness = 1.0
        self.base_velocity.resampling_time_range = (0.0, 5.0)
        self.base_velocity.rel_standing_envs = 0.2
        self.base_velocity.rel_heading_envs = 1.0
        self.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.25, 0.25), ang_vel_z=(-0.75, 0.75), heading=(-math.pi, math.pi)
            # lin_vel_x=(-0.5, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(-math.pi, math.pi)
        )


@configclass
class ActionsLipCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["J_L0", "J_R0", "J_L1", "J_R1", "J_L2", "J_R2", "J_L3", "J_R3", "J_L4_ankle", "J_R4_ankle"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsLipCfg:
    """Observation specifications for the MDP."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observation for policy group"""

        # base_height = ObsTerm(func=mdp.base_pos_z, noise=UniformNoise(operation="add", n_min=-0.05, n_max=0.05))
        # base_lin_vel_world = ObsTerm(func=mdp.base_lin_vel, noise=UniformNoise(operation="add", n_min=-0.2, n_max=0.2))

        # base_heading = ObsTerm(func=mdp.base_heading, noise=UniformNoise(operation="add", n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=UniformNoise(operation="add", n_min=-0.1, n_max=0.1))
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=UniformNoise(operation="add", n_min=-0.05, n_max=0.05))
        
        foot_states_right = ObsTerm(func=mdp.foot_states_right)
        foot_states_left = ObsTerm(func=mdp.foot_states_left)

        foot_target_right = ObsTerm(func=mdp.step_command_right)
        foot_target_left = ObsTerm(func=mdp.step_command_left)

        commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"}
        )

        base_height_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_height_command"}
        )
        
        gait_phase = ObsTerm(func=mdp.get_gait_phase)

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=UniformNoise(operation="add", n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=UniformNoise(operation="add", n_min=-0.5, n_max=0.5))

    @configclass
    class CriticCfg(ObsGroup):
        """Observation for critic group"""

        # Policy observation
        
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        proj_gravity = ObsTerm(func=mdp.projected_gravity)
        
        foot_states_right = ObsTerm(func=mdp.foot_states_right)
        foot_states_left = ObsTerm(func=mdp.foot_states_left)

        step_command_right = ObsTerm(func=mdp.step_command_right)
        step_command_left = ObsTerm(func=mdp.step_command_left)

        commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"}
        )
        
        base_height_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_height_command"}
        )
        
        gait_phase = ObsTerm(func=mdp.get_gait_phase)

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel)

        # Privileged observation

        base_height = ObsTerm(func=mdp.base_pos_z)
        base_heading = ObsTerm(func=mdp.base_heading)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        robot_feet_contact_force = ObsTerm(
            func=mdp.robot_feet_contact_force,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*[LR]4.*ankle.*"]),
            },
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_pos = ObsTerm(func=mdp.robot_pos)
        robot_vel = ObsTerm(func=mdp.robot_vel)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsLipCfg:
    """Event specifications for the MDP."""
    
    # startup
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    # reset
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )


@configclass
class LipRewardParamsCfg:
    """Reward parameters for the LIP environment."""

    rew_shaping: float = 0.1
    base_height_target: float = 0.28
    
    step_position_sigma: float = 0.05
    step_yaw_sigma: float = 0.25
    heading_sigma: float = 0.25
    contact_threshold: float = 1.0
    contact_sigma: float = 0.25
    feet_air_time_scale: float = 0.4
    feet_air_time_min_threshold: float = 0.1
    stand_still_lin_threshold: float = 0.1
    stand_still_ang_threshold: float = 0.1

    weights: dict[str, float] = {
        # rewards
        "rew_lin_vel_xy": 4.0,
        "rew_ang_vel_z": 2.0,
        "rew_step_tracking": 3.0,
        "rew_heading": 0.5,
        "rew_feet_air_time": 1.0,
        "contact_schedule": 0.0,
        # penalities
        "base_height": -1.0,
        "joint_torques": -1e-4,
        "joint_vel": -1e-3,
        "joint_pos_limits": -1.0,
        "stand_still": -0.05,
        "no_contact": -0.2,
        "foot_slip": -0.2,
        "action_smoothness": -1e-3,
        "ang_vel_xy": -1e-2,
        "lin_vel_z": -1e-1,
        "flat_orientation": -1.0,
    }


@configclass
class RewardsLipCfg:
    """Reward specifications for the MDP."""
    
    # Rewards
    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    rew_step_tracking = RewTerm(
        func=mdp.step_command_tracking,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["R4_Link_ankle", "L4_Link_ankle"]),
            "command_name": "lip_step_command",
            "position_sigma": 0.01,
            "yaw_sigma": 0.05,
        },
    )

    rew_heading = RewTerm(
        func=mdp.heading_tracking,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "heading_sigma": 0.05,
        },
    )

    rew_feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["R4_Link_ankle", "L4_Link_ankle"]),
            "command_name": "base_velocity",
            "threshold": 0.4,
            "gait_command_name": "gait_command",
            "swing_time_scale": 0.5,
            "min_threshold": 0.0,
            "dense": False,
            "contact_force_threshold": 1.0,
            "single_support_only": True,
        },
    )

    rew_contact_schedule = RewTerm(
        func=mdp.contact_schedule,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["R4_Link_ankle", "L4_Link_ankle"]),
            "command_name": "gait_command",
            "threshold": 1.0,
            "sigma": 0.05,
        },
    )

    # Regularization
    pen_base_height = RewTerm(
        func=mdp.base_height_tracking_l2,
        weight=1.0,
        params={"command_name": "base_height_command"}
    )
    pen_joint_torq = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    pen_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-3)
    pen_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    pen_stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.05,
        params={
            "lin_threshold": 0.1,
            "ang_threshold": 0.1,
        },
    )
    pen_no_contact = RewTerm(
        func=mdp.no_contact,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["R4_Link_ankle", "L4_Link_ankle"]),
        },
    )
    pen_foot_slip = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["R4_Link_ankle", "L4_Link_ankle"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["R4_Link_ankle", "L4_Link_ankle"]),
            "contact_threshold": 1.0,
        },
    )
    pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-1e-3) # type: ignore
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-1e-2)
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1e-1)
    pen_flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1)


    def apply(self, params: "LipRewardParamsCfg") -> None:
        weight_map = {
            "rew_lin_vel_xy": "rew_lin_vel_xy",
            "rew_ang_vel_z": "rew_ang_vel_z",
            "rew_step_tracking": "rew_step_tracking",
            "rew_heading": "rew_heading",
            "rew_feet_air_time": "rew_feet_air_time",
            "contact_schedule": "rew_contact_schedule",
            "base_height": "pen_base_height",
            "joint_torques": "pen_joint_torq",
            "joint_vel": "pen_joint_vel",
            "joint_pos_limits": "pen_joint_pos_limits",
            "stand_still": "pen_stand_still",
            "no_contact": "pen_no_contact",
            "foot_slip": "pen_foot_slip",
            "action_smoothness": "pen_action_smoothness",
            "ang_vel_xy": "pen_ang_vel_xy",
            "lin_vel_z": "pen_lin_vel_z",
            "flat_orientation": "pen_flat_orientation",
        }

        for key, term_name in weight_map.items():
            getattr(self, term_name).weight = params.weights[key]

        param_map = {
            ("rew_lin_vel_xy", "std"): ("rew_shaping", lambda v: math.sqrt(v)),
            ("rew_ang_vel_z", "std"): ("rew_shaping", lambda v: math.sqrt(v)),
            ("rew_step_tracking", "position_sigma"): ("step_position_sigma", None),
            ("rew_step_tracking", "yaw_sigma"): ("step_yaw_sigma", None),
            ("rew_heading", "heading_sigma"): ("heading_sigma", None),
            ("rew_feet_air_time", "swing_time_scale"): ("feet_air_time_scale", None),
            ("rew_feet_air_time", "min_threshold"): ("feet_air_time_min_threshold", None),
            ("rew_contact_schedule", "threshold"): ("contact_threshold", None),
            ("rew_contact_schedule", "sigma"): ("contact_sigma", None),
            ("pen_stand_still", "lin_threshold"): ("stand_still_lin_threshold", None),
            ("pen_stand_still", "ang_threshold"): ("stand_still_ang_threshold", None),
        }

        for (term_name, param_key), (src_attr, transform) in param_map.items():
            value = getattr(params, src_attr)
            if transform is not None:
                value = transform(value)
            getattr(self, term_name).params[param_key] = value


@configclass
class TerminationsLipCfg:
    """Termination terms for the MDP"""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True
    )

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )
    
    max_velocity = DoneTerm(
        func=mdp.exceeds_max_velocity,
        params={"max_velocity": 3.0},
    )


##
# Environment configuration
##


@configclass
class BipedLipEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: BDLipSceneCfg = BDLipSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsLipCfg = ObservationsLipCfg()
    actions: ActionsLipCfg = ActionsLipCfg()
    commands: CommandsLipCfg = CommandsLipCfg()
    events: EventsLipCfg = EventsLipCfg()
    # MDP settings
    rewards: RewardsLipCfg = RewardsLipCfg()
    terminations: TerminationsLipCfg = TerminationsLipCfg()
    reward_params: LipRewardParamsCfg = LipRewardParamsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        self.scene: BDLipSceneCfg = BDLipSceneCfg(num_envs=2048, env_spacing=2.5)
        # general settings
        self.decimation = 4
        self.episode_length_s = 25.0
        # viewer settings
        self.viewer.eye = (10.0, 10.0, 5.0)
        self.viewer.lookat = (-5.0, 0.0, 0.0)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = 2 * self.decimation
        self.seed = 42

        self.rewards.apply(self.reward_params)

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt