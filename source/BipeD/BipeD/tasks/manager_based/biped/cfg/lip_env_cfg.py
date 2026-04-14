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
            static_friction=1.0,
            dynamic_friction=1.0,
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

    # robot
    # robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore
    robot: ArticulationCfg = MISSING # type: ignore

    ##TODO: continue with cfg 

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
    
    #TODO

    def __post_init__(self):
        self.base_velocity.asset_name = "robot"
        self.base_velocity.heading_command = True
        self.base_velocity.debug_vis = True
        self.base_velocity.heading_control_stiffness = 1.0
        self.base_velocity.resampling_time_range = (0.0, 5.0)
        self.base_velocity.rel_standing_envs = 0.2
        self.base_velocity.rel_heading_envs = 0.0
        self.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.25, 0.25), ang_vel_z=(-0.75, 0.75), heading=(-math.pi, math.pi)
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
        pass

    @configclass
    class CriticCfg(ObsGroup):
        pass


@configclass
class EventsLipCfg:
    """Event specifications for the MDP."""
    pass


@configclass
class RewardsLipCfg:
    """Reward specifications for the MDP."""
    pass


@configclass
class TerminationsLipCfg:
    """Termination terms for the MDP"""
    pass


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

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        scene: BDLipSceneCfg = BDLipSceneCfg(num_envs=2048, env_spacing=2.5)
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = 2 * self.decimation
        self.seed = 42

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt