import math

from isaaclab.utils import configclass

from BipeD.assets.config.biped_config import BD_CFG
from BipeD.tasks.manager_based.biped.cfg import BipedEnvCfg, BipedLipEnvCfg
from .cfg.terrains_cfg import (
    BLIND_ROUGH_TERRAINS_CFG,
    BLIND_ROUGH_TERRAINS_PLAY_CFG,
    STAIRS_TERRAINS_CFG,
    STAIRS_TERRAINS_PLAY_CFG
)

from isaaclab.sensors import RayCasterCfg, patterns
from BipeD.tasks.manager_based.biped import mdp
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg


####################
# Base Environment #
####################

@configclass
class BDBaseEnvCfg(BipedEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = BD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore
        self.scene.robot.init_state.joint_pos = {
            "J_L0":   0.0,
            "J_L1":  0.08,
            "J_L2":  0.56,
            "J_L3":  -1.12,
            "J_L4_ankle": -0.57,

            "J_R0":   0.0,
            "J_R1":  -0.08,
            "J_R2":  -0.56,
            "J_R3":  1.12,
            "J_R4_ankle": 0.57
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.25, 0.25)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"

        self.observations.policy.heights = None  # type: ignore
        self.observations.critic.heights = None  # type: ignore

        self.viewer.origin_type = "env"

        self.terminations.max_velocity.params["max_velocity"] = 3.0

@configclass
class BDBaseEnvCfg_Play(BDBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.push_robot = None
        # remove random base mass addition event
        # self.events.add_base_mass = None


#################
# Rough Terrain #
#################

@configclass
class BDRoughEnvCfg(BDBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()


        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG

@configclass
class BDRoughEnvCfg_Play(BDBaseEnvCfg_Play):
    def __post_init__(self):
        super().__post_init__()


        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_PLAY_CFG


##################
# Stairs Terrain #
##################

###################
# LIP Environment #
###################

@configclass
class BDStairsEnvCfg(BDBaseEnvCfg):
    pass

@configclass
class BDStairsEnvCfg_Play(BDBaseEnvCfg_Play):
    pass

        self.observations.policy.heights = None  # type: ignore
        self.observations.critic.heights = None  # type: ignore

###################
# LIP Environment #
###################

@configclass
class BDLipEnvCfg(BipedLipEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = BD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore
        self.scene.robot.init_state.joint_pos = {
            # Left
            "J_L0":   0.0,
            "J_L1":  0.08,
            "J_L2":  0.56,
            "J_L3":  -1.12,
            "J_L4_ankle": -0.57,
            # Right
            "J_R0":   0.0,
            "J_R1":  -0.08,
            "J_R2":  -0.56,
            "J_R3":  1.12,
            "J_R4_ankle": 0.57
        }
        
        self.reward_params.base_height_target = 0.30
        self.reward_params.weights = {
            # rewards
            "rew_lin_vel_xy": 4.0,
            "rew_ang_vel_z": 2.0,
            "rew_step_tracking": 3.0,
            "rew_heading": 0.0,
            "contact_schedule": 9.0,
            # penalities
            "base_height": 1.0,
            "joint_torques": -1e-4,
            "joint_vel": -1e-3,
            "joint_pos_limits": -10,
            "action_smoothness": -1e-3,
            "ang_vel_xy": -1e-2,
            "lin_vel_z": -1e-1,
            "flat_orientation": -1,
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.25, 0.25)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"

        self.observations.policy.heights = None  # type: ignore
        self.observations.critic.heights = None  # type: ignore

        self.viewer.origin_type = "env"

        self.terminations.max_velocity.params["max_velocity"] = 3.0

        # Step command settings (match reference-style explicit step geometry).
        self.commands.lip_step_command.nominal_step_length = 0.02
        self.commands.lip_step_command.nominal_step_width = 0.2
        self.commands.lip_step_command.step_period_s = 0.24
        self.commands.lip_step_command.use_cmd_heading = True

        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.3), 
            lin_vel_y=(-0.1, 0.1), 
            ang_vel_z=(-0.75, 0.75), 
            heading=(-math.pi, math.pi)
        )

@configclass
class BDLipEnvCfg_Play(BDLipEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # remove random pushing event
        # self.events.push_robot = None

        # remove random base mass addition event
        self.events.add_base_mass = None # type: ignore
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.3), 
            lin_vel_y=(-0.1, 0.1), 
            ang_vel_z=(-0.75, 0.75),
            heading=(-math.pi, math.pi)
        )
