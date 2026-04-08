import math

from isaaclab.utils import configclass

from BipeD.assets.config.biped_config import BD_CFG
from BipeD.tasks.manager_based.biped.biped_env_cfg import BipedEnvCfg
from terrains_cfg import (
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
class BipedBaseEnvCfg(BipedEnvCfg):
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

        ### continue

        self.viewer.origin_type = "env"

@configclass
class BipedBaseEnvCfg_Play(BipedBaseEnvCfg):
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


#####################
# Rough Environment #
#####################

@configclass
class BipedRoughEnvCfg(BipedBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        #TODO: check 

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG

@configclass
class BipedRoughEnvCfg_Play(BipedBaseEnvCfg_Play):
    def __post_init__(self):
        super().__post_init__()

        #TODO: check

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_PLAY_CFG


##################
# Stairs Terrain #
##################

@configclass
class BipedStairsEnvCfg(BipedBaseEnvCfg):
    pass

@configclass
class BipedStairsEnvCfg_Play(BipedBaseEnvCfg_Play):
    pass