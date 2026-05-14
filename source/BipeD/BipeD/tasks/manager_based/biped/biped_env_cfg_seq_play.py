from isaaclab.utils import configclass

from BipeD.tasks.manager_based.biped.biped_env_cfg import BDLipEnvCfg_Play


@configclass
class BDLipEnvCfg_SeqPlay(BDLipEnvCfg_Play):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.add_base_mass = None  # type: ignore

        self.commands.base_velocity.resampling_time_range = (1e6, 1e6)
        self.commands.lip_step_command.resampling_time_range = (1e6, 1e6)
        self.commands.gait_command.resampling_time_range = (1e6, 1e6)
        self.commands.base_height_command.resampling_time_range = (1e6, 1e6)
