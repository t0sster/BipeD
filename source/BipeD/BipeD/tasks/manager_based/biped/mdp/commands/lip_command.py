from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import LipStepCommandCfg


class LipStepCommand(CommandTerm):
    """Command generator that generates position of desired steps by LIPM."""

    cfg: LipStepCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: LipStepCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env) # type: ignore

        # create buffers to store the command
        # command format: [p_x, p_y, rotation=atan(v_y, v_x)]
        self.step_target_command = torch.zeros(self.num_envs, 3, device=self.device)
        # create metrics dictionary for logging
        self.metrics = {}

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GaitCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The gait command. Shape is (num_envs, 3)."""
        return self.step_target_command

    def _update_command(self, env_ids):
        #TODO: call LIPM MPC
        # Example: self.step_targets[env_ids] = mpc.compute(env.root_states[env_ids])
        pass
    
    def get_desired_step(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            return self.step_target_command
        return self.step_target_command[env_ids]
    
    def _resample_command(self, env_ids):
        #TODO: implement
        pass

    def _update_metrics(self):
        """Update the metrics based on the current state.

        In this implementation, we don't track any specific metrics.
        """
        pass