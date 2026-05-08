"""Sub-module containing command generator for base height."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import BaseHeightCommandCfg


class BaseHeightCommand(CommandTerm):
    """Command generator that samples a target base height."""

    cfg: BaseHeightCommandCfg

    def __init__(self, cfg: BaseHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)  # type: ignore

        self._height_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics = {}

    def __str__(self) -> str:
        msg = "BaseHeightCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The base height command. Shape is (num_envs, 1)."""
        return self._height_command

    def _resample_command(self, env_ids):
        r = torch.empty(len(env_ids), device=self.device)
        self._height_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass
