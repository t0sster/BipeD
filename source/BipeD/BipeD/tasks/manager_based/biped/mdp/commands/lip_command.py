from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, SceneEntityCfg

from ..lipm import compute_xcom_step_targets, gait_phase_from_command

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import LipStepCommandCfg


class LipStepCommand(CommandTerm):
    """Command generator that generates desired step targets using 3D-LIPM XCoM."""

    cfg: LipStepCommandCfg

    def __init__(self, cfg: LipStepCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)  # type: ignore

        self.step_target_command = torch.zeros(self.num_envs, 6, device=self.device)
        self.metrics = {}

        self._foot_cfg = SceneEntityCfg(cfg.asset_name, body_names=list(cfg.foot_body_names))
        self._foot_cfg.resolve(env.scene)  # type: ignore
        self._foot_body_ids = self._foot_cfg.body_ids

        self._forward = torch.tensor([1.0, 0.0, 0.0], device=self.device)

    def __str__(self) -> str:
        msg = "LipStepCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Command format: [r_x, r_y, r_yaw, l_x, l_y, l_yaw]."""
        return self.step_target_command

    def _update_command(self, env_ids):
        env = self._env  # type: ignore
        asset = env.scene[self.cfg.asset_name]

        root_pos = asset.data.root_pos_w
        root_vel = asset.data.root_lin_vel_w
        base_quat = asset.data.root_quat_w

        forward = math_utils.quat_apply(base_quat, self._forward)
        heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)

        foot_pos = asset.data.body_pos_w[:, self._foot_body_ids, :]
        foot_quat = asset.data.body_quat_w[:, self._foot_body_ids, :]

        gait_command = env.command_manager.get_command("gait_command")
        right_phase, left_phase, duration = gait_phase_from_command(
            env.episode_length_buf, env.step_dt, gait_command
        )

        right_contact = right_phase < duration
        left_contact = left_phase < duration

        both_contact = right_contact & left_contact
        both_swing = (~right_contact) & (~left_contact)

        swing_right = ~right_contact
        swing_left = ~left_contact

        swing_right[both_contact] = True
        swing_left[both_contact] = False
        swing_right[both_swing] = True
        swing_left[both_swing] = False

        cmd_vel = env.command_manager.get_command("base_velocity")[:, :2]
        freq = gait_command[:, 0].clamp(min=1e-3)
        T = (0.5 / freq).unsqueeze(1)

        dstep_width = torch.full((self.num_envs, 1), self.cfg.nominal_step_width, device=self.device)

        support_pos = torch.where(
            swing_right.unsqueeze(1),
            foot_pos[:, 1, :],
            foot_pos[:, 0, :],
        )

        target = compute_xcom_step_targets(
            root_pos,
            root_vel,
            support_pos,
            cmd_vel,
            heading,
            T,
            dstep_width,
        )

        right_target = torch.zeros(self.num_envs, 3, device=self.device)
        left_target = torch.zeros(self.num_envs, 3, device=self.device)

        right_yaw = torch.atan2(
            math_utils.quat_apply(foot_quat[:, 0, :], self._forward)[:, 1],
            math_utils.quat_apply(foot_quat[:, 0, :], self._forward)[:, 0],
        )
        left_yaw = torch.atan2(
            math_utils.quat_apply(foot_quat[:, 1, :], self._forward)[:, 1],
            math_utils.quat_apply(foot_quat[:, 1, :], self._forward)[:, 0],
        )

        right_target[:, :2] = torch.where(
            swing_right.unsqueeze(1), target[:, :2], foot_pos[:, 0, :2]
        )
        left_target[:, :2] = torch.where(
            swing_left.unsqueeze(1), target[:, :2], foot_pos[:, 1, :2]
        )

        right_target[:, 2] = torch.where(swing_right, target[:, 2], right_yaw)
        left_target[:, 2] = torch.where(swing_left, target[:, 2], left_yaw)

        self.step_target_command[:, 0:3] = right_target
        self.step_target_command[:, 3:6] = left_target

        if env_ids is not None:
            self.step_target_command[env_ids] = self.step_target_command[env_ids]

    def get_desired_step(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            return self.step_target_command
        return self.step_target_command[env_ids]

    def _resample_command(self, env_ids):
        pass

    def _update_metrics(self):
        pass