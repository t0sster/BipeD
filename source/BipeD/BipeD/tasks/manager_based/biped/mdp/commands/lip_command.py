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

        self._swing_state = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._frozen_targets_w = torch.zeros(self.num_envs, 2, 3, device=self.device)

        self._use_step_length = cfg.nominal_step_length is not None
        self._use_step_period = cfg.step_period_s is not None
        if cfg.ranges is not None:
            if cfg.ranges.step_length is not None:
                self._use_step_length = True
            if cfg.ranges.step_period_s is not None:
                self._use_step_period = True

        self._step_length = torch.full(
            (self.num_envs, 1),
            cfg.nominal_step_length if cfg.nominal_step_length is not None else 0.0,
            device=self.device,
        )
        self._step_width = torch.full((self.num_envs, 1), cfg.nominal_step_width, device=self.device)
        self._step_period = torch.full(
            (self.num_envs, 1),
            cfg.step_period_s if cfg.step_period_s is not None else 0.0,
            device=self.device,
        )

        if cfg.ranges is not None:
            if cfg.ranges.step_length is not None:
                mean_len = sum(cfg.ranges.step_length) * 0.5
                self._step_length.fill_(mean_len)
            if cfg.ranges.step_width is not None:
                mean_width = sum(cfg.ranges.step_width) * 0.5
                self._step_width.fill_(mean_width)
            if cfg.ranges.step_period_s is not None:
                mean_period = sum(cfg.ranges.step_period_s) * 0.5
                self._step_period.fill_(mean_period)

    def __str__(self) -> str:
        msg = "LipStepCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Command format: [r_x, r_y, r_yaw, l_x, l_y, l_yaw]."""
        return self.step_target_command

    def _update_command(self):
        env = self._env  # type: ignore
        asset = env.scene[self.cfg.asset_name]

        root_pos = asset.data.root_pos_w
        root_vel = asset.data.root_lin_vel_w
        base_quat = asset.data.root_quat_w

        yaw_quat = math_utils.yaw_quat(base_quat)

        forward = self._forward.repeat(base_quat.shape[0], 1)
        forward = math_utils.quat_apply(yaw_quat, forward)
        base_heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)

        foot_pos = asset.data.body_pos_w[:, self._foot_body_ids, :]
        foot_quat = asset.data.body_quat_w[:, self._foot_body_ids, :]

        gait_command = env.command_manager.get_command("gait_command")  # type: ignore
        right_phase, left_phase, duration = gait_phase_from_command(
            env.episode_length_buf, env.step_dt, gait_command # type: ignore
        )

        right_contact = right_phase < duration
        left_contact = left_phase < duration

        both_contact = right_contact & left_contact
        both_swing = (~right_contact) & (~left_contact)

        swing_right = ~right_contact
        swing_left = ~left_contact

        ###
        tie_break = both_contact | both_swing
        swing_right[tie_break] = right_phase[tie_break] > left_phase[tie_break]
        swing_left[tie_break] = ~swing_right[tie_break]

        # swing_right[both_contact] = True
        # swing_left[both_contact] = False
        # swing_right[both_swing] = True
        # swing_left[both_swing] = False
        ###

        cmd = env.command_manager.get_command("base_velocity") # type: ignore
        cmd_vel = cmd[:, :2]
        cmd_wz = cmd[:, 2:3]
        cmd_speed = torch.norm(cmd_vel, dim=1, keepdim=True)

        if self._use_step_period:
            T = self._step_period
        elif self.cfg.step_period_s is None:
            freq = gait_command[:, 0].clamp(min=1e-3)
            T = (0.5 / freq).unsqueeze(1)
        else:
            T = torch.full((self.num_envs, 1), self.cfg.step_period_s, device=self.device)

        if self.cfg.use_cmd_heading:
            vel_heading = torch.atan2(cmd_vel[:, 1], cmd_vel[:, 0]).unsqueeze(1)
            desired_heading = math_utils.wrap_to_pi(base_heading + vel_heading + cmd_wz * T)
            heading = torch.where(
                cmd_speed > self.cfg.heading_speed_eps,
                desired_heading,
                math_utils.wrap_to_pi(base_heading + cmd_wz * T),
            )
            heading_b = math_utils.wrap_to_pi(vel_heading + cmd_wz * T)
            heading_b = torch.where(cmd_speed > self.cfg.heading_speed_eps, heading_b, cmd_wz * T)
        else:
            heading = base_heading
            heading_b = torch.zeros_like(base_heading)

        dstep_width = self._step_width
        if self._use_step_length:
            dstep_length = self._step_length
        else:
            dstep_length = None

        support_pos = torch.where(swing_right.unsqueeze(1), foot_pos[:, 1, :], foot_pos[:, 0, :])

        if self.cfg.use_base_frame:
            root_pos_plan = torch.zeros_like(root_pos)
            root_pos_plan[:, 2:3] = root_pos[:, 2:3]
            root_vel_plan = math_utils.quat_apply_inverse(yaw_quat, root_vel)
            support_pos_plan = math_utils.quat_apply_inverse(yaw_quat, support_pos - root_pos)

            target_b = compute_xcom_step_targets(
                root_pos_plan,
                root_vel_plan,
                support_pos_plan,
                cmd_vel,
                heading_b,
                T,
                dstep_width,
                dstep_length,
                swing_left,
            )

            target_vec_b = torch.zeros_like(target_b)
            target_vec_b[:, :2] = target_b[:, :2]
            target_xy_w = root_pos[:, :2] + math_utils.quat_apply(yaw_quat, target_vec_b)[:, :2]
            target = torch.zeros_like(target_b)
            target[:, :2] = target_xy_w
            target[:, 2] = heading.squeeze(1)
        else:
            target = compute_xcom_step_targets(
                root_pos,
                root_vel,
                support_pos,
                cmd_vel,
                heading,
                T,
                dstep_width,
                dstep_length,
                swing_left,
            )

        right_target = torch.zeros(self.num_envs, 3, device=self.device)
        left_target = torch.zeros(self.num_envs, 3, device=self.device)

        foot_forward = self._forward.repeat(foot_quat.shape[0], 1)
        right_forward = math_utils.quat_apply(foot_quat[:, 0, :], foot_forward)
        left_forward = math_utils.quat_apply(foot_quat[:, 1, :], foot_forward)
        right_yaw = torch.atan2(right_forward[:, 1], right_forward[:, 0])
        left_yaw = torch.atan2(left_forward[:, 1], left_forward[:, 0])

        if self.cfg.lock_target_on_swing:
            swing_now = torch.stack((swing_right, swing_left), dim=1)
            swing_start = swing_now & ~self._swing_state
            self._swing_state = swing_now

            if swing_start[:, 0].any():
                right_ids = swing_start[:, 0]
                self._frozen_targets_w[right_ids, 0, :2] = target[right_ids, :2]
                self._frozen_targets_w[right_ids, 0, 2] = target[right_ids, 2]
            if swing_start[:, 1].any():
                left_ids = swing_start[:, 1]
                self._frozen_targets_w[left_ids, 1, :2] = target[left_ids, :2]
                self._frozen_targets_w[left_ids, 1, 2] = target[left_ids, 2]

            right_target[:, :2] = torch.where(
                swing_right.unsqueeze(1), self._frozen_targets_w[:, 0, :2], foot_pos[:, 0, :2]
            )
            left_target[:, :2] = torch.where(
                swing_left.unsqueeze(1), self._frozen_targets_w[:, 1, :2], foot_pos[:, 1, :2]
            )

            right_target[:, 2] = torch.where(swing_right, self._frozen_targets_w[:, 0, 2], right_yaw)
            left_target[:, 2] = torch.where(swing_left, self._frozen_targets_w[:, 1, 2], left_yaw)
        else:
            right_target[:, :2] = torch.where(swing_right.unsqueeze(1), target[:, :2], foot_pos[:, 0, :2])
            left_target[:, :2] = torch.where(swing_left.unsqueeze(1), target[:, :2], foot_pos[:, 1, :2])

            right_target[:, 2] = torch.where(swing_right, target[:, 2], right_yaw)
            left_target[:, 2] = torch.where(swing_left, target[:, 2], left_yaw)

        self.step_target_command[:, 0:3] = right_target
        self.step_target_command[:, 3:6] = left_target

    def get_desired_step(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            return self.step_target_command
        return self.step_target_command[env_ids]

    def _resample_command(self, env_ids):
        if self.cfg.ranges is None:
            return

        r = torch.empty(len(env_ids), device=self.device)

        if self.cfg.ranges.step_length is not None:
            self._step_length[env_ids, 0] = r.uniform_(*self.cfg.ranges.step_length)
        if self.cfg.ranges.step_width is not None:
            self._step_width[env_ids, 0] = r.uniform_(*self.cfg.ranges.step_width)
        if self.cfg.ranges.step_period_s is not None:
            self._step_period[env_ids, 0] = r.uniform_(*self.cfg.ranges.step_period_s)

    def _update_metrics(self):
        pass