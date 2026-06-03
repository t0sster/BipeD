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

        foot_pos_w = asset.data.body_pos_w[:, self._foot_body_ids, :]
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

        vel_heading = torch.atan2(cmd_vel[:, 1], cmd_vel[:, 0]).unsqueeze(1)
        step_heading_b = torch.where(
            cmd_speed > self.cfg.heading_speed_eps,
            vel_heading,
            torch.zeros_like(vel_heading),
        )

        if self.cfg.use_cmd_heading:
            heading = math_utils.wrap_to_pi(base_heading + cmd_wz * T)
        else:
            heading = base_heading

        dstep_width = self._step_width
        if self._use_step_length:
            dstep_length = self._step_length
        else:
            dstep_length = None

        if dstep_length is None:
            dstep_length = torch.norm(cmd_vel, dim=1, keepdim=True) * T

        root_pos_plan = torch.zeros_like(root_pos)
        root_pos_plan[:, 2:3] = root_pos[:, 2:3]
        root_vel_plan = math_utils.quat_apply_inverse(yaw_quat, root_vel)
        yaw_quat_rep = yaw_quat.unsqueeze(1).expand(-1, foot_pos_w.shape[1], -1).reshape(-1, 4)
        foot_pos_b = math_utils.quat_apply_inverse(
            yaw_quat_rep,
            (foot_pos_w - root_pos.unsqueeze(1)).reshape(-1, 3),
        ).reshape(foot_pos_w.shape)
        support_pos_plan = torch.where(swing_right.unsqueeze(1), foot_pos_b[:, 1, :], foot_pos_b[:, 0, :])

        heading_dir_b = torch.stack((torch.cos(step_heading_b.squeeze(1)), torch.sin(step_heading_b.squeeze(1))), dim=1)
        speed_scale = cmd_vel[:, 0:1].abs() / (cmd_vel[:, 0:1].abs() + cmd_vel[:, 1:2].abs() + 1e-6)
        delta_along = (foot_pos_b[:, 0, :2] - foot_pos_b[:, 1, :2]).mul(heading_dir_b).sum(dim=1, keepdim=True)
        comp = self.cfg.stride_compensation_gain * speed_scale * delta_along
        comp_limit = self.cfg.stride_compensation_max_ratio * dstep_length
        comp = torch.clamp(comp, min=-comp_limit, max=comp_limit)
        swing_sign = (swing_left.float() - swing_right.float()).unsqueeze(1)
        dstep_length_eff = torch.clamp(dstep_length + swing_sign * comp, min=0.0)

        target_b = compute_xcom_step_targets(
            root_pos_plan,
            root_vel_plan,
            support_pos_plan,
            cmd_vel,
            step_heading_b,
            T,
            dstep_width,
            dstep_length_eff,
            swing_left,
        )

        target_vec_b = torch.zeros_like(target_b)
        target_vec_b[:, :2] = target_b[:, :2]
        target_xy_w = root_pos[:, :2] + math_utils.quat_apply(yaw_quat, target_vec_b)[:, :2]
        target = torch.zeros_like(target_b)
        target[:, :2] = target_xy_w
        target[:, 2] = heading.squeeze(1)

        right_target = torch.zeros(self.num_envs, 3, device=self.device)
        left_target = torch.zeros(self.num_envs, 3, device=self.device)

        foot_forward = self._forward.repeat(foot_quat.shape[0], 1)
        right_forward = math_utils.quat_apply(foot_quat[:, 0, :], foot_forward)
        left_forward = math_utils.quat_apply(foot_quat[:, 1, :], foot_forward)
        right_yaw = torch.atan2(right_forward[:, 1], right_forward[:, 0])
        left_yaw = torch.atan2(left_forward[:, 1], left_forward[:, 0])

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
            swing_right.unsqueeze(1), self._frozen_targets_w[:, 0, :2], foot_pos_w[:, 0, :2]
        )
        left_target[:, :2] = torch.where(
            swing_left.unsqueeze(1), self._frozen_targets_w[:, 1, :2], foot_pos_w[:, 1, :2]
        )

        right_target[:, 2] = torch.where(swing_right, self._frozen_targets_w[:, 0, 2], right_yaw)
        left_target[:, 2] = torch.where(swing_left, self._frozen_targets_w[:, 1, 2], left_yaw)

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
        asset = self._env.scene[self.cfg.asset_name]  # type: ignore

        base_pos = asset.data.root_pos_w
        base_quat = asset.data.root_quat_w
        foot_pos_w = asset.data.body_pos_w[:, self._foot_body_ids, :]
        foot_quat = asset.data.body_quat_w[:, self._foot_body_ids, :]

        target = self.step_target_command
        target_xy_w = torch.stack((target[:, 0:2], target[:, 3:5]), dim=1)
        target_yaw = torch.stack((target[:, 2], target[:, 5]), dim=1)

        foot_pos_b = math_utils.quat_apply_inverse(
            base_quat.unsqueeze(1).repeat(1, foot_pos_w.shape[1], 1),
            foot_pos_w - base_pos.unsqueeze(1),
        )

        target_pos_w = torch.zeros(target_xy_w.shape[0], target_xy_w.shape[1], 3, device=target_xy_w.device)
        target_pos_w[:, :, :2] = target_xy_w
        target_pos_w[:, :, 2] = base_pos[:, 2:3]
        target_pos_b = math_utils.quat_apply_inverse(
            base_quat.unsqueeze(1).repeat(1, target_pos_w.shape[1], 1),
            target_pos_w - base_pos.unsqueeze(1),
        )

        pos_err = torch.norm(foot_pos_b[:, :, :2] - target_pos_b[:, :, :2], dim=2).mean(dim=1)

        forward = self._forward.repeat(foot_quat.shape[0], 1)
        right_forward = math_utils.quat_apply(foot_quat[:, 0, :], forward)
        left_forward = math_utils.quat_apply(foot_quat[:, 1, :], forward)
        foot_yaw = torch.stack(
            (
                torch.atan2(right_forward[:, 1], right_forward[:, 0]),
                torch.atan2(left_forward[:, 1], left_forward[:, 0]),
            ),
            dim=1,
        )
        yaw_err = torch.abs(math_utils.wrap_to_pi(foot_yaw - target_yaw)).mean(dim=1)

        self.metrics["error_step_pos"] = pos_err
        self.metrics["error_step_yaw"] = yaw_err

    # def _update_metrics(self):
    #     pass