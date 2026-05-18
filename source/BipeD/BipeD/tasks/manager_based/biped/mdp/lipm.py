from __future__ import annotations

import torch


def gait_phase_from_command(
    episode_length_buf: torch.Tensor,
    step_dt: float,
    gait_command: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute gait phases and contact duration from the gait command.

    gait_command: [frequency, phase_offset, contact_duration]
    Returns:
        right_phase, left_phase, contact_duration
    """
    freq = gait_command[:, 0].clamp(min=1e-3)
    offset = gait_command[:, 1]
    duration = gait_command[:, 2].clamp(0.05, 0.95)

    phase = torch.remainder(episode_length_buf * step_dt * freq, 1.0)
    right_phase = phase
    left_phase = torch.remainder(phase + offset, 1.0)
    return right_phase, left_phase, duration


def compute_xcom_step_targets(
    root_pos_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    support_foot_pos_w: torch.Tensor,
    cmd_vel_xy: torch.Tensor,
    heading: torch.Tensor,
    T: torch.Tensor,
    dstep_width: torch.Tensor,
    dstep_length: torch.Tensor | None = None,
    left_swing: torch.Tensor | None = None,
    use_mid_stance: bool = False,
    g: float = 9.81,
) -> torch.Tensor:
    """Compute XCoM step target for the swing foot.

    All inputs are per-env tensors.
    Returns:
        target_pos_yaw (N, 3): [x, y, yaw]
    """
    z = root_pos_w[:, 2:3].clamp(min=0.05)
    w = torch.sqrt(g / z)

    x0 = root_pos_w[:, 0:1] - support_foot_pos_w[:, 0:1]
    y0 = root_pos_w[:, 1:2] - support_foot_pos_w[:, 1:2]
    vx0 = root_lin_vel_w[:, 0:1]
    vy0 = root_lin_vel_w[:, 1:2]

    x_f = x0 * torch.cosh(T * w) + vx0 * torch.sinh(T * w) / w
    vx_f = x0 * w * torch.sinh(T * w) + vx0 * torch.cosh(T * w)
    y_f = y0 * torch.cosh(T * w) + vy0 * torch.sinh(T * w) / w
    vy_f = y0 * w * torch.sinh(T * w) + vy0 * torch.cosh(T * w)

    x_f_world = x_f + support_foot_pos_w[:, 0:1]
    y_f_world = y_f + support_foot_pos_w[:, 1:2]

    eicp_x = x_f_world + vx_f / w
    eicp_y = y_f_world + vy_f / w

    if dstep_length is None:
        step_length = torch.norm(cmd_vel_xy, dim=1, keepdim=True) * T
        stride_length = 2.0 * step_length
    else:
        if use_mid_stance:
            stride_length = dstep_length
            step_length = 0.5 * dstep_length
        else:
            step_length = dstep_length
            stride_length = 2.0 * dstep_length

    b_x = step_length / (torch.exp(T * w) - 1.0)  # pyright: ignore[reportOptionalOperand]
    b_y = dstep_width / (torch.exp(T * w) + 1.0)

    original_offset_x = -b_x
    original_offset_y = -b_y
    if left_swing is not None:
        original_offset_y[left_swing] = b_y[left_swing]

    offset_x = torch.cos(heading) * original_offset_x - torch.sin(heading) * original_offset_y
    offset_y = torch.sin(heading) * original_offset_x + torch.cos(heading) * original_offset_y

    target = torch.zeros(root_pos_w.shape[0], 3, device=root_pos_w.device)
    target[:, 0] = (eicp_x + offset_x).squeeze(1)
    target[:, 1] = (eicp_y + offset_y).squeeze(1)

    if use_mid_stance:
        forward_x = torch.cos(heading) * (0.5 * stride_length)
        forward_y = torch.sin(heading) * (0.5 * stride_length)
        target[:, 0] += forward_x.squeeze(1)
        target[:, 1] += forward_y.squeeze(1)
    target[:, 2] = heading.squeeze(1)
    return target