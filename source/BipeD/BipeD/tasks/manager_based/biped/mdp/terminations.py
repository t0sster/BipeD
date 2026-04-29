from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def exceeds_max_velocity(env: ManagerBasedRLEnv, max_velocity: float) -> torch.Tensor:
    return torch.norm(env.scene["robot"].data.root_lin_vel_b, dim=1) > max_velocity