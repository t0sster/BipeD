import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .gait_command import GaitCommand
from .lip_command import LipStepCommand


@configclass
class UniformGaitCommandCfg(CommandTermCfg):
    """Configuration for the gait command generator."""

    class_type: type = GaitCommand  # Specify the class type for dynamic instantiation

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gait parameters."""

        frequencies: tuple[float, float] = MISSING # type: ignore
        """Range for gait frequencies [Hz]."""
        offsets: tuple[float, float] = MISSING # type: ignore
        """Range for phase offsets [0-1]."""
        durations: tuple[float, float] = MISSING # type: ignore
        """Range for contact durations [0-1]."""

    ranges: Ranges = MISSING # type: ignore
    """Distribution ranges for the gait parameters."""

    resampling_time_range: tuple[float, float] = MISSING # type: ignore
    """Time interval for resampling the gait (in seconds)."""


@configclass
class LipStepCommandCfg(CommandTermCfg):
    """Configuration for the LIPM step command generator."""

    class_type: type = LipStepCommand

    asset_name: str = "robot"
    update_period: float = 0.1

    # feet order must be [RIGHT, LEFT]
    foot_body_names: tuple[str, str] = ("R4_Link_ankle", "L4_Link_ankle")

    nominal_step_width: float = 0.22

    resampling_time_range: tuple[float, float] = MISSING  # type: ignore