import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .gait_command import GaitCommand
from .base_height_command import BaseHeightCommand
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

    @configclass
    class Ranges:
        """Uniform distribution ranges for step parameters."""

        step_length: tuple[float, float] | None = None
        """Range for step length [m]. If None, uses nominal or velocity-based length."""
        step_width: tuple[float, float] | None = None
        """Range for step width [m]. If None, uses nominal value."""
        step_period_s: tuple[float, float] | None = None
        """Range for step period [s]. If None, uses nominal or gait frequency."""

    asset_name: str = "robot"
    update_period: float = 0.1

    # feet order must be [RIGHT, LEFT]
    foot_body_names: tuple[str, str] = ("R4_Link_ankle", "L4_Link_ankle")

    # Desired step geometry and timing (half-step duration).
    nominal_step_length: float | None = None
    nominal_step_width: float = 0.22
    step_period_s: float | None = None

    # Use commanded velocity direction for step yaw when speed is non-trivial.
    use_cmd_heading: bool = True
    heading_speed_eps: float = 1e-3


    ranges: Ranges | None = None
    """Optional ranges for step parameters."""

    resampling_time_range: tuple[float, float] = MISSING  # type: ignore


@configclass
class BaseHeightCommandCfg(CommandTermCfg):
    """Configuration for base height command generator."""

    class_type: type = BaseHeightCommand

    @configclass
    class Ranges:
        """Uniform distribution ranges for base height."""

        height: tuple[float, float] = MISSING # type: ignore
        """Range for base height [m]."""

    ranges: Ranges = MISSING # type: ignore
    """Distribution ranges for base height."""

    resampling_time_range: tuple[float, float] = MISSING  # type: ignore