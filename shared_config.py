from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

class ControlMode(Enum):
    IDLING = auto()
    OBSTACLE_AVOIDANCE = auto()
    WAYPOINT_NAVIGATION = auto()
    HOLD_POSITION = auto()

@dataclass
class PIDGains:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0

@dataclass
class PIDState:
    integral: float = 0.0
    prev_error: float = 0.0

@dataclass
class ControlOutput:
    left_speed: float = 0.0
    right_speed: float = 0.0
    linear_cmd: float = 0.0
    angular_cmd: float = 0.0
    dist_to_wp: float = float('inf')
    current_mode: ControlMode = ControlMode.IDLING
    heading_error: float = 0.0