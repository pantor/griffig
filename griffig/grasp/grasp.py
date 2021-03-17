from dataclasses import dataclass

from frankx import Affine
from _griffig import RobotPose

from ..infer.selection import Method


@dataclass
class Grasp:
    pose: RobotPose
    estimated_reward: float
    calculation_duration: float

    method: str
    # clamping_distance: float
