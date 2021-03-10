from dataclasses import dataclass

from frankx import Affine
from _griffig import RobotPose

@dataclass
class Grasp:
    pose: RobotPose
    estimated_reward: float
    method: str
    clamping_distance: float
