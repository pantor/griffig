from dataclasses import dataclass

from _griffig import RobotPose


@dataclass
class Grasp:
    pose: RobotPose
    estimated_reward: float
    calculation_duration: float
    
    method: str
    # clamping_distance: float
