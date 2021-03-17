from dataclasses import dataclass
from enum import Enum
from typing import List


class ModelArchitecture(str, Enum):
    Planar = 'planar'  # Model-based z, and bc=0
    PlanarTypes = 'planar-types'  # Model-based z, and bc=0, for multiple learned types
    PlanarZ = 'planar-z'  # Regression-based z, and bc=0
    Lateral = 'lateral'  # Model-based z, b, and c
    ActorCritic = 'actor-critic'  # Actor-critic architecture


@dataclass
class ModelData:
    name: str = None
    path: str = None
    architecture: ModelArchitecture = None
    pixel_size: float = None
    depth_diff: float = None
    gripper_widths: List[float] = None
    description: str = None
    task: str = None
    input: List[str] = None
    input_type: str = None
    output: List[str] = None
