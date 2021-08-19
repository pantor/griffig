from enum import Enum
from typing import List
from pathlib import Path


class ModelArchitecture(str, Enum):
    Planar = 'planar'  # Model-based z, and bc=0
    PlanarTypes = 'planar-types'  # Model-based z, and bc=0, for multiple learned types
    PlanarZ = 'planar-z'  # Regression-based z, and bc=0
    PlanarSemantic = 'planar-semantic'  # Semantic Grasping
    Lateral = 'lateral'  # Model-based z, b, and c
    ActorCritic = 'actor-critic'  # Actor-critic
    ModelBasedConvolution = 'model-based-convolution'  # Model-based convolution
    NonFCNActorCritic = 'non-fcn-actor-critic'  # Non-fully-convolutional actor-critic
    NonFCNPlanar = 'non-fcn-planar'  # Non-fully-convolutional planar


class ModelData:
    def __init__(
        self,
        name: str = None,
        path: Path = None,
        architecture: ModelArchitecture = None,
        pixel_size: float = None,
        depth_diff: float = None,
        gripper_widths: List[float] = None,
        description: str = None,
        size_area_cropped = None,
        size_result = None,
        task = None,
        input = None,
        input_type = None,
        output = None,
        version = None,
    ):
        self.name = name
        self.path = path
        self.architecture = architecture
        self.pixel_size = pixel_size
        self.depth_diff = depth_diff
        self.gripper_widths = gripper_widths
        self.description = description
        self.size_area_cropped = size_area_cropped
        self.size_result = size_result
        self.task = task
        self.input = input
        self.input_type = input_type
        self.output = output
        self.version = version

    def to_dict(self):
        return self.__dict__

