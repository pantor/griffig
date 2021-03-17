import time
from typing import Generator

from loguru import logger
import numpy as np

from ..grasp.grasp import Grasp
from _griffig import RobotPose, OrthographicImage, BoxData
from ..infer.inference import Inference
from ..infer.selection import Method, Max


class InferencePlanar(Inference):
    pass