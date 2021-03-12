from typing import Tuple

from _griffig import RobotPose, OrthographicImage
from utils.image import get_area_of_interest


class ModelBasedCalculation:
    def __init__(self, with_z=False, with_bc=False):
        self.with_z = with_z
        self.with_bc = with_bc

    def calculate(self, image: OrthographicImage, pose: RobotPose):
        area_image = get_area_of_interest(image, pose, (200, 200), (200, 200))

        if self.with_z:
            pose.z = self.calculate_z(area_image, pose)

        if self.with_bc:
            pose.b, pose.c = self.calculate_bc(area_image, pose)

    def calculate_z(self, area_image: OrthographicImage, pose: RobotPose) -> float:
        return 0.0

    def calculate_bc(self, area_image: OrthographicImage, pose: RobotPose) -> Tuple[float, float]:
        return 0.0, 0.0