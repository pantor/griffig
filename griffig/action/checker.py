import cv2
import numpy as np

from pyaffx import Affine
from _griffig import BoxData, Grasp, Gripper, OrthographicImage, RobotPose
from ..utility.image import get_area_of_interest


class Checker:
    def __init__(self, converter, lateral_grasps=False, avoid_collisions=True):
        self.converter = converter
        self.lateral_grasps = lateral_grasps
        self.avoid_collisions = avoid_collisions

    def find_grasp(self, grasp_gen, image: OrthographicImage, box_data: BoxData = None, gripper: Gripper = None):
        grasp = next(grasp_gen)

        while not self.check(grasp, image, box_data, gripper):
            try:
                grasp = next(grasp_gen)
            except StopIteration:
                return None

        grasp_gen.close()

        if gripper:
            grasp.pose = grasp.pose * gripper.offset

        return grasp

    def check(self, grasp: Grasp, image: OrthographicImage, box_data: BoxData = None, gripper: Gripper = None):
        self.converter.index_to_action(grasp)

        area_size = (0.1, 0.1)  # [m]
        image_area = get_area_of_interest(image, grasp.pose, (int(image.pixel_size * area_size[0]), int(image.pixel_size * area_size[1])))

        self.converter.calculate_z(image_area, grasp)

        if self.lateral_grasps:
            self.converter.calculate_b(image_area, grasp)
            self.converter.calculate_c(image_area, grasp)

        is_safe = np.isfinite([grasp.pose.x, grasp.pose.y, grasp.pose.z]).all()

        if box_data:
            is_safe &= box_data.is_pose_inside(RobotPose(grasp.pose, d=grasp.stroke))

        if self.avoid_collisions:
            pass  # Do the gripper rendering here

        return is_safe
