import cv2
import numpy as np

from pyaffx import Affine
from .grasp import Grasp
from _griffig import BoxData, Gripper, OrthographicImage
from ..utility.image2 import get_area_of_interest


class Checker:
    def __init__(self, converter, lateral_grasps=False, avoid_collisions=True):
        self.converter = converter
        self.lateral_grasps = lateral_grasps
        self.avoid_collisions = avoid_collisions

        self.area_size = (200, 200)

    def find(self, grasp_gen, image: OrthographicImage, box_data: BoxData, gripper: Gripper):
        grasp = next(grasp_gen)

        while not self.check_safety(grasp, image, box_data, gripper):
            try:
                grasp = next(grasp_gen)
            except StopIteration:
                return None

        grasp_gen.close()

        # grasp.pose = grasp.pose * gripper.robot_to_tip
        return grasp

    def check_safety(self, grasp: Grasp, image: OrthographicImage, box_data: BoxData, gripper: Gripper):
        self.converter.index_to_action(grasp)

        image_area = get_area_of_interest(image, grasp.pose, self.area_size, self.area_size)

        self.converter.calculate_z(image_area, grasp)

        if self.lateral_grasps:
            self.converter.calculate_b(image_area, grasp)
            self.converter.calculate_c(image_area, grasp)

        return (np.isfinite([grasp.pose.x, grasp.pose.y, grasp.pose.z]).all() and self.is_grasp_inside_box(grasp, box_data))

    def is_grasp_inside_box(self, grasp: Grasp, box_data: BoxData):
        if not box_data:
            return True

        half_stroke = 0.5 * (grasp.stroke + 0.002)  # [m]
        gripper_b1 = (grasp.pose * Affine(y=half_stroke)).translation()[0:2]
        gripper_b2 = (grasp.pose * Affine(y=-half_stroke)).translation()[0:2]

        check_contour = np.array([(c[0], c[1]) for c in box_data.contour], dtype=np.float32)
        jaw1_inside_box = cv2.pointPolygonTest(check_contour, tuple(gripper_b1), False) >= 0
        jaw2_inside_box = cv2.pointPolygonTest(check_contour, tuple(gripper_b2), False) >= 0

        return jaw1_inside_box and jaw2_inside_box
