from pathlib import Path
import unittest

import cv2

from griffig import BoxData, Grasp, Griffig, RobotPose

from loader import Loader


class WidgetTestCase(unittest.TestCase):
    output_path = Path(__file__).parent / 'data'

    def setUp(self):
        self.box_data = BoxData([-0.002, -0.0065, 0.0], [0.174, 0.282, 0.0])

    def test_draw_box_on_image(self):
        self.assertTrue(self.box_data.is_pose_inside(RobotPose(x=0.04, y=-0.01, z=-0.34, d=0.05)))
        self.assertFalse(self.box_data.is_pose_inside(RobotPose(x=0.08, y=-0.01, z=-0.34, a=0.4, d=0.05)))
        self.assertTrue(self.box_data.is_pose_inside(RobotPose(x=0.02, y=-0.12, z=-0.34, a=-1.0, b=-0.3, d=0.07)))
        self.assertFalse(self.box_data.is_pose_inside(RobotPose(x=0.02, y=-0.18, z=-0.34, a=-1.0, b=-0.3, d=0.01)))
        self.assertTrue(self.box_data.is_pose_inside(RobotPose(x=0.02, y=-0.10, z=-0.34, a=-1.4, b=-0.2, d=0.03)))
        self.assertFalse(self.box_data.is_pose_inside(RobotPose(x=0.02, y=-0.10, z=-0.34, a=-1.4, b=0.4, d=0.03)))

        # image = Loader.get_image('1')
        # grasp = Grasp(pose=RobotPose(x=0.02, y=-0.10, z=-0.34, a=-1.4, b=0.4), stroke=0.03)
        # Griffig.draw_grasp_on_image(image, grasp)
        # Griffig.draw_box_on_image(image, self.box_data, draw_lines=True)
        # print(self.box_data.is_pose_inside(RobotPose(grasp.pose, d=grasp.stroke)))
        # cv2.imwrite(str(self.output_path / 'image-box-check-d.jpg'), image.mat[:, :, 3] / 255)


if __name__ == '__main__':
    unittest.main()
