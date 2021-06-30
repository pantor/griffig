from pathlib import Path
import unittest

import cv2

from griffig import BoxData, Grasp, Griffig, Gripper, Renderer, RobotPose

from loader import Loader


class DrawTestCase(unittest.TestCase):
    output_path = Path(__file__).parent / 'data'

    def setUp(self):
        self.box_data = BoxData([-0.002, -0.0065, 0.0], [0.174, 0.282, 0.0])
        self.renderer = Renderer((752, 480), 0.41, 2000.0, 0.19)
        self.gripper = Gripper(min_stroke=0.0, max_stroke=0.086, finger_width=0.024, finger_extent=0.008)

    def test_draw_box_on_image(self):
        image = Loader.get_image('1')
        Griffig.draw_box_on_image(image, self.box_data)
        img_c = Griffig.convert_to_pillow_image(image, channels='RGB')
        img_c.save(self.output_path / 'image-box-c.png')
        img_d = Griffig.convert_to_pillow_image(image, channels='D')
        img_d.save(self.output_path / 'image-box-d.png')

    def test_draw_gipper_on_image(self):
        image = Loader.get_image('1')
        pose = RobotPose(x=0.04, y=-0.01, z=-0.34, a=0.0, b=-0.3, d=0.05)

        self.renderer.draw_gripper_on_image(image, self.gripper, pose)
        img_c = Griffig.convert_to_pillow_image(image, channels='RGB')
        img_c.save(self.output_path / 'image-gripper-c.png')
        img_d = Griffig.convert_to_pillow_image(image, channels='D')
        img_d.save(self.output_path / 'image-gripper-d.png')

    def test_draw_pose_on_image(self):
        image = Loader.get_image('1')
        grasp = Grasp(pose=RobotPose(x=0.04, y=-0.01, z=-0.34, a=0.0, b=-0.3), stroke=0.05)

        img = Griffig.draw_grasp_on_image(image, grasp)
        img.save(self.output_path / 'image-pose-cd.png')
        img_c = Griffig.draw_grasp_on_image(image, grasp, channels='RGB')
        img_c.save(self.output_path / 'image-pose-c.png')
        img_c = Griffig.draw_grasp_on_image(image, grasp, channels='D')
        img_c.save(self.output_path / 'image-pose-d.png')

    def test_check_collision(self):
        image = Loader.get_image('1')
        pose1 = RobotPose(x=0.04, y=-0.01, z=-0.34, a=0.0, b=-0.3, d=0.05)
        self.assertFalse(self.renderer.check_gripper_collision(image, self.gripper, pose1))

        pose2 = RobotPose(x=0.04, y=0.0, z=-0.34, a=0.0, b=-0.2, d=0.05)
        self.assertTrue(self.renderer.check_gripper_collision(image, self.gripper, pose2))


if __name__ == '__main__':
    unittest.main()
