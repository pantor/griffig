from time import time
from pathlib import Path

import cv2
import numpy as np

from griffig import BoxData, Renderer, OrthographicImage, Gripper, RobotPose
from griffig.utility.image import draw_pose, draw_around_box

from loader import Loader


if __name__ == '__main__':
    box_data = BoxData([-0.002, -0.0065, 0.0], [0.174, 0.282, 0.0])
    renderer = Renderer((752, 480), 2000.0, 0.19, box_data)
    gripper = Gripper(min_stroke=0.0, max_stroke=0.086, finger_width=0.024, finger_extent=0.008)

    image = Loader.get_image('1')

    pose = RobotPose(x=0.04, y=0.00, z=-0.34, a=0.0, b=0.7, d=0.05)

    start = time()

    # renderer.draw_box_on_image(image)
    # renderer.draw_gripper_on_image(image, gripper, pose)
    print(renderer.check_gripper_collision(image, gripper, pose))

    # draw_around_box(image, box_data)
    # draw_pose(image, pose)

    print(time() - start)

    cv2.imwrite('../tmp/image.png', image.mat[:, :])
    # cv2.waitKey(1500)
