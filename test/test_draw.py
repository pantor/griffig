from time import time
from pathlib import Path

import cv2
import numpy as np

from griffig import BoxData, Renderer, OrthographicImage, Gripper, RobotPose
from griffig.utility.image import draw_pose, draw_around_box


class Loader:
    data_path = Path(__file__).parent / 'data'

    @classmethod
    def get_image(cls, episode_id: str):
        mat_color = cv2.imread(str(cls.data_path / f'{episode_id}-rc.jpeg'), cv2.IMREAD_UNCHANGED).astype(np.uint16) * 255
        mat_depth = cv2.imread(str(cls.data_path / f'{episode_id}-rd.jpeg'), cv2.IMREAD_UNCHANGED).astype(np.uint16) * 255
        mat_depth = cv2.cvtColor(mat_depth, cv2.COLOR_BGR2GRAY)

        mat = np.concatenate([mat_color, np.expand_dims(mat_depth, axis=2)], axis=2)
        return OrthographicImage(mat, 2000.0, 0.22, 0.41)


if __name__ == '__main__':
    box_data = BoxData([-0.002, -0.0065, 0.372], [0.174, 0.282, 0.22 + 0.068])
    renderer = Renderer((752, 480), 2000.0, 0.19, box_data)
    gripper = Gripper(min_stroke=0.0, max_stroke=0.086, finger_width=0.024, finger_extent=0.008)

    image = Loader.get_image('1')

    pose = RobotPose(x=0.02, y=0.06, z=0.32, a=0.5, d=0.05)

    start = time()

    renderer.draw_box_on_image(image)
    renderer.draw_gripper_on_image(image, gripper, pose)

    # draw_around_box(image, box_data)
    # draw_pose(image, pose)

    print(time() - start)

    cv2.imwrite('../tmp/image.png', image.mat[:, :, :3])
    # cv2.waitKey(1500)
