from time import time
from pathlib import Path

import cv2
import numpy as np

from griffig import BoxData, Renderer, OrthographicImage


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
    
    image = Loader.get_image('1')

    start = time()
    image_drawn = renderer.draw_box_on_image(image)

    print(time() - start)

    cv2.imwrite('tmp/image.png', image_drawn.mat[:, :, :3])
    # cv2.waitKey(1500)
