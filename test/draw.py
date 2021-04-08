from time import time

import cv2

from griffig import BoxData, Renderer
from data.loader import Loader


if __name__ == '__main__':
    image = Loader.get_image('human-grasping-2', '2021-03-04-18-36-22-507', 0, 'rcd', 'v')
    # [-0.002, -0.0065, 0.372], [0.174, 0.282, 0.068]
    box_data = BoxData([-0.002, -0.0065, 0.372], [0.174, 0.282, 0.22 + 0.068])
    print(box_data.contour)

    renderer = Renderer(image.mat.shape[1::-1], 2000.0, 0.19, box_data)

    start = time()
    image_drawn = renderer.draw_box_on_image(image)

    print(time() - start)

    cv2.imwrite('tmp/image.png', image_drawn.mat[:, :, :3])
    # cv2.waitKey(1500)
