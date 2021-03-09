from time import time

import cv2

from _griffig import BoxData, Griffig
from data.loader import Loader
from frankx import Affine


if __name__ == '__main__':
    image = Loader.get_image('human-grasping-2', '2021-03-04-18-36-22-507', 0, 'rcd', 'v')

    box_data = BoxData([0, 0, 0], [0.282, 0.174, 0.068], Affine())
    print(box_data.contour)

    griffig = Griffig(box_data)

    start = time()
    image_drawn = griffig.draw_box_on_image(image.mat)

    print(time() - start)

    cv2.imshow('image.png', image_drawn[:, :, :3])
    cv2.waitKey(1500)
