import sys
from time import time

sys.path.insert(0, '../../build')
sys.path.insert(0, '../../../scripts')

from griffig import BoxContour, Griffig
from data.loader import Loader

import cv2
import numpy as np

image = Loader.get_image('human-grasping-2', '2021-03-04-18-36-22-507', 0, 'rcd', 'v')

contour = BoxContour([0, 0, 0], [0.282, 0.174, 0.068])
print(contour.corners)

griffig = Griffig(contour)

start = time()
image_drawn = griffig.draw_box_on_image(image.mat)

print(time() - start)

cv2.imshow('image.png', image_drawn[:, :, 3])
cv2.waitKey(1500)
