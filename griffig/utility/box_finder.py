import cv2
import numpy as np

from _griffig import BoxData


class BoxFinder:
    def __init__(self, method='highest-contour', height_buffer=20):
        self.method = method
        self.height_buffer = height_buffer

    @staticmethod
    def contour_rect_area(x):
        _, _, w, h = cv2.boundingRect(x)
        return w * h

    def find(self, image):
        assert self.method == 'highest-contour'

        depth = image.mat[:, :, 3]

        max_value = np.max(depth) - self.height_buffer
        _, depth = cv2.threshold(depth, max_value, 255, cv2.THRESH_BINARY)
        canny_output = cv2.Canny(depth, 255 / 2, 255)
        contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 2:
            return None

        # Find second largest contour (inside of the box)
        sorted_contours = sorted(contours, key=self.contour_rect_area, reverse=True)
        x, y, w, h = cv2.boundingRect(sorted_contours[1])

        box_center = np.array([y + (h - image.mat.shape[0]) / 2, -x + (-w + image.mat.shape[1]) / 2, 0.0]) / image.pixel_size
        box_size = np.array([h, -w, 0.1]) / image.pixel_size
        return BoxData(box_center, box_size)
