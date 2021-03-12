from typing import Dict, Iterable

import cv2
import numpy as np
import tensorflow.keras as tk  # pylint: disable=E0401
import tensorflow.keras.backend as tkb  # pylint: disable=E0401

import inference.selection as Selection
from griffig import RobotPose, OrthographicImage, BoxData
from utils.image import draw_around_box, get_inference_image
from frankx import Affine


class Inference:
    def __init__(
            self,
            model_data,
            verbose=0,
            seed: int = None,
        ):
        self.model_data = model_data

        self.size_area_cropped = (200, 200)
        self.size_result = (32, 32)
        self.size_cropped = (110, 110)
        self.scale_factors = (self.size_area_cropped[0] / self.size_result[0], self.size_area_cropped[1] / self.size_result[1])

        self.a_space = np.linspace(-np.pi/2 + 0.1, np.pi/2 - 0.1, 20)  # [rad] # Don't use a=0.0 -> even number
        self.keep_indixes = None

        self.verbose = verbose
        self.rs = np.random.default_rng(seed=seed)

    def pose_from_index(self, index, index_shape, main_image: OrthographicImage, resolution_factor=2.0) -> RobotPose:
        return RobotPose(Affine(
            x=resolution_factor * self.scale_factors[0] * main_image.position_from_index(index[1], index_shape[1]),
            y=resolution_factor * self.scale_factors[1] * main_image.position_from_index(index[2], index_shape[2]),
            a=self.a_space[index[0]],
        ).inverse(), d=0.0)

    def transform_for_prediction(
            self,
            image: OrthographicImage,
            scale_around_zero=False,
    ):
        # Rotate images
        rotated = []
        for a in self.a_space:
            dst_depth = get_inference_image(image, Affine(a=a), self.size_cropped, self.size_area_cropped, self.size_result)
            rotated.append(dst_depth.mat)

        result = np.array(rotated) / np.iinfo(image.mat.dtype).max

        if len(result.shape) == 3:
            result = np.expand_dims(result, axis=-1)

        if self.verbose:
            cv2.imwrite('/tmp/test-input-c.png', result[0][0, :, :, :3] * 255)
            cv2.imwrite('/tmp/test-input-d.png', result[0][0, :, :, 3:] * 255)

        if scale_around_zero:
            return 2 * result - 1.0
        else:
            return result

    @classmethod
    def keep_array_at_last_indixes(cls, array, indixes) -> None:
        mask = np.zeros(array.shape)
        mask[:, :, :, indixes] = 1
        array *= mask
