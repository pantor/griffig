from time import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from pyaffx import Affine
from _griffig import BoxData, Grasp, Gripper, OrthographicImage
from ..infer.inference_base import InferenceBase
from ..infer.selection import Method, Max
from ..utility.image import draw_around_box2, get_box_projection, get_inference_image


class InferencePlanarSemantic(InferenceBase):
    def infer(self, image: OrthographicImage, object_image: OrthographicImage, method: Method, box_data: BoxData = None, gripper: Gripper = None):
        assert object_image is not None

        start = time()

        input_images = self.get_input_images(image, box_data)
        input_object_images = [get_inference_image(object_image, Affine(a=a), (224, 224), (224, 224), (224, 224), return_mat=True) for a in self.a_space]

        input_object_images = np.array(input_object_images) / np.iinfo(object_image.mat.dtype).max

        pre_duration = time() - start
        start = time()

        estimated_grasp_reward, estimated_object_reward = self.model([input_images, [input_object_images]])

        nn_duration = time() - start
        start = time()

        estimated_reward = estimated_object_reward * estimated_grasp_reward # np.cbrt(estimated_object_reward * np.power(estimated_grasp_reward, 2))

        if gripper:
            possible_indices = gripper.consider_indices(self.model_data.gripper_widths)
            self.set_last_dim_to_zero(estimated_reward, np.invert(possible_indices))

        for _ in range(estimated_reward.size):
            index_raveled = method(estimated_reward)
            index = np.unravel_index(index_raveled, estimated_reward.shape)

            action = Grasp()
            action.index = index[3]
            action.pose = self.pose_from_index(index, estimated_reward.shape, image)
            action.pose.z = np.nan
            action.estimated_reward = estimated_reward[index]
            action.detail_durations = {
                'pre': pre_duration,
                'nn': nn_duration,
                'post': time() - start,
            }
            action.calculation_duration = sum(action.detail_durations.values())

            yield action

            method.disable(index, estimated_reward)
        return
