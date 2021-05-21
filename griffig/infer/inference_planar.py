from time import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from _griffig import BoxData, Grasp, Gripper, OrthographicImage
from ..infer.inference_base import InferenceBase
from ..infer.selection import Method, Max
from ..utility.image import draw_around_box2, get_box_projection, get_inference_image


class InferencePlanar(InferenceBase):
    def infer(self, image: OrthographicImage, method: Method, box_data: BoxData = None, gripper: Gripper = None):
        start = time()
        input_images = self.get_input_images(image, box_data)

        pre_duration = time() - start
        start = time()

        estimated_reward = self.model(input_images)
        
        nn_duration = time() - start
        start = time()

        if self.gaussian_sigma:
            for i in range(estimated_reward.shape[0]):
                estimated_reward[i] = gaussian_filter(estimated_reward[i], self.gaussian_sigma)

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
