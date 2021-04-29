from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from pyaffx import Affine
from _griffig import BoxData, Grasp, Gripper, OrthographicImage
from ..infer.inference_base import InferenceBase
from ..infer.selection import Method, Max
from ..utility.image import draw_around_box2, get_box_projection, get_inference_image


class InferencePlanar(InferenceBase):
    def get_input_images(self, orig_image, box_data: BoxData):
        image = orig_image.clone()
        size_cropped = self._get_size_cropped(orig_image, box_data)

        if box_data:
            draw_around_box2(image, box_data)

        result_ = []

        for a in self.a_space:
            result_.append(
                get_inference_image(image, Affine(a=a), size_cropped, self.size_area_cropped, self.size_result, return_mat=True)
            )

        if self.verbose:
            cv2.imwrite('/tmp/inf.png', result_[10][:, :, 3])

        result = np.array(result_, dtype=np.float32) / np.iinfo(image.mat.dtype).max
        if len(result.shape) == 3:
            result = np.expand_dims(result, axis=-1)

        return result

    def pose_from_index(self, index, index_shape, image: OrthographicImage, resolution_factor=2.0):
        return Affine(
            x=resolution_factor * self.scale_factors[0] * image.position_from_index(index[1], index_shape[1]),
            y=resolution_factor * self.scale_factors[1] * image.position_from_index(index[2], index_shape[2]),
            a=self.a_space[index[0]],
        ).inverse()

    def infer(self, image, method, box_data: BoxData = None, gripper: Gripper = None):
        input_images = self.get_input_images(image, box_data)
        estimated_reward = self.model.predict(input_images, batch_size=128)

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

            yield action

            method.disable(index, estimated_reward)
        return
