from os import environ, path
import json

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import tensorflow.keras as tk

from pyaffx import Affine
from ..grasp.grasp import Grasp
from _griffig import BoxData, Gripper, OrthographicImage
from ..utility.image2 import draw_around_box, get_box_projection, get_inference_image


class Inference:
    def __init__(self, model_data, converter, gaussian_sigma=None, seed=None, debug=False):
        self.model = self._load_model(model_data['name'], 'grasp')
        self.debug = debug

        self.size_area_cropped = model_data['size_area_cropped']
        self.size_result = model_data['size_result']
        self.scale_factors = (self.size_area_cropped[0] / self.size_result[0], self.size_area_cropped[1] / self.size_result[1])

        self.a_space = np.linspace(-np.pi/2 + 0.05, np.pi/2 - 0.05, 20)  # [rad] # Don't use a=0.0 -> even number
        self.keep_indixes = None
        self.gaussian_sigma = gaussian_sigma

        self.converter = converter

    @classmethod
    def load_model_data(cls, name: str):
        base_path = path.join(path.dirname(path.realpath(__file__)), '../../../models/{}')
        with open(base_path.format(name) + '.json', 'r') as read_file:
            model_data = json.load(read_file)
        return model_data

    def _load_model(self, name: str, submodel=None):
        base_path = path.join(path.dirname(path.realpath(__file__)), '../../../models/{}')
        model = tk.models.load_model(base_path.format(name) + '.tf', compile=False)

        if submodel:
            model = model.get_layer(submodel)
        return model

    def _get_size_cropped(self, image, box_data: BoxData):
        box_projection = get_box_projection(image, box_data)
        center = np.array([image.mat.shape[1], image.mat.shape[0]]) / 2
        farthest_corner = np.max(np.linalg.norm(box_projection - center, axis=1))
        side_length = int(np.ceil(2 * farthest_corner * self.size_result[0] / self.size_area_cropped[0]))
        return (side_length, side_length)

    def get_input_images(self, orig_image, box_data: BoxData):
        image = orig_image.clone()
        size_cropped = self._get_size_cropped(orig_image, box_data)

        if box_data:
            draw_around_box(image, box_data)

        result_ = []

        for a in self.a_space:
            result_.append(
                get_inference_image(image, Affine(a=a), size_cropped, self.size_area_cropped, self.size_result, return_mat=True)
            )

        if self.debug:
            cv2.imwrite('src/learned_grasping/tmp/inf.png', result_[10][:, :, 3])

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

    @classmethod
    def set_last_dim_to_zero(cls, array, indixes):
        array[:, :, :, indixes] = 0

    def infer(self, method, image, box_data: BoxData = None, gripper: Gripper = None):
        input_images = self.get_input_images(image, box_data)
        estimated_reward = self.model.predict(input_images, batch_size=128)

        if self.gaussian_sigma:
            for i in range(estimated_reward.shape[0]):
                estimated_reward[i] = gaussian_filter(estimated_reward[i], self.gaussian_sigma)

        if gripper:
            possible_indices = self.converter.consider_indices(gripper)
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
