from typing import Union
from pathlib import Path

import numpy as np
from PIL import Image

from pyaffx import Affine
from _griffig import BoxData, Renderer, Gripper, Pointcloud
from .action.checker import Checker
from .infer.inference_actor_critic import InferenceActorCritic
from .infer.inference_planar import InferencePlanar
from .infer.selection import Method, Max, Top
from .utility.model_library import ModelData, ModelLibrary, ModelArchitecture


class Griffig:
    def __init__(
        self,
        model: Union[str, ModelData, Path] ='two-finger-planar',
        gripper: Gripper = None,
        box_data: BoxData = None,
        check_collisions=False,
        verbose = 0,
    ):
        self.gripper = gripper
        self.box_data = box_data

        if isinstance(model, ModelData):
            self.model_data = model
        else:
            self.model_data = ModelLibrary.load_model_data(model)

        if self.model_data.architecture == ModelArchitecture.ActorCritic:
            self.inference = InferenceActorCritic(self.model_data, verbose=verbose)

        elif self.model_data.architecture in [ModelArchitecture.Planar, ModelArchitecture.Lateral]:
            self.inference = InferencePlanar(self.model_data, verbose=verbose)

        else:
            raise Exception(f'Model architecture {self.model_data.architecture} is not yet implemented.')

        self.checker = Checker(box_data, check_collisions)

        image_size = box_data.get_image_size(self.model_data.pixel_size, offset=6) if box_data else (752, 480)
        self.renderer = Renderer(image_size, [0.0, 0.0, 0.0])

        self.last_grasp_successful = True

    def calculate_grasp(self, pointcloud, camera_pose=None, box_data=None, method=None):
        image = self.renderer.render(pointcloud, camera_pose, box_data=box_data)
        return self.calculate_grasp_from_image(image, method=method)

    def calculate_grasp_from_image(self, image, box_data=None, method=None):
        selection_method = method if method else (Max() if self.last_grasp_successful else Top(5))

        action_generator = self.inference.infer(image, selection_method)
        grasp = self.checker.find_grasp(action_generator)

        if self.gripper:
            grasp.pose = grasp.pose * self.gripper.offset

        self.last_grasp_successful = True
        return grasp

    def render(self, pointcloud: Pointcloud, pixel_size, min_depth, max_depth, size=(752, 480), position=[0.0, 0.0, 0.0]):
        img = self.renderer.render_pointcloud_mat(pointcloud, size, pixel_size, min_depth, max_depth, position)
        return Image.fromarray((img[:, :, :3] / 255).astype(np.uint8))

    def calculate_heatmap():
        pass

    def calculate_heatmap_from_image():
        pass

    def report_grasp_failure(self):
        self.last_grasp_successful = False
