from typing import Union

import cv2
import numpy as np
from PIL import Image

from pyaffx import Affine
from _griffig import BoxData, Renderer, Gripper, Pointcloud, OrthographicData
from .action.checker import Checker
from .infer.inference_actor_critic import InferenceActorCritic
from .infer.inference_planar import InferencePlanar
from .infer.selection import Method, Max, Top
from .utility.model_library import ModelData, ModelLibrary, ModelArchitecture


class Griffig:
    def __init__(
        self,
        model: Union[str, ModelData] ='two-finger',
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
        self.renderer = Renderer(box_data.get_image_size(self.model_data.pixel_size, offset=6), [0.0, 0.0, 0.0])

        self.last_grasp_successful = True

    def calculate_grasp(self, camera_pose, pointcloud, box_data=None, method=None):
        image = self.renderer.render(pointcloud, camera_pose, box_data=box_data)
        selection_method = method if method else (Max() if self.last_grasp_successful else Top(5))

        action_generator = self.inference.infer(image, selection_method)
        grasp = self.checker.find_grasp(action_generator)

        if self.gripper:
            grasp.pose = grasp.pose * self.gripper.robot_to_tip

        self.last_grasp_successful = True
        return grasp

    def render(self, pointcloud: Pointcloud, pixel_size, min_depth, max_depth, size=(752, 480), position=[0.0, 0.0, 0.0]):
        img = self.renderer.draw_pointcloud(pointcloud, size, OrthographicData(pixel_size, min_depth, max_depth), position)
        return Image.fromarray((img[:, :, :3] / 255).astype(np.uint8))

    def report_grasp_failure(self):
        self.last_grasp_successful = False
