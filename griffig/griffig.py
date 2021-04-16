from typing import Union
from pathlib import Path

import numpy as np
from PIL import Image

from _griffig import BoxData, Gripper, Pointcloud, Renderer, RobotPose
from .action.checker import Checker
from .action.converter import Converter
from .infer.inference import Inference
from .infer.selection import Method, Max, Top
from .utility.heatmap import Heatmap
from .utility.image import draw_pose
from .utility.model_library import ModelData, ModelLibrary, ModelArchitecture


class Griffig:
    def __init__(
        self,
        model: Union[str, ModelData, Path] ='two-finger-planar',
        gripper: Gripper = None,
        box_data: BoxData = None,
        avoid_collisions=False,
        gpu: int = None,
        verbose = 0,
    ):
        self.gripper = gripper
        self.box_data = box_data

        self.model_data = model if isinstance(model, ModelData) else ModelLibrary.load_model_data(model)
        self.inference = Inference.create(self.model_data, gpu=gpu, verbose=verbose)

        self.converter = Converter(self.model_data.gripper_widths)
        self.checker = Checker(self.converter, avoid_collisions=avoid_collisions)

        image_size = box_data.get_image_size(self.model_data.pixel_size, offset=6) if box_data else (752, 480)
        self.renderer = Renderer(image_size, [0.0, 0.0, 0.0])

        self.last_grasp_successful = True

    def calculate_grasp(self, pointcloud: Pointcloud, camera_pose=None, box_data=None, gripper=None, method=None):
        image = self.renderer.render(pointcloud, camera_pose, box_data=box_data)
        return self.calculate_grasp_from_image(image, box_data=box_data, gripper=gripper, method=method)

    def calculate_grasp_from_image(self, image, box_data=None, gripper=None, method=None):
        box_data = box_data if box_data else self.box_data
        gripper = gripper if gripper else self.gripper
        selection_method = method if method else (Max() if self.last_grasp_successful else Top(5))

        action_generator = self.inference.infer(image, selection_method, box_data=box_data)
        grasp = self.checker.find_grasp(action_generator, image, box_data=box_data, gripper=self.gripper)
        self.last_grasp_successful = True
        return grasp

    def render(self, pointcloud: Pointcloud, pixel_size=None, min_depth=None, max_depth=None, size=(752, 480), position=[0.0, 0.0, 0.0]):
        pixel_size = pixel_size if pixel_size is not None else self.model_data.pixel_size
        min_depth = min_depth if min_depth is not None else self.model_data.min_depth
        max_depth = max_depth if max_depth is not None else self.model_data.max_depth

        img = self.renderer.render_pointcloud_mat(pointcloud, size, pixel_size, min_depth, max_depth, position)
        return Image.fromarray((img[:, :, 2::-1] / 255).astype(np.uint8))

    def calculate_heatmap(self, pointcloud: Pointcloud, box_data: BoxData = None, a_space=[0.0]):
        pixel_size = self.model_data.pixel_size
        min_depth = self.model_data.min_depth
        max_depth = self.model_data.max_depth

        img = self.renderer.render_pointcloud_mat(pointcloud, size, pixel_size, min_depth, max_depth, position)
        return self.calculate_heatmap_from_image(img, box_data, a_space)

    def calculate_heatmap_from_image(self, image, box_data: BoxData = None, a_space=[0.0]):
        heatmapper = Heatmap(self.inference, a_space=a_space)
        heatmap = heatmapper.render(image, box_data=box_data)
        return Image.fromarray((heatmap[:, :, 2::-1]).astype(np.uint8))

    def draw_grasp_on_image(self, image, grasp):
        draw_pose(image, RobotPose(grasp.pose, d=grasp.stroke))
        return Image.fromarray((image.mat[:, :, 2::-1] / 255).astype(np.uint8))

    def report_grasp_failure(self):
        self.last_grasp_successful = False
