from typing import Union
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from _griffig import BoxData, Gripper, Pointcloud, Renderer, RobotPose
from .action.checker import Checker
from .action.converter import Converter
from .infer.inference import Inference
from .infer.selection import Method, Max, Top
from .utility.heatmap import Heatmap
from .utility.image import draw_around_box, draw_pose
from .utility.model_library import ModelData, ModelLibrary, ModelArchitecture


class Griffig:
    def __init__(
        self,
        model: Union[str, ModelData, Path] ='two-finger-planar',
        gripper: Gripper = None,
        box_data: BoxData = None,
        typical_camera_distance: int = None,
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

        self.typical_camera_distance = typical_camera_distance if typical_camera_distance is not None else 0.5

        if box_data:
            self.renderer = Renderer(box_data, self.typical_camera_distance, self.model_data.pixel_size, self.model_data.depth_diff)
        else:
            self.renderer = Renderer((752, 480), self.typical_camera_distance, self.model_data.pixel_size, self.model_data.depth_diff)

        self.last_grasp_successful = True

    def render(self, pointcloud: Pointcloud, pixel_size=None, min_depth=None, max_depth=None, position=[0.0, 0.0, 0.0]):
        pixel_size = pixel_size if pixel_size is not None else self.model_data.pixel_size
        min_depth = min_depth if min_depth is not None else self.typical_camera_distance - self.model_data.depth_diff
        max_depth = max_depth if max_depth is not None else self.typical_camera_distance

        img = self.renderer.render_pointcloud_mat(pointcloud, pixel_size, min_depth, max_depth, position)
        return self.convert_to_pillow_image(img)

    def calculate_grasp(self, pointcloud: Pointcloud, camera_pose=None, box_data=None, gripper=None, method=None, return_image=False, channels='RGBD'):
        image = self.renderer.render_pointcloud(pointcloud)
        grasp = self.calculate_grasp_from_image(image, box_data=box_data, gripper=gripper, method=method)

        if return_image:
            image = image.clone()
            return grasp, self.draw_grasp_on_image(image, grasp, channels=channels)
        return grasp

    def calculate_grasp_from_image(self, image, box_data=None, gripper=None, method=None):
        box_data = box_data if box_data else self.box_data
        gripper = gripper if gripper else self.gripper
        selection_method = method if method else (Max() if self.last_grasp_successful else Top(5))

        action_generator = self.inference.infer(image, selection_method, box_data=box_data)
        grasp = self.checker.find_grasp(action_generator, image, box_data=box_data, gripper=self.gripper)
        self.last_grasp_successful = True
        return grasp

    def calculate_heatmap(self, pointcloud: Pointcloud, box_data: BoxData = None, a_space=None):
        pixel_size = self.model_data.pixel_size
        min_depth = self.typical_camera_distance - self.model_data.depth_diff
        max_depth = self.typical_camera_distance
        position = [0.0, 0.0, 0.0]

        img = self.renderer.render_pointcloud_mat(pointcloud, pixel_size, min_depth, max_depth, position)
        return self.calculate_heatmap_from_image(img, box_data, a_space)

    def calculate_heatmap_from_image(self, image, box_data: BoxData = None, a_space=None):
        a_space = a_space if a_space is not None else [0.0]
        heatmapper = Heatmap(self.inference, a_space=a_space)
        heatmap = heatmapper.render(image, box_data=box_data)
        return Image.fromarray((heatmap[:, :, 2::-1]).astype(np.uint8))

    def report_grasp_failure(self):
        self.last_grasp_successful = False

    @classmethod
    def convert_to_pillow_image(cls, image, channels='RGBD'):
        mat = cv2.convertScaleAbs(cv2.cvtColor(image.mat, cv2.COLOR_BGRA2RGBA), alpha=(255.0/65535.0)).astype(np.uint8)
        pillow_image = Image.fromarray(mat, 'RGBA')
        if channels == 'RGB':
            return pillow_image.convert('RGB')
        elif channels == 'D':
            return pillow_image.getchannel('A')
        return pillow_image

    @classmethod
    def draw_box_on_image(cls, image, box_data, draw_lines=False, channels='RGBD'):
        draw_around_box(image, box_data, draw_lines)
        return cls.convert_to_pillow_image(image, channels)

    @classmethod
    def draw_grasp_on_image(cls, image, grasp, channels='RGBD', convert_to_rgb=True):
        if channels == 'D' and convert_to_rgb:
            image.mat = cv2.cvtColor(image.mat[:, :, 3], cv2.COLOR_GRAY2RGB)
            channels = 'RGB'

        draw_pose(image, RobotPose(grasp.pose, d=grasp.stroke))
        return cls.convert_to_pillow_image(image, channels)
