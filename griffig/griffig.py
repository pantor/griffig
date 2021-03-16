from typing import Union

from frankx import Affine
from _griffig import BoxData, Renderer, Gripper, Pointcloud
from griffig.grasp.checker import Checker
from griffig.inference.inference_actor_critic import InferenceActorCritic
import griffig.inference.selection as Selection
from griffig.utils.model_library import ModelData, ModelLibrary, ModelArchitecture


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
            self.model_data = ModelLibrary.get_model_or_throw(model)

        if self.model_data.architecture ==  ModelArchitecture.ActorCritic:
            self.inference = InferenceActorCritic(self.model_data, verbose=verbose)
        else:
            raise Exception(f'Model architecture {self.model_data.architecture} is not yet implemented.')

        self.checker = Checker(box_data, check_collisions)

        self.renderer = Renderer(self.model_data.pixel_size, self.model_data.depth_diff)
        self.last_grasp_successful = True

    def calculate_grasp(self, camera_pose, pointcloud, box_data=None, method=None):
        image = self.renderer.render(pointcloud, camera_pose, box_data=box_data)
        selection_method = Selection.Max() if self.last_grasp_successful else Selection.Top(5)

        action_iterator = self.inference.infer(image, selection_method)
        grasp = self.checker.find_grasp(action_iterator)

        self.last_grasp_successful = True

        return grasp

    def render(self, pointcloud: Pointcloud):
        return self.renderer.draw_pointcloud(pointcloud)

    def report_grasp_failure(self):
        self.last_grasp_successful = False
