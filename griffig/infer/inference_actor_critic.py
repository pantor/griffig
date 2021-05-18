from typing import Generator

from loguru import logger
import numpy as np

from _griffig import BoxData, Grasp, OrthographicImage, RobotPose
from ..infer.inference_base import InferenceBase
from ..infer.selection import Method, Max


class InferenceActorCritic(InferenceBase):
    def infer(
            self,
            image: OrthographicImage,
            method: Method = Max(),
            box_data: BoxData = None,
            sigma: float = None,  # Factor for randomize actor result, magnitude of [cm]
            verbose=1,
        ) -> Generator[Grasp, None, None]:

        input_images = self.get_input_images(image, box_data)
        estimated_rewards, actions = self.model(input_images)

        if sigma is not None:
            actions += self.rs.normal([0.0, 0.0, 0.0], [sigma * 0.01, sigma * 0.1, sigma * 0.1], size=actions.shape)

        if gripper:
            possible_indices = gripper.consider_indices(self.model_data.gripper_widths)
            self.set_last_dim_to_zero(estimated_reward, np.invert(possible_indices))

        for _ in range(estimated_rewards.size):
            index_raveled = method(estimated_rewards)
            index = np.unravel_index(index_raveled, estimated_rewards.shape)

            action = Grasp()
            action.index = index[3]
            action.pose = self.pose_from_index(index, estimated_rewards.shape, image)
            action.pose.z, action.pose.b, action.pose.c = actions[index[0], index[1], index[2]]
            action.estimated_reward = estimated_rewards[index]
            action.estimated_reward_std = None
            action.method = str(method)
            action.sigma = sigma
            action.step = 0  # Default value

            yield action

            method.disable(index, estimated_rewards)
        return
