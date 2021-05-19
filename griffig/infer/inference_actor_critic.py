from time import time

import numpy as np

from _griffig import BoxData, Grasp, Gripper, OrthographicImage
from ..infer.inference_base import InferenceBase
from ..infer.selection import Method, Max


class InferenceActorCritic(InferenceBase):
    def infer(self, image: OrthographicImage, method: Method = Max(), box_data: BoxData = None, gripper: Gripper = None):
        start = time()

        input_images = self.get_input_images(image, box_data)
        rewards_and_actions = self.model(input_images)

        estimated_rewards = np.array(rewards_and_actions[::2])
        action_from_actor = np.array(rewards_and_actions[1::2])

        if self.model_data.output[0] == 'reward+human':
            estimated_rewards = estimated_rewards[:, :, :, :, :4]

        if gripper:
            possible_indices = gripper.consider_indices(self.model_data.gripper_widths)
            # for estimated_reward in estimated_rewards:
            #     self.set_last_dim_to_zero(estimated_reward, np.invert(possible_indices))

        for _ in range(estimated_rewards.size):
            index_raveled = method(estimated_rewards)
            index = np.unravel_index(index_raveled, estimated_rewards.shape)

            action = Grasp()
            action.index = index[4]
            action.pose = self.pose_from_index(index[1:], estimated_rewards.shape[1:], image)
            action.pose.z, action.pose.b, action.pose.c = action_from_actor[index[0], index[1], index[2], index[3]]
            action.estimated_reward = estimated_rewards[index]
            action.calculation_duration = time() - start

            yield action

            method.disable(index, estimated_rewards)
        return
