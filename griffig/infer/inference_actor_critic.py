from time import time

import numpy as np
from scipy.ndimage import gaussian_filter

from _griffig import BoxData, Grasp, Gripper, OrthographicImage
from ..infer.inference_base import InferenceBase
from ..infer.selection import Method, Max


class InferenceActorCritic(InferenceBase):
    def infer(self, image: OrthographicImage, method: Method, box_data: BoxData = None, gripper: Gripper = None):
        start = time()

        input_images = self.get_input_images(image, box_data)

        pre_duration = time() - start
        start = time()

        rewards_and_actions = self.model(input_images)

        nn_duration = time() - start
        start = time()

        estimated_rewards = np.array(rewards_and_actions[::2])
        action_from_actor = np.array(rewards_and_actions[1::2])

        if self.model_data.output[0] == 'reward+human':
            estimated_rewards = estimated_rewards[:, :, :, :, :4]

        if gripper:
            possible_indices = gripper.consider_indices(self.model_data.gripper_widths)
            for i in range(estimated_rewards.shape[0]):
                self.set_last_dim_to_zero(estimated_rewards[i], np.invert(possible_indices))

        if self.gaussian_sigma:
            for i in range(estimated_rewards.shape[0]):
                for j in range(estimated_rewards.shape[1]):
                    estimated_rewards[i][j] = gaussian_filter(estimated_rewards[i][j], self.gaussian_sigma)

        for _ in range(estimated_rewards.size):
            index_raveled = method(estimated_rewards)
            index = np.unravel_index(index_raveled, estimated_rewards.shape)

            action = Grasp()
            action.index = index[4]
            action.pose = self.pose_from_index(index[1:], estimated_rewards.shape[1:], image)
            action.pose.z, action.pose.b, action.pose.c = action_from_actor[index[0], index[1], index[2], index[3]]
            action.estimated_reward = estimated_rewards[index]
            action.detail_durations = {
                'pre': pre_duration,
                'nn': nn_duration,
                'post': time() - start,
            }
            action.calculation_duration = sum(action.detail_durations.values())

            yield action

            method.disable(index, estimated_rewards)
        return
