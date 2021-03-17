import time
from typing import Generator

from loguru import logger
import numpy as np

from ..grasp.grasp import Grasp
from _griffig import RobotPose, OrthographicImage, BoxData
from ..infer.inference import Inference
from ..infer.selection import Method, Max


class InferenceActorCritic(Inference):
    def infer(
            self,
            image: OrthographicImage,
            method: Method = Max(),
            sigma: float = None,  # Factor for randomize actor result, magnitude of [cm]
            verbose=1,
        ) -> Generator[Grasp, None, None]:

        start = time.time()

        input_images = self.transform_for_prediction(image)
        estimated_rewards, actions = self.model.predict(input_images, batch_size=256)

        if sigma is not None:
            actions += self.rs.normal([0.0, 0.0, 0.0], [sigma * 0.01, sigma * 0.1, sigma * 0.1], size=actions.shape)

        # Set some action (indices) to zero
        if self.keep_indixes:
            self.keep_array_at_last_indixes(estimated_rewards, self.keep_indixes)

        # Find the index and corresponding action using the selection method
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

            if verbose:
                logger.info(f'NN inference time [s]: {time.time() - start:.3}, Sigma: {sigma}')

            yield action

            method.disable(index, estimated_rewards)
        return
