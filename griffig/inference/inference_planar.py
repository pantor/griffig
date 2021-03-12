import time
from typing import Generator

from loguru import logger
import numpy as np

from actions.action import Action
from _griffig import RobotPose, OrthographicImage
from inference.inference import Inference
import inference.selection as Selection
from ..utils.image import get_area_of_interest


class InferencePlanar(Inference):
    def infer(
            self,
            image: OrthographicImage,
            method: Selection.Method,
        ) -> Generator[Action, None, None]:

        start = time.time()

        input_images = self.transform_for_prediction(image)
        estimated_reward = self.model.predict([input_images], batch_size=256)

        # Set some action (indices) to zero
        if self.keep_indixes:
            self.keep_array_at_last_indixes(estimated_reward, self.keep_indixes)

        # Find the index and corresponding action using the selection method
        for _ in range(estimated_reward.size):
            index_raveled = method(estimated_reward)
            index = np.unravel_index(index_raveled, estimated_reward.shape)

            action = Action()
            action.index = index[3]
            action.pose = self.pose_from_index(index, estimated_reward.shape, image)
            action.pose.z = np.nan
            action.estimated_reward = estimated_reward[index]
            action.method = str(method)

            if self.verbose:
                logger.info(f'NN inference time [s]: {time.time() - start:.3}')

            yield action

            method.disable(index, estimated_reward)
        return

    def infer_at_pose(self, image: OrthographicImage, pose: RobotPose):
        def get_area_images(image):
            area_mat = get_area_of_interest(image, pose, size_cropped=self.size_area_cropped, size_result=self.size_result).mat
            area_mat = np.array(area_mat, dtype=np.float32) / np.iinfo(area_mat.dtype).max

            if len(area_mat.shape) == 2:
                area_mat = np.expand_dims(area_mat, axis=-1)
            return area_mat

        input_images = get_area_images(image)
        return self.model.predict([np.asarray([input_images])])[0]
