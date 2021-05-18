import time
from typing import Dict, Generator

from loguru import logger
import numpy as np

from actions.action import Action
from griffig import Affine, RobotPose, OrthographicImage, BoxData
from inference.inference_base import InferenceBase
import inference.selection as Selection
from utils.image import draw_around_box, get_area_of_interest, get_inference_image


class InferencePlanarSemantic(InferenceBase):
    def infer(
            self,
            images: Dict[str, OrthographicImage],
            object_images: Dict[str, OrthographicImage],
            method: Selection.Method,
            box_data: BoxData = None,
            verbose=0,
            **kwargs,
        ) -> Generator[Action, None, None]:

        start = time.time()

        if isinstance(method, Selection.Random):
            while True:
                action = Action()
                action.index = self.rs.choice(range(self.number_primitives))
                action.pose = RobotPose(Affine(
                    x=self.rs.uniform(self.lower_random_pose[0], self.upper_random_pose[0]),  # [m]
                    y=self.rs.uniform(self.lower_random_pose[1], self.upper_random_pose[1]),  # [m]
                    a=self.rs.uniform(self.lower_random_pose[3], self.upper_random_pose[3]),  # [rad]
                ))
                action.pose.z = np.nan
                action.estimated_reward = -1
                action.estimated_reward_std = None
                action.method = str(method)
                action.box_data = box_data
                action.step = 0
                yield action
            return

        input_images = self.transform_for_prediction(images, box_data=box_data)

        result_ = []
        for a in self.a_space:
            object_image = get_inference_image(object_images['rcd'], Affine(a=a), (224, 224), (224, 224), (224, 224), return_mat=True)
            result_.append(object_image)

        input_object_images = np.array(result_) / np.iinfo(object_images['rcd'].mat.dtype).max
        estimated_grasp_reward, estimated_object_reward = self.model.predict_on_batch([input_images[0], [input_object_images]])
        estimated_reward = estimated_object_reward * estimated_grasp_reward # np.cbrt(estimated_object_reward * np.power(estimated_grasp_reward, 2))

        # Set some action (indices) to zero
        if self.keep_indixes:
            self.keep_array_at_last_indixes(estimated_reward, self.keep_indixes)

        # Find the index and corresponding action using the selection method
        for _ in range(estimated_reward.size):
            index_raveled = method(estimated_reward)
            index = np.unravel_index(index_raveled, estimated_reward.shape)

            action = Action()
            action.index = index[3]
            action.pose = self.pose_from_index(index, estimated_reward.shape, images[self.main_depth_camera])
            action.pose.z = np.nan
            action.estimated_reward = estimated_reward[index]
            action.estimated_reward_std = None
            action.method = str(method)
            action.box_data = box_data
            action.step = 0  # Default value

            if verbose:
                logger.info(f'NN inference time [s]: {time.time() - start:.3}')

            yield action

            method.disable(index, estimated_reward)
        return

    def infer_at_pose(self, images: Dict[str, OrthographicImage], pose: Affine, box_data: BoxData = None):
        def get_area_images(images, camera):
            if camera == 'rcd' and 'rcd' not in images:
                rc_area = get_area_images(images, 'rc')
                rd_area = get_area_images(images, 'rd')

                return np.concatenate((rc_area, rd_area), axis=2)

            image = images[camera]
            draw_around_box(image, box_data)

            area_mat = get_area_of_interest(image, pose, size_cropped=self.size_area_cropped, size_result=self.size_result).mat
            area_mat = np.array(area_mat, dtype=np.float32) / np.iinfo(area_mat.dtype).max

            if len(area_mat.shape) == 2:
                area_mat = np.expand_dims(area_mat, axis=-1)
            return area_mat

        input_images = [get_area_images(images, camera) for camera in self.camera_list]

        return self.model.predict([np.asarray(input_images)])[0]
