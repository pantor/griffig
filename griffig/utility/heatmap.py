import cv2
import numpy as np

from actions.action import Affine
from inference.inference import Inference
from inference.inference_planar import InferencePlanar
from inference.inference_actor_critic import InferenceActorCritic
from griffig import OrthographicImage, BoxData
from utils.image import draw_line


class Heatmap:
    def __init__(self, inf, a_space=None, augment_resolution=True):
        self.inf = inf

        if a_space is not None:
            self.inf.a_space = a_space

        self.augment_resolution = augment_resolution

        self.size_area_cropped = 200
        self.size_area_result = 32

    def calculate_heat(self, reward, size_result=(752, 480)):
        size_reward_center = (reward.shape[1] / 2, reward.shape[2] / 2)
        scale = self.size_area_cropped / self.size_area_result * (80.0 / reward.shape[1])

        a_space_idx = range(len(self.inf.a_space))

        heat_values = np.zeros(size_result[::-1], dtype=np.float)
        for i in a_space_idx:
            a = self.inf.a_space[i]
            rot_mat = cv2.getRotationMatrix2D(size_reward_center, -a * 180.0 / np.pi, scale)
            rot_mat[0][2] += size_result[0] / 2 - size_reward_center[0]
            rot_mat[1][2] += size_result[1] / 2 - size_reward_center[1]
            heat_values += cv2.warpAffine(reward[i], rot_mat, size_result, borderValue=0)

        norm = (5 * heat_values.max() + len(a_space_idx)) / 6
        # norm = heat_values.max()

        return heat_values * 255.0 / norm

    def render(
            self,
            image: OrthographicImage,
            goal_image: OrthographicImage = None,
            box_data: BoxData =None,
            alpha=0.5,
            save_path=None,
            reward_index=None,
            draw_directions=False,
            indices=None,
        ):
        base = image.mat
        inputs = self.inf.transform_for_prediction({'rcd': image}, box_data=box_data, augment_resolution=self.augment_resolution)

        if goal_image:
            base = goal_image.mat
            inputs += self.inf.transform_for_prediction({'rcd': goal_image}, box_data=box_data, augment_resolution=self.augment_resolution)

        if isinstance(self.inf, InferenceActorCritic):
            rewards, actor_result = self.inf.model.predict(inputs, batch_size=128)

        else:
            rewards = self.inf.model.predict(inputs, batch_size=128)

        if self.augment_resolution:
            reward = self.inf.split_resolution(rewards)
        else:
            reward = rewards

        if reward_index is not None:
            reward = reward[reward_index]

        # reward = np.maximum(reward, 0)
        reward_mean = np.mean(reward, axis=3)
        # reward_mean = reward[:, :, :, 0]

        heat_values = self.calculate_heat(reward_mean)

        if base.shape[-1] == 4:
            base = base[:, :, 0]

        heatmap = cv2.applyColorMap(heat_values.astype(np.uint8), cv2.COLORMAP_JET)
        base_heatmap = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB) / 255 + alpha * heatmap
        result = OrthographicImage(base_heatmap, image.pixel_size, image.min_depth, image.max_depth)

        if indices is not None:
            self.draw_indices(result, reward, indices)

        if draw_directions:
            for _ in range(10):
                self.draw_shift_arrow(result, reward, np.unravel_index(reward.argmax(), reward.shape))
                reward[np.unravel_index(reward.argmax(), reward.shape)] = 0

        if save_path:
            cv2.imwrite(str(save_path), result.mat)
        return result.mat

    def draw_indices(self, image: OrthographicImage, reward_shape, indices):
        point_color = (255, 255, 255)

        for index in indices:
            pose = self.inf.pose_from_index(index, reward_shape, image)
            pose.x /= reward_shape[1] / 40
            pose.y /= reward_shape[2] / 40

            draw_line(image, pose, Affine(-0.001, 0), Affine(0.001, 0), color=point_color, thickness=1)
            draw_line(image, pose, Affine(0, -0.001), Affine(0, 0.001), color=point_color, thickness=1)

    def draw_shift_arrow(self, image: OrthographicImage, reward_shape, index):
        pose = self.inf.pose_from_index(index, reward_shape, image)

        arrow_color = (255, 255, 255)
        draw_line(image, pose, Affine(0, 0), Affine(0.036, 0), color=arrow_color, thickness=2)
        draw_line(image, pose, Affine(0.036, 0.0), Affine(0.026, -0.008), color=arrow_color, thickness=2)
        draw_line(image, pose, Affine(0.036, 0.0), Affine(0.026, 0.008), color=arrow_color, thickness=2)
