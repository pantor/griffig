import cv2
import numpy as np

from pyaffx import Affine
from _griffig import BoxData, OrthographicImage
from ..utility.image import draw_line, get_inference_image
from ..utility.model_data import ModelArchitecture


class Heatmap:
    def __init__(self, inference, a_space=None):
        self.inference = inference

        if a_space is not None:
            self.inference.a_space = a_space

    def calculate_heat(self, reward, size_cropped, size_result):
        size_reward_center = (reward.shape[1] / 2, reward.shape[2] / 2)
        scale = self.inference.size_area_cropped[0] / self.inference.size_result[0] * ((size_cropped[0] - 30) / reward.shape[1])

        a_space_idx = range(len(self.inference.a_space))

        heat_values = np.zeros(size_result[::-1], dtype=np.float)
        for i in a_space_idx:
            a = self.inference.a_space[i]
            rot_mat = cv2.getRotationMatrix2D(size_reward_center, -a * 180.0 / np.pi, scale)
            rot_mat[0][2] += size_result[0] / 2 - size_reward_center[0]
            rot_mat[1][2] += size_result[1] / 2 - size_reward_center[1]
            heat_values += cv2.warpAffine(reward[i], rot_mat, size_result, borderValue=0)

        norm = (5 * heat_values.max() + len(a_space_idx)) / 6
        return (heat_values / norm * 255.0).astype(np.uint8)

    @staticmethod
    def get_background(image, use_rgb):
        if len(image.mat.shape) >= 3 and image.mat.shape[-1] >= 3:
            if use_rgb:
                back = cv2.cvtColor(image.mat[:, :, :3], cv2.COLOR_RGB2GRAY)
                return cv2.cvtColor(back, cv2.COLOR_GRAY2RGB)

            return cv2.cvtColor(image.mat[:, :, 3:], cv2.COLOR_GRAY2RGB)

        return cv2.cvtColor(image.mat, cv2.COLOR_GRAY2RGB)

    def render(
            self,
            image: OrthographicImage,
            object_image: OrthographicImage = None,
            goal_image: OrthographicImage = None,
            box_data: BoxData = None,
            alpha=0.5,
            use_rgb=True,  # Otherwise depth
            save_path=None,
            reward_index=None,
            draw_lateral=False,
            draw_shifts=False,
            draw_indices=None,
            alpha_human=0.0,
        ):
        input_images = self.inference.get_input_images(image, box_data)

        # if goal_image:
        #     input_images += self.inference.get_input_images(goal_image, box_data)

        if self.inference.model_data.architecture == ModelArchitecture.ActorCritic:
            estimated_reward, actor_result = self.inference.model(input_images)

        elif self.inference.model_data.architecture == ModelArchitecture.PlanarSemantic:
            input_object_images = [get_inference_image(object_image, Affine(a=a), (224, 224), (224, 224), (224, 224), return_mat=True) for a in self.inference.a_space]
            input_object_images = np.array(input_object_images) / np.iinfo(object_image.mat.dtype).max

            estimated_grasp_reward, estimated_object_reward = self.inference.model([input_images, [input_object_images]])
            estimated_reward = estimated_object_reward * estimated_grasp_reward

        else:
            estimated_reward = self.inference.model(input_images)
            actor_result = None

        if reward_index is not None:
            estimated_reward = estimated_reward[reward_index]

        if estimated_reward.shape[-1] > 4:  # self.inference.model_data.output[0] == 'reward+human':
            estimated_reward = (1 - alpha_human) * estimated_reward[:, :, :, :4] + alpha_human * estimated_reward[:, :, :, 4:]

        # reward_reduced = np.maximum(estimated_reward, 0)
        reward_reduced = np.mean(estimated_reward, axis=3)
        # reward_reduced = estimated_reward[:, :, :, 0]

        # For heatmapping the actor
        # reward_reduced = actor_result[:, :, :, 2]
        # reward_reduced = (reward_reduced - np.min(reward_reduced)) / np.ptp(reward_reduced)
        # reward_reduced += 0.5

        size_cropped = input_images[0].shape[1::-1]
        size_result = image.mat.shape[1::-1]

        heat = self.calculate_heat(reward_reduced, size_cropped, size_result)
        heat = cv2.applyColorMap(heat.astype(np.uint8), cv2.COLORMAP_JET)

        background = self.get_background(image, use_rgb)
        if background.dtype == np.uint16:
            background = background.astype(np.float32) / 255
        else:
            background = background.astype(np.float32)
        result = (1 - alpha) * background + alpha * heat
        result = OrthographicImage(result.astype(np.float32), image.pixel_size, image.min_depth, image.max_depth)

        if draw_indices is not None:
            self.draw_indices(result, reward_reduced, draw_indices)

        if draw_lateral:
            for _ in range(10):
                index = np.unravel_index(estimated_reward.argmax(), estimated_reward.shape)
                action = actor_result[index[0], index[1], index[2]]
                self.draw_lateral(result, estimated_reward.shape, index, action)
                estimated_reward[np.unravel_index(estimated_reward.argmax(), estimated_reward.shape)] = 0

        if draw_shifts:
            for _ in range(10):
                self.draw_arrow(result, reward_reduced, np.unravel_index(reward_reduced.argmax(), reward_reduced.shape))
                reward_reduced[np.unravel_index(reward_reduced.argmax(), reward_reduced.shape)] = 0

        if save_path:
            cv2.imwrite(str(save_path), result.mat)
        return result.mat

    def draw_lateral(self, image: OrthographicImage, reward_shape, index, action):
        pose = self.inference.pose_from_index(index, reward_shape, image)
        arrow_color = (255*255, 255*255, 255*255)
        draw_line(image, pose, Affine(0.0, 0.0), Affine(a=pose.a, b=action[1], c=action[2]) * Affine(0.0, 0.0, -0.14), color=arrow_color, thickness=1)

    def draw_indices(self, image: OrthographicImage, reward_shape, indices):
        point_color = (255, 255, 255)

        for index in indices:
            pose = self.inference.pose_from_index(index, reward_shape, image)
            pose.x /= reward_shape[1] / 40
            pose.y /= reward_shape[2] / 40

            draw_line(image, pose, Affine(-0.001, 0), Affine(0.001, 0), color=point_color, thickness=1)
            draw_line(image, pose, Affine(0, -0.001), Affine(0, 0.001), color=point_color, thickness=1)

    def draw_arrow(self, image: OrthographicImage, reward_shape, index):
        pose = self.inference.pose_from_index(index, reward_shape, image)

        arrow_color = (255, 255, 255)
        draw_line(image, pose, Affine(0, 0), Affine(0.036, 0), color=arrow_color, thickness=2)
        draw_line(image, pose, Affine(0.036, 0.0), Affine(0.026, -0.008), color=arrow_color, thickness=2)
        draw_line(image, pose, Affine(0.036, 0.0), Affine(0.026, 0.008), color=arrow_color, thickness=2)
