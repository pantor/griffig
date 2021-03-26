import cv2
import numpy as np

from ..infer.inference_planar import InferencePlanar
from ..infer.inference_actor_critic import InferenceActorCritic
from _griffig import BoxData, OrthographicImage
from ..utility.image import draw_line


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

    def render(
            self,
            image: OrthographicImage,
            goal_image: OrthographicImage = None,
            box_data: BoxData = None,
            alpha=0.4,
            use_rgb=True,
            save_path=None,
            reward_index=None,
            draw_directions=False,
            indices=None,
        ):
        # inputs = self.inference.transform_for_prediction({'rcd': image}, box_data=box_data)
        input_images = self.inference.get_input_images(image, box_data)

        if goal_image:
            # inputs += self.inference.transform_for_prediction({'rcd': goal_image}, box_data=box_data)
            input_images += self.inference.get_input_images(goal_image, box_data)

        if isinstance(self.inference, InferenceActorCritic):
            estimated_reward, actor_result = self.inference.model.predict(inputs, batch_size=128)

        else:
            estimated_reward = self.inference.model.predict(inputs, batch_size=128)

        if reward_index is not None:
            estimated_reward = estimated_reward[reward_index]

        # reward_reduced = np.maximum(estimated_reward, 0)
        reward_reduced = np.mean(estimated_reward, axis=3)
        # reward_reduced = estimated_reward[:, :, :, 0]

        size_cropped = (input_images.shape[2], input_images.shape[1])
        size_result = image.mat.shape[1::-1]

        heat = self.calculate_heat(reward_reduced, size_cropped, size_result)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

        if len(image.mat.shape) >= 3 and image.mat.shape[-1] >= 3:
            if use_rgb:
                back = cv2.cvtColor(image.mat[:, :, :3], cv2.COLOR_RGB2GRAY)
                back = cv2.cvtColor(back, cv2.COLOR_GRAY2RGB)
            else:
                back = cv2.cvtColor(image.mat[:, :, 3:], cv2.COLOR_GRAY2RGB)
        else:
            back = cv2.cvtColor(image.mat, cv2.COLOR_GRAY2RGB)
        
        result = (1 - alpha) * back + alpha * heat
        result = OrthographicImage(result, image.pixel_size, image.min_depth, image.max_depth)

        if indices is not None:
            self.draw_indices(result, reward, indices)

        if draw_directions:
            for _ in range(10):
                self.draw_shift_arrow(result, reward, np.unravel_index(reward.argmax(), reward.shape))
                reward[np.unravel_index(reward.argmax(), reward.shape)] = 0

        if save_path:
            cv2.imwrite(str(save_path), result.mat)
        return result.mat

    def render2(self, image, box_data: BoxData, alpha=0.4, use_rgb=True, reward_index=None, return_mat=True):
        input_images = self.inference.get_input_images(image, box_data)

        if isinstance(self.inference, InferenceActorCritic):
            estimated_reward, _ = self.inference.model.predict(input_images, batch_size=128)

        else:
            estimated_reward = self.inference.model.predict(input_images, batch_size=128)

        if reward_index is not None:
            estimated_reward = estimated_reward[reward_index]

        # reward_reduced = np.max(estimated_reward, axis=3)
        reward_reduced = np.mean(estimated_reward, axis=3)
        # reward_reduced = estimated_reward[:, :, :, 0]

        size_cropped = (input_images.shape[2], input_images.shape[1])
        size_result = image.mat.shape[1::-1]

        heat = self.calculate_heat(reward_reduced, size_cropped, size_result)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

        if len(image.mat.shape) >= 3 and image.mat.shape[-1] >= 3:
            if use_rgb:
                back = cv2.cvtColor(image.mat[:, :, :3], cv2.COLOR_RGB2GRAY)
                back = cv2.cvtColor(back, cv2.COLOR_GRAY2RGB)
            else:
                back = cv2.cvtColor(image.mat[:, :, 3:], cv2.COLOR_GRAY2RGB)
        else:
            back = cv2.cvtColor(image.mat, cv2.COLOR_GRAY2RGB)
        
        result = (1 - alpha) * back + alpha * heat
        if return_mat:
            return result

        return result

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
