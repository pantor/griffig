import cv2
import numpy as np

from _griffig import BoxData


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

    def render(self, image, box_data: BoxData, alpha=0.4, use_rgb=True):
        input_images = self.inference.get_input_images(image, box_data)
        estimated_reward = self.inference.model.predict(input_images, batch_size=128)

        size_cropped = (input_images.shape[2], input_images.shape[1])
        size_result = image.mat.shape[1::-1]

        reward_reduced = np.mean(estimated_reward, axis=3)
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
        return result
