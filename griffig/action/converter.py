import cv2
import numpy as np

from ..utility.image import crop


class Converter:
    def __init__(self, gripper_strokes, z_offset=0.0):
        self.gripper_strokes = gripper_strokes
        self.z_offset = z_offset

        self.gripper_width = 0.016  # [m]
        self.gripper_height = 0.004  # [m]

        self.max_lateral_angle = 0.5  # [rad]
        self.image_area_size = 0.12  # [m]

    def __len__(self):
        return len(self.gripper_strokes)

    def index_to_action(self, grasp):
        grasp.stroke = self.gripper_strokes[grasp.index]

    def consider_indices(self, gripper):
        arr = np.array(self.gripper_strokes)
        return (gripper.min_stroke <= arr) & (arr <= gripper.max_stroke)

    def calculate_z(self, image_area, grasp):
        """Model-based calculation of the grasp distance z"""

        mat_area_image = image_area.mat[:, :, 3]
        mat_area_image = mat_area_image.astype(np.float32) / np.iinfo(image_area.mat.dtype).max
        mat_area_image[mat_area_image < 0.02] = np.NaN  # Make every not found pixel NaN

        # Get distance at gripper for possible collisions
        area_center_default_size = 0.012  # [m]
        area_center_default_size_px = image_area.pixel_size * area_center_default_size

        gripper_one_side_size = 0.5 * image_area.pixel_size * (grasp.stroke + 0.002)  # [px]
        area_center = crop(mat_area_image, (area_center_default_size_px, area_center_default_size_px))
        side_gripper_image_size = (image_area.pixel_size * 0.025, image_area.pixel_size * 0.025)
        area_left = crop(mat_area_image, side_gripper_image_size, (-gripper_one_side_size, 0))
        area_right = crop(mat_area_image, side_gripper_image_size, (gripper_one_side_size, 0))

        z_raw = image_area.depth_from_value(np.nanmedian(area_center) * 255 * 255)
        if z_raw is np.NaN:
            area_center = crop(mat_area_image, (image_area.pixel_size * 0.03, image_area.pixel_size * 0.03))
            z_raw = image_area.depth_from_value(np.nanmedian(area_center) * 255 * 255)

            if z_raw is np.NaN:
                print('[converter] z is NaN!')

        analyze_collision = True and area_left.size > 0 and area_right.size > 0

        if analyze_collision:
            z_raw_left = image_area.depth_from_value(np.nanmin(area_left) * 255 * 255)
            z_raw_right = image_area.depth_from_value(np.nanmin(area_right) * 255 * 255)
            z_raw_collision = min(z_raw_left, z_raw_right) - 0.03  # [m]
        else:
            z_raw_collision = np.Inf

        grasp.pose.z = min(z_raw, z_raw_collision) + self.z_offset # Get the maximum [m] for impedance mode

    def calculate_b(self, image_area, grasp):
        b_lateral_default_size_px_height = int(image_area.pixel_size * self.image_area_size)

        b_gripper_height = int(image_area.pixel_size * self.gripper_width)
        b_gripper_width = int(image_area.pixel_size * self.gripper_height)

        b_lateral_default_size_px_width = image_area.pixel_size * (action.pose.d + self.gripper_height)  # [px]
        b_lateral_area = crop(image_area, (b_lateral_default_size_px_height, b_lateral_default_size_px_width)) / 255

        b_lateral_area_left = b_lateral_area[:, :2*b_gripper_width] / 255
        b_lateral_area_right = b_lateral_area[:, -2*b_gripper_width:] / 255

        values_left = np.gradient(b_lateral_area_left, axis=0)
        values_right = np.gradient(b_lateral_area_right, axis=0)

        weights_left = np.ones_like(values_left)
        weights_left[np.isnan(values_left)] = np.nan

        weights_right = np.ones_like(values_right)
        weights_right[np.isnan(values_right)] = np.nan

        a = np.linspace(-self.image_area_size / 2, self.image_area_size / 2, b_lateral_area.shape[0])
        sigma = 1.3 * self.gripper_width
        s = 0.042 * np.exp(-0.5 * np.power(a / sigma, 2.)) / (np.sqrt(2 * np.pi) * sigma)
        s = np.expand_dims(s, -1)
        weights_left *= s
        weights_right *= s

        weighted_gradient = np.nansum(weights_left * values_left) / weights_left.size + np.nansum(weights_right * values_right) / weights_right.size

        new_b = np.arctan2((image_area.max_depth - image_area.min_depth) * 0.5 * weighted_gradient, 1. / image_area.pixel_size)
        grasp.pose.b = np.clip(new_b, -self.max_lateral_angle, self.max_lateral_angle)

    def calculate_c(self, image_area, grasp):
        c_lateral_default_size_px_height = int(image_area.pixel_size * 0.016 / 2)
        c_lateral_default_size_px_width = image_area.pixel_size * (action.pose.d + 0.004)  # [px]
        c_lateral_area = crop(image_area, (c_lateral_default_size_px_height, c_lateral_default_size_px_width)) / 255

        mask = np.isnan(c_lateral_area).astype(np.uint8)

        # Make distance constant 2cm below the grasp point
        min_value_for_gradient = image_area.value_from_depth(-action.pose.z + 0.04) / (255 * 255)
        lateral_area_pre_inpaint = c_lateral_area.astype(np.uint8)
        lateral_area_pre_inpaint[lateral_area_pre_inpaint < 255 * min_value_for_gradient] = 255 * min_value_for_gradient
        c_lateral_area_inpaint = cv2.inpaint(lateral_area_pre_inpaint, mask, 3, cv2.INPAINT_TELEA) / 255

        gradient = np.mean(np.diff(c_lateral_area_inpaint, axis=1), axis=0)

        left_gradient_max = np.max(gradient)
        right_gradient_max = np.min(gradient)

        gradient_norm = (image_area.max_depth - image_area.min_depth)
        left_gamma_1 = np.pi / 2 - np.arctan2(gradient_norm * left_gradient_max, 1. / image_area.pixel_size)
        right_gamma_1 = -np.pi / 2 - np.arctan2(gradient_norm * right_gradient_max, 1. / image_area.pixel_size)

        gamma_mean = np.arctan2(gradient_norm * np.mean(gradient), 1. / image_area.pixel_size)

        certainty_parameter = 0.0
        eps = 0.03
        if left_gamma_1 < eps and right_gamma_1 > -eps:
            new_c = (left_gamma_1 + right_gamma_1 + 2 * gamma_mean) / 4
        elif left_gamma_1 >= eps and right_gamma_1 <= -eps:
            new_c = gamma_mean
        elif left_gamma_1 <= eps:
            new_c = ((1 - certainty_parameter) * right_gamma_1 + gamma_mean) / 2
        else:
            new_c = ((1 - certainty_parameter) * -left_gamma_1 + gamma_mean) / 2

        grasp.pose.c = np.clip(new_c, -self.max_lateral_angle, self.max_lateral_angle)
