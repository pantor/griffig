from typing import Any, List, Sequence, Tuple, Optional, Union

import cv2
import numpy as np

from _griffig import BoxData, Gripper, RobotPose, OrthographicImage
from pyaffx import Affine


def get_color_scale(dtype):
    if dtype == np.float32:
        return 1. / 255
    if dtype == np.uint16:
        return 255
    return 1


def crop(mat_image: Any, size_output: Sequence[float], vec=(0, 0)) -> Any:
    margin_x_lower = int(round((mat_image.shape[0] - size_output[0]) / 2 + vec[1]))
    margin_y_lower = int(round((mat_image.shape[1] - size_output[1]) / 2 + vec[0]))
    margin_x_upper = margin_x_lower + int(round(size_output[0]))
    margin_y_upper = margin_y_lower + int(round(size_output[1]))
    return mat_image[margin_x_lower:margin_x_upper, margin_y_lower:margin_y_upper]


def get_transformation(x: float, y: float, a: float, center: Sequence[float], scale=1.0, cropped: Sequence[float] = None):  # [rad]
    trans = cv2.getRotationMatrix2D((round(center[0] - x), round(center[1] - y)), a * 180.0 / np.pi, scale)  # [deg]
    trans[0][2] += x + scale * cropped[0] / 2 - center[0] if cropped else x
    trans[1][2] += y + scale * cropped[1] / 2 - center[1] if cropped else y
    return trans


def get_area_of_interest(
        image: OrthographicImage,
        pose: Affine,
        size_cropped: Tuple[float, float] = None,
        size_result: Tuple[float, float] = None,
        flags=cv2.INTER_LINEAR,
        return_mat=False,
        planar=True,
) -> Union[OrthographicImage, Any]:
    size_input = (image.mat.shape[1], image.mat.shape[0])
    center_image = (image.mat.shape[1] / 2, image.mat.shape[0] / 2)

    if size_result and size_cropped:
        scale = size_result[0] / size_cropped[0]
        assert scale == (size_result[1] / size_cropped[1])
    elif size_result:
        scale = size_result[0] / size_input[0]
        assert scale == (size_result[1] / size_input[1])
    else:
        scale = 1.0

    size_final = size_result or size_cropped or size_input

    if not planar:
        pose = image.pose.inverse() * pose

    trans = get_transformation(
        image.pixel_size * pose.y,
        image.pixel_size * pose.x,
        -pose.a,
        center_image,
        scale=scale,
        cropped=size_cropped,
    )
    mat_result = cv2.warpAffine(image.mat, trans, size_final, flags=flags, borderMode=cv2.BORDER_REPLICATE)  # INTERPOLATION_METHOD
    if return_mat:
        return mat_result

    image_pose = Affine(x=pose.x, y=pose.y, a=-pose.a) * image.pose

    return OrthographicImage(
        mat_result,
        scale * image.pixel_size,
        image.min_depth,
        image.max_depth,
        image.camera,
        image_pose,
    )


def get_inference_image(
        image: OrthographicImage,
        pose: Affine,
        size_cropped: Tuple[float, float],
        size_area_cropped: Tuple[float, float],
        size_area_result: Tuple[float, float],
        return_mat=False,
    ) -> OrthographicImage:
    size_input = (image.mat.shape[1], image.mat.shape[0])
    center_image = (size_input[0] / 2, size_input[1] / 2)

    scale = size_area_result[0] / size_area_cropped[0]

    trans = get_transformation(
        image.pixel_size * pose.y,
        image.pixel_size * pose.x,
        pose.a,
        center_image,
        scale=scale,
        cropped=(size_cropped[0] / scale, size_cropped[1] / scale),
    )
    mat_result = cv2.warpAffine(image.mat, trans, size_cropped, borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
    if return_mat:
        return mat_result

    image_pose = Affine(x=pose.x, y=pose.y, a=pose.a) * image.pose

    return OrthographicImage(
        mat_result,
        scale * image.pixel_size,
        image.min_depth,
        image.max_depth,
        image.camera,
        image_pose,
    )


def _get_rect_contour(center: Sequence[float], size: Sequence[float]) -> List[Sequence[float]]:
    return [
        [center[0] + size[0] / 2, center[1] + size[1] / 2, size[2]],
        [center[0] + size[0] / 2, center[1] - size[1] / 2, size[2]],
        [center[0] - size[0] / 2, center[1] - size[1] / 2, size[2]],
        [center[0] - size[0] / 2, center[1] + size[1] / 2, size[2]],
    ]


def get_box_projection(image: OrthographicImage, box_data: BoxData):
    if not box_data:
        return

    box_border = [Affine(*p) for p in box_data.contour]
    return [image.project(p) for p in box_border]


def draw_line(
        image: OrthographicImage,
        action_pose: Affine,
        pt1: Affine,
        pt2: Affine,
        color,
        thickness=1,
    ) -> None:
    cm = get_color_scale(image.mat.dtype)
    pose = image.pose.inverse() * action_pose
    pt1_projection = image.project(pose * pt1)
    pt2_projection = image.project(pose * pt2)
    cv2.line(image.mat, tuple(pt1_projection), tuple(pt2_projection), (color[0] * cm, color[1] * cm, color[2] * cm), thickness, lineType=cv2.LINE_AA)


def draw_polygon(
        image: OrthographicImage,
        action_pose: Affine,
        polygon,
        color,
        thickness=1,
    ) -> None:
    cm = get_color_scale(image.mat.dtype)
    pose = image.pose.inverse() * action_pose
    polygon_projection = np.asarray([tuple(image.project(pose * p)) for p in polygon])
    cv2.polylines(image.mat, [polygon_projection], True, (color[0] * cm, color[1] * cm, color[2] * cm), thickness, lineType=cv2.LINE_AA)


def draw_around_box(image: OrthographicImage, box_data: Optional[BoxData], draw_lines=False) -> None:
    if not box_data:
        return

    assert box_data, 'Box contour should be drawn, but is false.'

    box_border = [Affine(*p) for p in box_data.contour]
    image_border = [Affine(*p) for p in _get_rect_contour([0.0, 0.0, 0.0], [10.0, 10.0, box_data.contour[0][2]])]
    box_projection = [image.project(image.pose.inverse() * p) for p in box_border]

    cm = get_color_scale(image.mat.dtype)

    if draw_lines:
        number_channels = image.mat.shape[-1] if len(image.mat.shape) > 2 else 1

        color = np.array([255 * cm] * number_channels)  # White
        cv2.polylines(image.mat, [np.asarray(box_projection)], True, color, 2, lineType=cv2.LINE_AA)

    else:
        color_array = np.array([image.mat[np.clip(p[1], 0, image.mat.shape[0] - 1), np.clip(p[0], 0, image.mat.shape[1] - 1)] for p in box_projection], dtype=np.float32)
        if len(color_array.shape) > 1:
            color_array[np.mean(color_array, axis=1) < cm] = np.nan
        else:
            color_array[color_array < cm] = np.nan
        color = np.nanmean(color_array, axis=0)
        np.nan_to_num(color, copy=False)
        image_border_projection = [image.project(image.pose.inverse() * p) for p in image_border]
        cv2.fillPoly(image.mat, np.array([image_border_projection, box_projection]), color.tolist())


def draw_object(image: OrthographicImage, object_data) -> None:
    if 'polygon' not in object_data:
        return

    cm = get_color_scale(image.mat.dtype)
    polygon = np.asarray([(int(p['x'] * image.mat.shape[1]), int(p['y'] * image.mat.shape[0])) for p in object_data['polygon']])
    cv2.polylines(image.mat, [polygon], True, (0 * cm, 255 * cm, 255 * cm), 1, lineType=cv2.LINE_AA)


def draw_pose(image: OrthographicImage, action_pose: RobotPose, convert_to_rgb=False, calibration_pattern=False, border_thickness=2, line_thickness=1) -> None:
    if convert_to_rgb and image.mat.ndim == 2:
        image.mat = cv2.cvtColor(image.mat, cv2.COLOR_GRAY2RGB)

    color_rect = (255, 0, 0)  # Blue
    color_lines = (0, 0, 255)  # Red
    color_direction = (0, 255, 0)  # Green

    rect = [Affine(*p) for p in _get_rect_contour([0.0, 0.0, 0.0], [200.0 / image.pixel_size, 200.0 / image.pixel_size, 0.0])]

    draw_polygon(image, action_pose, rect, color_rect, 2)

    draw_line(image, action_pose, Affine(90 / image.pixel_size, 0), Affine(100 / image.pixel_size, 0), color_rect, border_thickness)
    draw_line(image, action_pose, Affine(0.012, action_pose.d / 2), Affine(-0.012, action_pose.d / 2), color_lines, line_thickness)
    draw_line(image, action_pose, Affine(0.012, -action_pose.d / 2), Affine(-0.012, -action_pose.d / 2), color_lines, line_thickness)
    draw_line(image, action_pose, Affine(0, action_pose.d / 2), Affine(0, -action_pose.d / 2), color_lines, line_thickness)
    draw_line(image, action_pose, Affine(0.006, 0), Affine(-0.006, 0), color_lines, line_thickness)
    draw_line(image, action_pose, Affine(z=0.14), Affine(), color_direction, line_thickness)

    if calibration_pattern:
        color_calibration = (255, 255, 255)
        draw_line(image, action_pose, Affine(0, -0.1), Affine(0, 0.1), color_calibration, line_thickness)
        draw_line(image, action_pose, Affine(-0.1, 0), Affine(0.1, 0), color_calibration, line_thickness)


def draw_around_box2(image, box_data: BoxData, draw_lines=False):
    if not box_data:
        return

    assert box_data, 'Box contour should be drawn, but is false.'

    box_border = [Affine(*p) for p in box_data.contour]
    image_border = [Affine(*p) for p in _get_rect_contour([0.0, 0.0, 0.0], [10.0, 10.0, box_data.contour[0][2]])]
    box_projection = [image.project(p) for p in box_border]

    cm = get_color_scale(image.mat.dtype)

    if draw_lines:
        number_channels = image.mat.shape[-1] if len(image.mat.shape) > 2 else 1
        color = np.array([255 * cm] * number_channels)  # White
        cv2.polylines(image.mat, [np.asarray(box_projection)], True, color, 2, lineType=cv2.LINE_AA)

    else:
        color_array = np.array([image.mat[np.clip(p[1], 0, image.mat.shape[0] - 1), np.clip(p[0], 0, image.mat.shape[1] - 1)] for p in box_projection], dtype=np.float32)
        if len(color_array.shape) > 1:
            color_array[np.mean(color_array, axis=1) < cm] = np.nan
        else:
            color_array[color_array < cm] = np.nan

        color = np.nanmean(color_array, axis=0)
        np.nan_to_num(color, copy=False)

        image_border_projection = [image.project(p) for p in image_border]
        cv2.fillPoly(image.mat, np.array([image_border_projection, box_projection]), color.tolist())


def draw_pose2(image, grasp, gripper: Gripper, convert_to_rgb=False):
    if convert_to_rgb and image.mat.ndim == 2:
        image.mat = cv2.cvtColor(image.mat, cv2.COLOR_GRAY2RGB)

    color_rect = (255, 0, 0)  # Blue
    color_lines = (0, 0, 255)  # Red
    color_direction = (0, 255, 0)  # Green

    rect = [Affine(*p) for p in _get_rect_contour([0.0, 0.0, 0.0], [200.0 / image.pixel_size, 200.0 / image.pixel_size, 0.0])]

    action_pose = RobotPose(grasp.pose, d=grasp.stroke)

    draw_polygon(image, action_pose, rect, color_rect, 2)
    draw_line(image, action_pose, Affine(90 / image.pixel_size, 0), Affine(100 / image.pixel_size, 0), color_rect, 2)

    half_width = gripper.finger_width / 2

    draw_line(image, action_pose, Affine(half_width, action_pose.d / 2), Affine(-half_width, action_pose.d / 2), color_lines, 1)
    draw_line(image, action_pose, Affine(half_width, -action_pose.d / 2), Affine(-half_width, -action_pose.d / 2), color_lines, 1)

    half_width_height = half_width + gripper.finger_height
    draw_line(image, action_pose, Affine(half_width_height, action_pose.d / 2), Affine(-half_width_height, action_pose.d / 2), color_lines, 1)
    draw_line(image, action_pose, Affine(half_width_height, -action_pose.d / 2), Affine(-half_width_height, -action_pose.d / 2), color_lines, 1)

    draw_line(image, action_pose, Affine(0, action_pose.d / 2), Affine(0, -action_pose.d / 2), color_lines, 1)
    draw_line(image, action_pose, Affine(0.006, 0), Affine(-0.006, 0), color_lines, 1)

    if np.isfinite(action_pose.z) and (action_pose.b != 0 or action_pose.c != 0):
        draw_line(image, action_pose, Affine(z=0.14), Affine(), color_direction, 1)
