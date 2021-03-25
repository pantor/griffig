import cv2
import numpy as np

from _griffig import BoxData, Gripper, OrthographicImage
from pyaffx import Affine


from .image import crop
from .image import get_transformation
from .image import _get_rect_contour
from .image import get_box_projection


def get_area_of_interest(image, pose, size_cropped, size_result, return_mat=False):
    size_input = (image.mat.shape[1], image.mat.shape[0])
    center_image = (size_input[0] / 2, size_input[1] / 2)

    if size_result and size_cropped:
        scale = size_result[0] / size_cropped[0]
        assert scale == (size_result[1] / size_cropped[1])
    elif size_result:
        scale = size_result[0] / size_input[0]
        assert scale == (size_result[1] / size_input[1])
    else:
        scale = 1.0

    size_final = size_result or size_cropped or size_input

    trans = get_transformation(
        image.pixel_size * pose.y,
        image.pixel_size * pose.x,
        -pose.a,
        center_image,
        scale=scale,
        cropped=size_cropped,
    )
    mat_result = cv2.warpAffine(image.mat, trans, size_final, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if return_mat:
        return mat_result
    
    return OrthographicImage(mat_result, scale * image.pixel_size, image.min_depth, image.max_depth)


def get_inference_image(image, pose, size_cropped, size_area_cropped, size_area_result, return_mat=False):
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
    
    return OrthographicImage(mat_result, scale * image.pixel_size, image.min_depth, image.max_depth)


def draw_line(image, action_pose, pt1, pt2, color, thickness=1):
    cm = 1. / 255 if image.mat.dtype == np.float32 else np.iinfo(image.mat.dtype).max / 255
    pt1_projection = image.project(action_pose * pt1)
    pt2_projection = image.project(action_pose * pt2)
    cv2.line(image.mat, tuple(pt1_projection), tuple(pt2_projection), (color[0] * cm, color[1] * cm, color[2] * cm), thickness, lineType=cv2.LINE_AA)


def draw_polygon(image, action_pose, polygon, color, thickness=1):
    cm = 1. / 255 if image.mat.dtype == np.float32 else np.iinfo(image.mat.dtype).max / 255
    polygon_projection = np.asarray([tuple(image.project(action_pose * p)) for p in polygon])
    cv2.polylines(image.mat, [polygon_projection], True, (color[0] * cm, color[1] * cm, color[2] * cm), thickness, lineType=cv2.LINE_AA)


def draw_around_box(image, box_data: BoxData, draw_lines=False):
    if not box_data:
        return

    assert box_data, 'Box contour should be drawn, but is false.'

    box_border = [Affine(*p) for p in box_data.contour]
    image_border = [Affine(*p) for p in _get_rect_contour([0.0, 0.0, 0.0], [10.0, 10.0, box_data.contour[0][2]])]
    box_projection = [image.project(p) for p in box_border]

    color_multiplier = 1. / 255 if image.mat.dtype == np.float32 else np.iinfo(image.mat.dtype).max / 255

    if draw_lines:
        number_channels = image.mat.shape[-1] if len(image.mat.shape) > 2 else 1
        color = np.array([255 * color_multiplier] * number_channels)  # White
        cv2.polylines(image.mat, [np.asarray(box_projection)], True, color, 2, lineType=cv2.LINE_AA)

    else:
        color_array = np.array([image.mat[np.clip(p[1], 0, image.mat.shape[0] - 1), np.clip(p[0], 0, image.mat.shape[1] - 1)] for p in box_projection], dtype=np.float32)
        if len(color_array.shape) > 1:
            color_array[np.mean(color_array, axis=1) < color_multiplier] = np.nan
        else:
            color_array[color_array < color_multiplier] = np.nan

        color = np.nanmean(color_array, axis=0)
        np.nan_to_num(color, copy=False)

        image_border_projection = [image.project(p) for p in image_border]
        cv2.fillPoly(image.mat, np.array([image_border_projection, box_projection]), color.tolist())


def draw_pose(image, grasp, gripper: Gripper, convert_to_rgb=False):
    if convert_to_rgb and image.mat.ndim == 2:
        image.mat = cv2.cvtColor(image.mat, cv2.COLOR_GRAY2RGB)

    color_rect = (255, 0, 0)  # Blue
    color_lines = (0, 0, 255)  # Red
    color_direction = (0, 255, 0)  # Green

    rect = [Affine(*p) for p in _get_rect_contour([0.0, 0.0, 0.0], [200.0 / image.pixel_size, 200.0 / image.pixel_size, 0.0])]

    draw_polygon(image, grasp.pose, rect, color_rect, 2)
    draw_line(image, grasp.pose, Affine(90 / image.pixel_size, 0), Affine(100 / image.pixel_size, 0), color_rect, 2)
    
    half_width = gripper.width / 2
    
    draw_line(image, grasp.pose, Affine(half_width, grasp.stroke / 2), Affine(-half_width, grasp.stroke / 2), color_lines, 1)
    draw_line(image, grasp.pose, Affine(half_width, -grasp.stroke / 2), Affine(-half_width, -grasp.stroke / 2), color_lines, 1)
    
    half_width_height = half_width + gripper.height
    draw_line(image, grasp.pose, Affine(half_width_height, grasp.stroke / 2), Affine(-half_width_height, grasp.stroke / 2), color_lines, 1)
    draw_line(image, grasp.pose, Affine(half_width_height, -grasp.stroke / 2), Affine(-half_width_height, -grasp.stroke / 2), color_lines, 1)

    draw_line(image, grasp.pose, Affine(0, grasp.stroke / 2), Affine(0, -grasp.stroke / 2), color_lines, 1)
    draw_line(image, grasp.pose, Affine(0.006, 0), Affine(-0.006, 0), color_lines, 1)

    if False and not isinstance(grasp.pose.z, str) and np.isfinite(grasp.pose.z):
        draw_line(image, grasp.pose, Affine(z=0.14), Affine(), color_direction, 1)
