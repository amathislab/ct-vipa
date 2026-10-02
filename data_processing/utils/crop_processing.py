import ast
import os
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np

# Boundary around the crop
BOUNDARY = 4


def open_img(img_path):

    with open(img_path, "rb") as img_file:
        img_str = img_file.read()
        img_array = np.frombuffer(img_str, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return img


def normalize_bbox(bbox: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert relative bounding box coordinates to absolute pixel coordinates

    Args:
        bbox: Relative bounding box coordinates as (xmin, ymin, width, height)
        img_size: Image size as (width, height) in pixels

    Returns:
        Bounding box coordinates as (x1, y1, x2, y2) in pixels.
    """
    if isinstance(bbox, str):
        xmin, ymin, width, height = ast.literal_eval(bbox)
    else:
        xmin, ymin, width, height = tuple(bbox)
    img_w, img_h = img_size
    x1, y1 = xmin, ymin
    x2, y2 = x1 + width, y1 + height
    return [int(x1 * img_w), int(y1 * img_h), int(x2 * img_w), int(y2 * img_h)]


def adjust_crop_dimensions(
    bbox_img: Union[List, np.ndarray], target_size: Tuple, img_dims: Tuple
) -> List:
    """
    Adjust bounding box dimensions if they are too small in any dimension.
    This has the effect to add background around the detected object in practice
    """
    img_width, img_height = img_dims
    bbox_img_w = bbox_img[2] - bbox_img[0]
    bbox_img_h = bbox_img[3] - bbox_img[1]
    extend_width = max(target_size[0] + BOUNDARY - bbox_img_w, 0)
    extend_height = max(target_size[1] + BOUNDARY - bbox_img_h, 0)

    bbox_custom = bbox_extend(
        bbox_img,
        (bbox_img_w, bbox_img_h),
        (img_width, img_height),
        (extend_width, extend_height),
    )

    return bbox_custom


def dim_padding(bbox_img: List, target_size: Tuple, img_dims: Tuple) -> List:
    """
    Adjust bounding box dimensions so that the bbox aspect ratio matches
    the desired aspect ratio
    """

    img_width, img_height = img_dims
    bbox_custom = bbox_img.copy()
    bbox_img_w = bbox_img[2] - bbox_img[0]
    bbox_img_h = bbox_img[3] - bbox_img[1]
    target_width, target_height = target_size

    # Get ratio
    target_ratio = target_width / target_height
    bbox_ratio = (bbox_img_w + BOUNDARY) / (bbox_img_h + BOUNDARY)
    extend_width, extend_height = (0, 0)

    # If bounding box is wider than desired ratio, we pad in height
    if bbox_ratio > target_ratio:
        desired_height = (bbox_img_w + BOUNDARY) / target_ratio
        extend_height = max(int(np.ceil(desired_height - bbox_img_h + BOUNDARY)), 0)

    # If bounding box is thinner than desired ratio, we pad in width
    elif bbox_ratio <= target_ratio:
        desired_width = target_ratio * (bbox_img_h + BOUNDARY)
        extend_width = max(int(np.ceil(desired_width - bbox_img_w + BOUNDARY)), 0)

    bbox_custom = bbox_extend(
        bbox_img,
        (bbox_img_w, bbox_img_h),
        (img_width, img_height),
        (extend_width, extend_height),
    )

    return bbox_custom


def bbox_extend(
    bbox_img: Tuple[int],
    bbox_dims: Tuple[int],
    img_dims: Tuple[int],
    extend_dims: Tuple[int],
) -> List[int]:

    bbox_img_w, bbox_img_h = bbox_dims
    img_width, img_height = img_dims
    extend_w, extend_h = extend_dims
    bbox_custom = bbox_img.copy()

    if extend_w > 0:
        space_remaining_left = bbox_img[0]
        space_remaining_right = img_width - bbox_img[2]

        # There is enough space to centrally extend the crop
        if (space_remaining_left >= np.ceil(extend_w / 2)) and (
            space_remaining_right >= np.ceil(extend_w / 2)
        ):
            bbox_custom[0] = np.ceil(bbox_img[0] - extend_w / 2)
            bbox_custom[2] = np.ceil(bbox_img[2] + extend_w / 2)

        # There is not enough space to centrally extend the crop on the left
        elif space_remaining_left < np.ceil(extend_w / 2):
            extra_space_right = np.ceil(extend_w / 2) - space_remaining_left
            bbox_custom[0] = 0
            bbox_custom[2] = min(
                np.ceil(bbox_img[2] + extend_w / 2 + extra_space_right), img_width
            )

        # There is not enough space to centrally extend the crop on the right
        elif space_remaining_right < np.ceil(extend_w / 2):
            extra_space_left = np.ceil(extend_w / 2) - space_remaining_right
            bbox_custom[0] = max(
                np.ceil(bbox_img[0] - extend_w / 2) - extra_space_left, 0
            )
            bbox_custom[2] = img_width

    # Same procedure for the height
    if extend_h > 0:
        space_remaining_top = bbox_img[1]
        space_remaining_bottom = img_height - bbox_img[3]
        if (space_remaining_top >= np.ceil(extend_h / 2)) and (
            space_remaining_bottom >= np.ceil(extend_h / 2)
        ):
            bbox_custom[1] = np.ceil(bbox_img[1] - extend_h / 2)
            bbox_custom[3] = np.ceil(bbox_img[3] + extend_h / 2)
        elif space_remaining_top < np.ceil(extend_h / 2):
            extra_space_bottom = np.ceil(extend_h / 2) - space_remaining_top
            bbox_custom[1] = 0
            bbox_custom[3] = min(
                np.ceil(bbox_img[3] + extend_h / 2 + extra_space_bottom), img_height
            )
        elif space_remaining_bottom < np.ceil(extend_h / 2):
            extra_space_top = np.ceil(extend_h / 2) - space_remaining_bottom
            bbox_custom[1] = max(
                0, np.ceil(bbox_img[1] - extend_h / 2) - extra_space_top
            )
            bbox_custom[3] = img_height

    bbox_custom = [int(c) for c in bbox_custom]
    return bbox_custom


def save_crop(crop: np.ndarray, filepath: Path) -> Path:
    if not filepath.parent.exists():
        os.makedirs(filepath.parent)
    cv2.imwrite(str(filepath), crop)
