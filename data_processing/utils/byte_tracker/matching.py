from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from supervision import box_iou_batch

if TYPE_CHECKING:
    from .single_object_track import STrack


def indices_to_matches(
    cost_matrix: np.ndarray, indices: np.ndarray, thresh: float
) -> tuple[np.ndarray, tuple, tuple]:
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = matched_cost <= thresh

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))
    return matches, unmatched_a, unmatched_b


def linear_assignment(
    cost_matrix: np.ndarray, thresh: float
) -> tuple[np.ndarray, tuple[int], tuple[int, int]]:
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )

    cost_matrix[cost_matrix > thresh] = thresh + 1e-4
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    indices = np.column_stack((row_ind, col_ind))

    return indices_to_matches(cost_matrix, indices, thresh)


def iou_distance(atracks: list[STrack], btracks: list[STrack]) -> np.ndarray:
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if _ious.size != 0:
        _ious = box_iou_batch(np.asarray(atlbrs), np.asarray(btlbrs))
    
    cost_matrix = 1 - _ious

    return cost_matrix

def buffered_box_iou_batch(
    boxes_a: np.ndarray, boxes_b: np.ndarray, buffer_scales: np.ndarray
) -> np.ndarray:
    """
    [N, M] IoU of the boxes enlarged on each side by buffer_scales[i] times their width
    and height, one buffer scale per box of boxes_a (C-BIoU, Yang et al. 2023).
    The boxes should be in [x0, y0, x1, y1] format.
    """
    buffer = buffer_scales[:, None, None]
    size_a = (boxes_a[:, 2:] - boxes_a[:, :2])[:, None, :]
    size_b = (boxes_b[:, 2:] - boxes_b[:, :2])[None, :, :]
    top_left_a = boxes_a[:, None, :2] - buffer * size_a
    bottom_right_a = boxes_a[:, None, 2:] + buffer * size_a
    top_left_b = boxes_b[None, :, :2] - buffer * size_b
    bottom_right_b = boxes_b[None, :, 2:] + buffer * size_b

    top_left = np.maximum(top_left_a, top_left_b)
    bottom_right = np.minimum(bottom_right_a, bottom_right_b)
    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    area_a = np.prod(bottom_right_a - top_left_a, 2)
    area_b = np.prod(bottom_right_b - top_left_b, 2)
    return area_inter / (area_a + area_b - area_inter + 1e-7)


def buffered_iou_distance(
    tracks: list[STrack], detections: list[STrack], buffer_scale: float, frame_id: int
) -> np.ndarray:
    """
    1 - buffered IoU between the predicted track boxes and the detections. The buffer
    grows with sqrt(frames since the last match of the track), so that lost tracks can
    be matched farther from their prediction.
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))
    frames_since_match = np.array([frame_id - track.frame_id for track in tracks])
    return 1 - buffered_box_iou_batch(
        np.array([track.tlbr for track in tracks]),
        np.array([detection.tlbr for detection in detections]),
        buffer_scale * np.sqrt(frames_since_match),
    )


def fuse_score(cost_matrix: np.ndarray, stracks: list[STrack]) -> np.ndarray:
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([strack.score for strack in stracks])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
