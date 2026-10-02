"""
Flags false positive triggers in sparse detection JSONs: videos where the camera
triggered but the only detections are objects of the scene (rocks, branches...).

A detection is a false positive detection when detections at the same location
(IoU >= iou_threshold) from the same camera appear in at least min_events events.
A video is a false positive video when all of its detections are false positive
detections. Only daytime (non-IR) videos are processed.
"""

import json
import logging
import multiprocessing
from functools import partial
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from data_processing.utils.db_utils import output_relative_path

logger = logging.getLogger(__name__)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one box and an array of boxes, all in (x, y, w, h) format."""
    inter_w = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2]) - np.maximum(
        box[0], boxes[:, 0]
    )
    inter_h = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3]) - np.maximum(
        box[1], boxes[:, 1]
    )
    inter_area = np.clip(inter_w, 0, None) * np.clip(inter_h, 0, None)
    union_area = box[2] * box[3] + boxes[:, 2] * boxes[:, 3] - inter_area
    return inter_area / np.maximum(union_area, 1e-8)


def flag_false_positive_detections(
    boxes: np.ndarray,
    event_ids: np.ndarray,
    iou_threshold: float,
    min_events: int,
) -> np.ndarray:
    """Flag the boxes that share their location with boxes of at least min_events events."""
    is_false_positive = np.zeros(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        if is_false_positive[i]:
            continue
        same_location = box_iou(boxes[i], boxes) >= iou_threshold
        if len(np.unique(event_ids[same_location])) >= min_events:
            is_false_positive[same_location] = True
    return is_false_positive


def flag_group(
    videos: List[Tuple[Path, str]],
    iou_threshold: float,
    min_events: int,
) -> int:
    """
    Flag the videos of one camera, given as (json_path, event_id) pairs, and write
    the flags back into their JSONs. Returns the number of false positive videos.
    """
    video_jsons = [json.loads(path.read_text()) for path, _ in videos]
    video_detections = [
        [det for frame in data["frames"] for det in frame["detections"]]
        for data in video_jsons
    ]
    boxes = np.array(
        [det["bbox"] for dets in video_detections for det in dets], dtype=np.float32
    ).reshape(-1, 4)
    event_ids = np.array(
        [
            event_id
            for (_, event_id), dets in zip(videos, video_detections)
            for _ in dets
        ]
    )
    is_false_positive = iter(
        flag_false_positive_detections(boxes, event_ids, iou_threshold, min_events)
    )

    n_false_positive_videos = 0
    for (path, _), data, dets in zip(videos, video_jsons, video_detections):
        for det in dets:
            det["false_positive_detection"] = bool(next(is_false_positive))
        is_false_positive_video = len(dets) > 0 and all(
            det["false_positive_detection"] for det in dets
        )
        data["info"]["false_positive_video"] = is_false_positive_video
        n_false_positive_videos += is_false_positive_video
        path.write_text(json.dumps(data, indent=2))
    return n_false_positive_videos


def flag_false_positive_jsons(
    json_folder: Path,
    raw_videos_df: pd.DataFrame,
    iou_threshold: float,
    min_events: int,
    n_jobs: int = 1,
) -> None:
    """Flag the daytime videos of json_folder in place, one camera group per process."""
    daytime_videos = raw_videos_df[~raw_videos_df["is_IR"].astype(bool)]
    groups = []
    for _, group_df in daytime_videos.groupby("group_id"):
        videos = [
            (Path(json_folder) / output_relative_path(raw_file_path, ".json"), event_id)
            for raw_file_path, event_id in zip(
                group_df["raw_file_path"], group_df["event_id"]
            )
        ]
        groups.append([(path, event_id) for path, event_id in videos if path.exists()])

    logger.info(
        "Checking for false positives among %d daytime videos from %d camera groups",
        sum(len(videos) for videos in groups),
        len(groups),
    )
    flag = partial(flag_group, iou_threshold=iou_threshold, min_events=min_events)
    with multiprocessing.Pool(max(1, n_jobs)) as pool:
        n_false_positive_videos = sum(pool.map(flag, groups))
    logger.info("Found %d false positive videos", n_false_positive_videos)
