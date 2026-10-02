"""
Unit tests for data_processing/utils/false_positive_filter.py

Tests verify:
- IoU between one box and many boxes
- Detections at the same location over min_events events are flagged
- A video is a false positive only if all of its detections are false positives
- Videos without detections and IR videos are never flagged
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from data_processing.utils.db_utils import file_id_from_path, output_relative_path
from data_processing.utils.false_positive_filter import (
    box_iou,
    flag_false_positive_detections,
    flag_false_positive_jsons,
    flag_group,
)

STATIC_BOX = [0.1, 0.1, 0.2, 0.2]
ANIMAL_BOX = [0.6, 0.5, 0.3, 0.3]


def write_detection_json(
    path: Path, file_id: str, frames_boxes: List[List[List[float]]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "info": {"file_id": file_id},
        "frames": [
            {
                "frame_id": frame_id,
                "detections": [
                    {"bbox": box, "conf": 0.9, "category": "0"} for box in boxes
                ],
            }
            for frame_id, boxes in enumerate(frames_boxes)
        ],
    }
    path.write_text(json.dumps(data))


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def build_raw_videos_df(
    raw_file_paths: List[str],
    event_ids: List[str],
    is_IR: Optional[List[bool]] = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file_id": [file_id_from_path(Path(path)) for path in raw_file_paths],
            "group_id": [Path(path).parent.as_posix() for path in raw_file_paths],
            "raw_file_path": raw_file_paths,
            "event_id": event_ids,
            "is_IR": is_IR if is_IR is not None else [False] * len(raw_file_paths),
        }
    )


class TestBoxIou:
    def test_identical_boxes(self):
        boxes = np.array([STATIC_BOX], dtype=np.float32)
        assert box_iou(boxes[0], boxes)[0] == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        boxes = np.array([STATIC_BOX, ANIMAL_BOX], dtype=np.float32)
        assert box_iou(boxes[0], boxes)[1] == 0.0

    def test_partial_overlap(self):
        boxes = np.array([[0, 0, 2, 2], [1, 0, 2, 2]], dtype=np.float32)
        # intersection 2, union 6
        assert box_iou(boxes[0], boxes)[1] == pytest.approx(1 / 3)


class TestFlagFalsePositiveDetections:
    def test_same_location_over_min_events_is_flagged(self):
        boxes = np.array([STATIC_BOX, STATIC_BOX], dtype=np.float32)
        flags = flag_false_positive_detections(
            boxes, np.array(["e1", "e2"]), iou_threshold=0.8, min_events=2
        )
        assert flags.tolist() == [True, True]

    def test_same_location_within_one_event_is_not_flagged(self):
        boxes = np.array([STATIC_BOX, STATIC_BOX, STATIC_BOX], dtype=np.float32)
        flags = flag_false_positive_detections(
            boxes, np.array(["e1", "e1", "e1"]), iou_threshold=0.8, min_events=2
        )
        assert flags.tolist() == [False, False, False]

    def test_different_locations_are_not_flagged(self):
        boxes = np.array([STATIC_BOX, ANIMAL_BOX], dtype=np.float32)
        flags = flag_false_positive_detections(
            boxes, np.array(["e1", "e2"]), iou_threshold=0.8, min_events=2
        )
        assert flags.tolist() == [False, False]

    def test_only_repeated_location_is_flagged(self):
        boxes = np.array([STATIC_BOX, ANIMAL_BOX, STATIC_BOX], dtype=np.float32)
        flags = flag_false_positive_detections(
            boxes, np.array(["e1", "e1", "e2"]), iou_threshold=0.8, min_events=2
        )
        assert flags.tolist() == [True, False, True]

    def test_empty_input(self):
        flags = flag_false_positive_detections(
            np.zeros((0, 4), dtype=np.float32),
            np.array([]),
            iou_threshold=0.8,
            min_events=2,
        )
        assert len(flags) == 0


class TestFlagGroup:
    def test_video_with_only_false_positive_detections_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video1, video2 = Path(tmpdir) / "v1.json", Path(tmpdir) / "v2.json"
            write_detection_json(video1, "cam/v1.mp4", [[STATIC_BOX], [STATIC_BOX]])
            write_detection_json(video2, "cam/v2.mp4", [[STATIC_BOX]])

            n_flagged = flag_group([(video1, "e1"), (video2, "e2")], 0.8, 2)

            assert n_flagged == 2
            for path in (video1, video2):
                data = read_json(path)
                assert data["info"]["false_positive_video"] is True
                assert all(
                    det["false_positive_detection"]
                    for frame in data["frames"]
                    for det in frame["detections"]
                )

    def test_video_with_an_animal_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video1, video2 = Path(tmpdir) / "v1.json", Path(tmpdir) / "v2.json"
            write_detection_json(video1, "cam/v1.mp4", [[STATIC_BOX, ANIMAL_BOX]])
            write_detection_json(video2, "cam/v2.mp4", [[STATIC_BOX]])

            n_flagged = flag_group([(video1, "e1"), (video2, "e2")], 0.8, 2)

            assert n_flagged == 1
            detections = read_json(video1)["frames"][0]["detections"]
            assert read_json(video1)["info"]["false_positive_video"] is False
            assert [det["false_positive_detection"] for det in detections] == [
                True,
                False,
            ]
            assert read_json(video2)["info"]["false_positive_video"] is True

    def test_video_without_detections_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video = Path(tmpdir) / "v1.json"
            write_detection_json(video, "cam/v1.mp4", [[], []])

            assert flag_group([(video, "e1")], 0.8, 1) == 0
            assert read_json(video)["info"]["false_positive_video"] is False

    def test_preserves_other_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video = Path(tmpdir) / "v1.json"
            write_detection_json(video, "cam/v1.mp4", [[STATIC_BOX]])

            flag_group([(video, "e1")], 0.8, 2)

            data = read_json(video)
            assert data["info"]["file_id"] == "cam/v1.mp4"
            assert data["frames"][0]["detections"][0]["conf"] == 0.9


class TestFlagFalsePositiveJsons:
    def test_flags_daytime_videos_per_camera(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_folder = Path(tmpdir)
            raw_file_paths = ["camA/v1.mp4", "camA/v2.mp4", "camB/v3.mp4"]
            for raw_file_path in raw_file_paths:
                write_detection_json(
                    json_folder / output_relative_path(raw_file_path, ".json"),
                    file_id_from_path(Path(raw_file_path)),
                    [[STATIC_BOX]],
                )

            flag_false_positive_jsons(
                json_folder,
                build_raw_videos_df(raw_file_paths, ["e1", "e2", "e3"]),
                iou_threshold=0.8,
                min_events=2,
            )

            # camA sees the same box in two events, camB only in one
            assert (
                read_json(json_folder / "camA/camA_v1.json")["info"]["false_positive_video"]
                is True
            )
            assert (
                read_json(json_folder / "camA/camA_v2.json")["info"]["false_positive_video"]
                is True
            )
            assert (
                read_json(json_folder / "camB/camB_v3.json")["info"]["false_positive_video"]
                is False
            )

    def test_IR_videos_are_not_processed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_folder = Path(tmpdir)
            raw_file_paths = ["cam/v1.mp4", "cam/v2.mp4"]
            for raw_file_path in raw_file_paths:
                write_detection_json(
                    json_folder / output_relative_path(raw_file_path, ".json"),
                    file_id_from_path(Path(raw_file_path)),
                    [[STATIC_BOX]],
                )

            flag_false_positive_jsons(
                json_folder,
                build_raw_videos_df(raw_file_paths, ["e1", "e2"], is_IR=[False, True]),
                iou_threshold=0.8,
                min_events=2,
            )

            assert (
                read_json(json_folder / "cam/cam_v1.json")["info"]["false_positive_video"]
                is False
            )
            assert (
                "false_positive_video"
                not in read_json(json_folder / "cam/cam_v2.json")["info"]
            )

    def test_videos_without_json_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_folder = Path(tmpdir)
            write_detection_json(
                json_folder / "cam/cam_v1.json", "cam_v1", [[STATIC_BOX]]
            )

            flag_false_positive_jsons(
                json_folder,
                build_raw_videos_df(["cam/v1.mp4", "cam/missing.mp4"], ["e1", "e2"]),
                iou_threshold=0.8,
                min_events=2,
            )

            assert (
                read_json(json_folder / "cam/cam_v1.json")["info"]["false_positive_video"]
                is False
            )
