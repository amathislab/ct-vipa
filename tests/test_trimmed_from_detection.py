"""
Unit tests for trimmed_from_detection.py

This module tests the functionality of video trimming based on MegaDetector detections,
including human detection removal, activity-based frame selection, full-length linking
when trimming is skipped, and error handling.
"""

import argparse
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_processing.trimmed_from_detection import (
    contains_human,
    get_animal_segments,
    get_input_folder,
    link_video,
    process_trimming,
)
from data_processing.utils.db_utils import file_id_from_path

MODULE = "data_processing.trimmed_from_detection"
PROJECT_PATH = "/test/path"
RAW_VIDEOS_FOLDER = "/test/videos"
DETECTOR_LABEL_MAP = {"1": "animal", "2": "person", "3": "vehicle"}
VIDEO_DURATION = 10.0


class TestGetAnimalSegments:
    """Test cases for the get_animal_segments function."""

    def test_get_animal_segments_valid_segment(self):
        """Test that valid animal segments are returned correctly."""
        video_detections_df = pd.DataFrame(
            {
                "frame_id": [0, 10, 20, 30, 40, 50],
                "conf": [0.5, 0.9, 0.85, 0.95, 0.88, 0.92],
                "category": ["1", "1", "1", "1", "1", "1"],
            }
        )
        min_frames = 20
        min_detection_threshold = 0.8

        result = get_animal_segments(
            video_detections_df, min_frames, min_detection_threshold
        )

        assert result is not None
        start_frame, end_frame = result
        assert start_frame == 0
        assert end_frame == 50
        # numpy integers are stored as BLOBs by sqlite3
        assert type(end_frame) is int

    def test_get_animal_segments_too_short(self):
        """Test that segments shorter than min_frames return None."""
        video_detections_df = pd.DataFrame(
            {
                "frame_id": [10, 15, 20],  # Only 10 frames span
                "conf": [0.9, 0.85, 0.95],
                "category": ["1", "1", "1"],
            }
        )
        min_frames = 20
        min_detection_threshold = 0.8

        result = get_animal_segments(
            video_detections_df, min_frames, min_detection_threshold
        )

        assert result is None

    def test_get_animal_segments_no_detections_above_threshold(self):
        """Test that None is returned when no detections pass threshold."""
        video_detections_df = pd.DataFrame(
            {
                "frame_id": [10, 20, 30, 40, 50],
                "conf": [0.5, 0.4, 0.6, 0.3, 0.5],  # All below threshold
                "category": ["1", "1", "1", "1", "1"],
            }
        )
        min_frames = 20
        min_detection_threshold = 0.8

        result = get_animal_segments(
            video_detections_df, min_frames, min_detection_threshold
        )

        assert result is None

    def test_get_animal_segments_exact_min_frames(self):
        """Test behavior when segment length equals min_frames."""
        # start_frame is always 0, so end_frame must equal min_frames for exact boundary
        video_detections_df = pd.DataFrame(
            {
                "frame_id": [
                    10,
                    20,
                ],  # end_frame=20, 20-0=20 == min_frames, not > min_frames
                "conf": [0.9, 0.95],
                "category": ["1", "1"],
            }
        )
        min_frames = 20
        min_detection_threshold = 0.8

        result = get_animal_segments(
            video_detections_df, min_frames, min_detection_threshold
        )

        # None as the condition end_frame - start_frame > min_frames is strict
        assert result is None


class TestContainsHuman:
    """Test cases for the contains_human function."""

    def test_contains_human_true(self):
        """Test that human detections are correctly identified."""
        video_detections_df = pd.DataFrame(
            {
                "category": ["1", "2", "1", "2"],  # "2" is human_key
                "conf": [0.9, 0.85, 0.95, 0.88],
            }
        )
        human_conf_threshold = 0.8
        human_key = "2"

        result = contains_human(video_detections_df, human_conf_threshold, human_key)

        assert result is True

    def test_contains_human_false_no_humans(self):
        """Test that videos without humans return False."""
        video_detections_df = pd.DataFrame(
            {
                "category": ["1", "1", "1", "1"],  # No human_key
                "conf": [0.9, 0.85, 0.95, 0.88],
            }
        )
        human_conf_threshold = 0.8
        human_key = "2"

        result = contains_human(video_detections_df, human_conf_threshold, human_key)

        assert result is False

    def test_contains_human_below_threshold(self):
        """Test that human detections below threshold are ignored."""
        video_detections_df = pd.DataFrame(
            {
                "category": ["1", "2", "1", "2"],  # "2" is human_key
                "conf": [0.9, 0.5, 0.95, 0.6],  # Human detections below threshold
            }
        )
        human_conf_threshold = 0.8
        human_key = "2"

        result = contains_human(video_detections_df, human_conf_threshold, human_key)

        assert result is False

    def test_contains_human_edge_case_threshold(self):
        """Test human detection at exact threshold."""
        video_detections_df = pd.DataFrame(
            {
                "category": ["1", "2", "1"],
                "conf": [0.9, 0.8, 0.95],  # Human exactly at threshold
            }
        )
        human_conf_threshold = 0.8
        human_key = "2"

        result = contains_human(video_detections_df, human_conf_threshold, human_key)

        assert result is True

    def test_contains_human_mixed_confidence(self):
        """Test with mixed human confidence levels."""
        video_detections_df = pd.DataFrame(
            {
                "category": ["2", "2", "1", "2"],
                "conf": [0.5, 0.6, 0.95, 0.85],  # One human above threshold
            }
        )
        human_conf_threshold = 0.8
        human_key = "2"

        result = contains_human(video_detections_df, human_conf_threshold, human_key)

        assert result is True


class TestGetInputFolder:
    """Test cases for the get_input_folder function."""

    @patch("data_processing.trimmed_from_detection.get_previous_output")
    def test_get_input_folder_from_false_positive_examination(self, mock_get_previous):
        """Test getting input folder from flag_false_positive_videos."""
        mock_conn = MagicMock()
        stage_name = "flag_false_positive_videos"
        output_type = "json"
        expected_folder = "/path/to/filtered/output"

        mock_get_previous.return_value = expected_folder

        result = get_input_folder(mock_conn, stage_name, output_type)

        assert result == expected_folder
        mock_get_previous.assert_called_once_with(
            conn=mock_conn,
            stage_name="flag_false_positive_videos",
            output_type="json",
        )

    @patch("data_processing.trimmed_from_detection.get_previous_output")
    def test_get_input_folder_fallback_to_sparse_detection(self, mock_get_previous):
        """Test fallback to process_videos_sparse_detection when false positive examination is None."""
        mock_conn = MagicMock()
        stage_name = "flag_false_positive_videos"
        output_type = "json"
        expected_folder = "/path/to/sparse/output"

        # First call returns None, second returns fallback
        mock_get_previous.side_effect = [None, expected_folder]

        result = get_input_folder(mock_conn, stage_name, output_type)

        assert result == expected_folder
        assert mock_get_previous.call_count == 2

    @patch("data_processing.trimmed_from_detection.get_previous_output")
    def test_get_input_folder_both_none_raises_error(self, mock_get_previous):
        """Test that ValueError is raised when both sources return None."""
        mock_conn = MagicMock()
        stage_name = "flag_false_positive_videos"
        output_type = "json"

        # Both calls return None
        mock_get_previous.return_value = None

        with pytest.raises(
            ValueError,
            match="No output found from either flag_false_positive_videos or process_videos_sparse_detection stages",
        ):
            get_input_folder(mock_conn, stage_name, output_type)


def make_trimming_database(
    video_fps: Dict[str, float],
    detector_label_map: Optional[Dict[str, str]] = DETECTOR_LABEL_MAP,
) -> sqlite3.Connection:
    """In-memory database holding the sparse detection parameters and the Videos
    rows looked up by process_trimming, keyed by raw relative path."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE Stages (stage_name TEXT, parameters TEXT)")
    conn.execute(
        "CREATE TABLE Videos (file_id TEXT, raw_file_path TEXT, fps REAL, duration REAL)"
    )
    conn.execute(
        "INSERT INTO Stages VALUES ('process_videos_sparse_detection', ?)",
        (
            (
                json.dumps({"detector_label_map": detector_label_map})
                if detector_label_map is not None
                else None
            ),
        ),
    )
    conn.executemany(
        "INSERT INTO Videos VALUES (?, ?, ?, ?)",
        [
            (file_id_from_path(Path(raw_file_path)), raw_file_path, fps, VIDEO_DURATION)
            for raw_file_path, fps in video_fps.items()
        ],
    )
    return conn


def make_detections(
    raw_file_path: str,
    frame_ids: List[int],
    confs: List[float],
    categories: Optional[List[str]] = None,
    false_positive_video: bool = False,
) -> pd.DataFrame:
    """Detections in the format returned by megadetector_utils.load_all_detections."""
    return pd.DataFrame(
        {
            "file_id": file_id_from_path(Path(raw_file_path)),
            "group_id": str(Path(raw_file_path).parent),
            "frame_id": frame_ids,
            "conf": confs,
            "category": categories or ["1"] * len(frame_ids),
            "false_positive_video": false_positive_video,
        }
    )


def make_trimming_params(**overrides: Any) -> Dict[str, Any]:
    return {
        "output_video_folder_name": "trimmed_videos",
        "debug_n_videos": None,
        "human_conf_threshold": 0.9,
        "min_frames": 30,
        "min_detection_threshold": 0.8,
        "save_videos": True,
        "skip_existing": False,
        "frame_padding": 10,
        "skip_trimming": False,
        **overrides,
    }


@pytest.fixture
def pipeline() -> Iterator[SimpleNamespace]:
    """Patch the config, database and I/O boundaries of process_trimming.
    Tests fill in the stage parameters, the database and the loaded detections."""
    with (
        patch(f"{MODULE}.parse_config") as parse_config,
        patch(f"{MODULE}.get_stage_parameters") as get_stage_parameters,
        patch(f"{MODULE}.load_database") as load_database,
        patch(f"{MODULE}.get_input_folder", return_value="/test/json"),
        patch(f"{MODULE}.load_all_detections") as load_all_detections,
        patch(f"{MODULE}.get_previous_output", return_value=RAW_VIDEOS_FOLDER),
        patch(f"{MODULE}.upsert_video_data") as upsert_video_data,
        patch(f"{MODULE}.upsert_stage") as upsert_stage,
    ):
        parse_config.return_value = {
            "global": {
                "project_path": PROJECT_PATH,
                "database_path": "/test/db.sqlite",
            }
        }
        get_stage_parameters.return_value = make_trimming_params()
        yield SimpleNamespace(
            opt=argparse.Namespace(config_file="test_config.yaml"),
            get_stage_parameters=get_stage_parameters,
            load_database=load_database,
            load_all_detections=load_all_detections,
            upsert_video_data=upsert_video_data,
            upsert_stage=upsert_stage,
        )


def upserted_output_df(pipeline: SimpleNamespace) -> pd.DataFrame:
    return pipeline.upsert_video_data.call_args[0][1]


class TestProcessTrimming:
    """Test cases for the main process_trimming function."""

    @patch(f"{MODULE}.trim_video", return_value=2.0)
    def test_process_trimming_success(self, mock_trim_video, pipeline):
        """Test that a video with animal activity is trimmed and recorded."""
        raw_file_path = "cam1/IMG_0001.MP4"
        conn = make_trimming_database({raw_file_path: 30.0})
        pipeline.load_database.return_value = conn
        pipeline.load_all_detections.return_value = make_detections(
            raw_file_path, frame_ids=[10, 20, 50], confs=[0.9, 0.85, 0.95]
        )

        process_trimming(pipeline.opt)

        output_video_folder = Path(PROJECT_PATH) / "trimmed_videos"
        mock_trim_video.assert_called_once_with(
            Path(RAW_VIDEOS_FOLDER) / raw_file_path,
            0,
            50,
            output_video_folder / "cam1" / "cam1_IMG_0001.MP4",
            True,
            False,
            10,
            30.0,
        )

        output_df = upserted_output_df(pipeline)
        assert output_df.to_dict("records") == [
            {
                "file_id": "cam1_IMG_0001",
                "trimmed_video_file_path": "cam1/cam1_IMG_0001.MP4",
                "trimmed_duration": 2.0,
                "trimmed_start_frame": 0,
                "trimmed_end_frame": 50,
            }
        ]
        assert (
            pipeline.upsert_stage.call_args[1]["output_video_folder"]
            == output_video_folder
        )
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

    @patch(f"{MODULE}.trim_video", return_value=2.0)
    def test_process_trimming_skips_videos_with_humans(self, mock_trim_video, pipeline):
        """Test that videos containing humans are skipped."""
        animal_video = "cam1/IMG_0001.MP4"
        human_video = "cam1/IMG_0002.MP4"
        pipeline.load_database.return_value = make_trimming_database(
            {animal_video: 30.0, human_video: 30.0}
        )
        pipeline.load_all_detections.return_value = pd.concat(
            [
                make_detections(animal_video, [10, 50], [0.9, 0.95]),
                make_detections(
                    human_video, [10, 20, 50], [0.9, 0.95, 0.95], ["1", "2", "1"]
                ),
            ]
        )

        process_trimming(pipeline.opt)

        mock_trim_video.assert_called_once()
        assert (
            mock_trim_video.call_args[0][0] == Path(RAW_VIDEOS_FOLDER) / animal_video
        )
        assert upserted_output_df(pipeline)["file_id"].tolist() == [file_id_from_path(Path(animal_video))]

    @patch(f"{MODULE}.trim_video", return_value=2.0)
    def test_process_trimming_skips_videos_with_short_segments(
        self, mock_trim_video, pipeline
    ):
        """Test that videos with segments shorter than min_frames are skipped."""
        long_video = "cam1/IMG_0001.MP4"
        short_video = "cam1/IMG_0002.MP4"
        pipeline.get_stage_parameters.return_value = make_trimming_params(min_frames=50)
        pipeline.load_database.return_value = make_trimming_database(
            {long_video: 30.0, short_video: 30.0}
        )
        pipeline.load_all_detections.return_value = pd.concat(
            [
                make_detections(long_video, [10, 80], [0.9, 0.95]),
                make_detections(short_video, [10, 30], [0.9, 0.95]),
            ]
        )

        process_trimming(pipeline.opt)

        mock_trim_video.assert_called_once()
        assert mock_trim_video.call_args[0][0] == Path(RAW_VIDEOS_FOLDER) / long_video
        assert upserted_output_df(pipeline)["file_id"].tolist() == [file_id_from_path(Path(long_video))]

    @patch(f"{MODULE}.trim_video", return_value=2.0)
    def test_process_trimming_removes_false_positives(self, mock_trim_video, pipeline):
        """Test that videos flagged as false positives are not trimmed."""
        true_positive_video = "cam1/IMG_0001.MP4"
        false_positive_video_path = "cam1/IMG_0002.MP4"
        pipeline.load_database.return_value = make_trimming_database(
            {true_positive_video: 30.0, false_positive_video_path: 30.0}
        )
        pipeline.load_all_detections.return_value = pd.concat(
            [
                make_detections(true_positive_video, [10, 50], [0.9, 0.95]),
                make_detections(
                    false_positive_video_path,
                    [10, 50],
                    [0.9, 0.95],
                    false_positive_video=True,
                ),
            ]
        )

        process_trimming(pipeline.opt)

        mock_trim_video.assert_called_once()
        assert upserted_output_df(pipeline)["file_id"].tolist() == [file_id_from_path(Path(true_positive_video))]

    @patch(f"{MODULE}.trim_video")
    def test_process_trimming_handles_missing_detector_params(
        self, mock_trim_video, pipeline
    ):
        """Test that processing stops when the sparse detection parameters are missing."""
        raw_file_path = "cam1/IMG_0001.MP4"
        pipeline.load_database.return_value = make_trimming_database(
            {raw_file_path: 30.0}, detector_label_map=None
        )
        pipeline.load_all_detections.return_value = make_detections(
            raw_file_path, [10, 50], [0.9, 0.95]
        )

        process_trimming(pipeline.opt)

        mock_trim_video.assert_not_called()
        pipeline.upsert_video_data.assert_not_called()
        pipeline.upsert_stage.assert_not_called()

    @patch(f"{MODULE}.trim_video", return_value=2.0)
    def test_process_trimming_respects_debug_n_videos(self, mock_trim_video, pipeline):
        """Test that debug_n_videos parameter limits the number of videos processed."""
        raw_file_paths = ["cam1/IMG_0001.MP4", "cam1/IMG_0002.MP4", "cam2/IMG_0001.MP4"]
        pipeline.get_stage_parameters.return_value = make_trimming_params(
            debug_n_videos=2
        )
        pipeline.load_database.return_value = make_trimming_database(
            {raw_file_path: 30.0 for raw_file_path in raw_file_paths}
        )
        pipeline.load_all_detections.return_value = pd.concat(
            [make_detections(raw_file_path, [10, 50], [0.9, 0.95]) for raw_file_path in raw_file_paths]
        )

        process_trimming(pipeline.opt)

        assert mock_trim_video.call_count == 2
        assert upserted_output_df(pipeline)["file_id"].tolist() == [
            file_id_from_path(Path(raw_file_path)) for raw_file_path in raw_file_paths[:2]
        ]

    @patch(f"{MODULE}.os.path.exists", return_value=True)
    @patch(f"{MODULE}.os.makedirs")
    @patch(f"{MODULE}.ffmpeg")
    def test_process_trimming_skip_existing_true(
        self, mock_ffmpeg, mock_makedirs, mock_exists, pipeline
    ):
        """Test that skip_existing=True prevents re-processing existing videos."""
        raw_file_path = "cam1/IMG_0001.MP4"
        pipeline.get_stage_parameters.return_value = make_trimming_params(
            skip_existing=True
        )
        pipeline.load_database.return_value = make_trimming_database({raw_file_path: 30.0})
        pipeline.load_all_detections.return_value = make_detections(
            raw_file_path, [10, 20, 50], [0.9, 0.85, 0.95]
        )
        mock_stream = MagicMock()
        mock_ffmpeg.input.return_value = mock_stream
        mock_ffmpeg.output.return_value = mock_stream

        process_trimming(pipeline.opt)

        # ffmpeg is not run because the output exists and skip_existing=True
        mock_stream.run.assert_not_called()
        # The existing trimmed video is still recorded
        assert upserted_output_df(pipeline)["file_id"].tolist() == [file_id_from_path(Path(raw_file_path))]

    @patch(f"{MODULE}.os.path.exists", return_value=True)
    @patch(f"{MODULE}.os.makedirs")
    @patch(f"{MODULE}.ffmpeg")
    def test_process_trimming_skip_existing_false(
        self, mock_ffmpeg, mock_makedirs, mock_exists, pipeline
    ):
        """Test that skip_existing=False replaces existing videos."""
        raw_file_path = "cam1/IMG_0001.MP4"
        pipeline.load_database.return_value = make_trimming_database({raw_file_path: 30.0})
        pipeline.load_all_detections.return_value = make_detections(
            raw_file_path, [10, 20, 50], [0.9, 0.85, 0.95]
        )
        mock_stream = MagicMock()
        mock_ffmpeg.input.return_value = mock_stream
        mock_ffmpeg.output.return_value = mock_stream
        mock_stream.compile.return_value = ["ffmpeg", "-i", "input.mp4"]
        mock_stream.run.return_value = (b"", b"")

        process_trimming(pipeline.opt)

        mock_ffmpeg.input.assert_called_once_with(
            str(Path(RAW_VIDEOS_FOLDER) / raw_file_path), ss=0.0
        )
        mock_stream.run.assert_called_once()
        assert mock_stream.run.call_args[1]["overwrite_output"] is True

    @patch(f"{MODULE}.trim_video")
    def test_process_trimming_without_trimmed_videos(self, mock_trim_video, pipeline):
        """Test that a run where every video is skipped completes without trimming."""
        raw_file_path = "cam1/IMG_0001.MP4"
        pipeline.load_database.return_value = make_trimming_database({raw_file_path: 30.0})
        pipeline.load_all_detections.return_value = make_detections(
            raw_file_path, [10, 20, 50], [0.9, 0.95, 0.95], ["1", "2", "1"]
        )

        process_trimming(pipeline.opt)

        mock_trim_video.assert_not_called()
        assert upserted_output_df(pipeline).empty

    @patch(f"{MODULE}.trim_video", return_value=2.0)
    def test_process_trimming_ignores_non_animal_detections(
        self, mock_trim_video, pipeline
    ):
        """Test that the trimmed segment ends at the last animal detection."""
        raw_file_path = "cam1/IMG_0001.MP4"
        pipeline.load_database.return_value = make_trimming_database({raw_file_path: 30.0})
        pipeline.load_all_detections.return_value = make_detections(
            raw_file_path,
            frame_ids=[10, 40, 200],
            confs=[0.9, 0.95, 0.95],
            categories=["1", "1", "3"],
        )

        process_trimming(pipeline.opt)

        assert upserted_output_df(pipeline)["trimmed_end_frame"].tolist() == [40]


class TestLinkVideo:
    """Test cases for the link_video function used when trimming is skipped."""

    def test_link_video_creates_symlink(self, tmp_path):
        input_path = tmp_path / "raw" / "IMG_0001.MP4"
        input_path.parent.mkdir()
        input_path.write_bytes(b"raw")
        output_path = tmp_path / "trimmed" / "cam1" / "cam1_IMG_0001.MP4"

        link_video(input_path, output_path, write_to_disk=True, skip_existing=False)

        assert output_path.is_symlink()
        assert output_path.resolve() == input_path.resolve()

    def test_link_video_skip_existing_true(self, tmp_path):
        input_path = tmp_path / "IMG_0001.MP4"
        input_path.write_bytes(b"raw")
        output_path = tmp_path / "cam1_IMG_0001.MP4"
        output_path.write_bytes(b"existing")

        link_video(input_path, output_path, write_to_disk=True, skip_existing=True)

        assert not output_path.is_symlink()
        assert output_path.read_bytes() == b"existing"

    def test_link_video_skip_existing_false(self, tmp_path):
        input_path = tmp_path / "IMG_0001.MP4"
        input_path.write_bytes(b"raw")
        output_path = tmp_path / "cam1_IMG_0001.MP4"
        output_path.write_bytes(b"existing")

        link_video(input_path, output_path, write_to_disk=True, skip_existing=False)

        assert output_path.is_symlink()
        assert output_path.read_bytes() == b"raw"

    def test_link_video_without_write_to_disk(self, tmp_path):
        input_path = tmp_path / "IMG_0001.MP4"
        input_path.write_bytes(b"raw")
        output_path = tmp_path / "cam1_IMG_0001.MP4"

        link_video(input_path, output_path, write_to_disk=False, skip_existing=False)

        assert not output_path.exists()


class TestProcessTrimmingSkipTrimming:
    """Test cases for process_trimming when skip_trimming is set."""

    @patch(f"{MODULE}.link_video")
    @patch(f"{MODULE}.trim_video")
    def test_skip_trimming_links_full_videos(
        self, mock_trim_video, mock_link_video, pipeline
    ):
        """Test that kept videos are linked at full length instead of being trimmed."""
        raw_file_path = "cam1/IMG_0001.MP4"
        pipeline.get_stage_parameters.return_value = make_trimming_params(
            skip_trimming=True
        )
        pipeline.load_database.return_value = make_trimming_database({raw_file_path: 30.0})
        pipeline.load_all_detections.return_value = make_detections(
            raw_file_path, frame_ids=[10, 20, 50], confs=[0.9, 0.85, 0.95]
        )

        process_trimming(pipeline.opt)

        output_video_folder = Path(PROJECT_PATH) / "trimmed_videos"
        mock_trim_video.assert_not_called()
        mock_link_video.assert_called_once_with(
            Path(RAW_VIDEOS_FOLDER) / raw_file_path,
            output_video_folder / "cam1" / "cam1_IMG_0001.MP4",
            True,
            False,
        )
        assert upserted_output_df(pipeline).to_dict("records") == [
            {
                "file_id": "cam1_IMG_0001",
                "trimmed_video_file_path": "cam1/cam1_IMG_0001.MP4",
                "trimmed_duration": VIDEO_DURATION,
                "trimmed_start_frame": 0,
                "trimmed_end_frame": int(VIDEO_DURATION * 30.0),
            }
        ]
        assert (
            pipeline.upsert_stage.call_args[1]["output_video_folder"]
            == output_video_folder
        )

    @patch(f"{MODULE}.link_video")
    @patch(f"{MODULE}.trim_video")
    def test_skip_trimming_still_filters_videos(
        self, mock_trim_video, mock_link_video, pipeline
    ):
        """Test that false positive, human and short videos are still discarded."""
        kept_video = "cam1/IMG_0001.MP4"
        human_video = "cam1/IMG_0002.MP4"
        short_video = "cam1/IMG_0003.MP4"
        false_positive_video_path = "cam1/IMG_0004.MP4"
        pipeline.get_stage_parameters.return_value = make_trimming_params(
            skip_trimming=True
        )
        pipeline.load_database.return_value = make_trimming_database(
            {
                kept_video: 30.0,
                human_video: 30.0,
                short_video: 30.0,
                false_positive_video_path: 30.0,
            }
        )
        pipeline.load_all_detections.return_value = pd.concat(
            [
                make_detections(kept_video, [10, 50], [0.9, 0.95]),
                make_detections(
                    human_video, [10, 20, 50], [0.9, 0.95, 0.95], ["1", "2", "1"]
                ),
                make_detections(short_video, [5, 20], [0.9, 0.95]),
                make_detections(
                    false_positive_video_path,
                    [10, 50],
                    [0.9, 0.95],
                    false_positive_video=True,
                ),
            ]
        )

        process_trimming(pipeline.opt)

        mock_trim_video.assert_not_called()
        mock_link_video.assert_called_once()
        assert mock_link_video.call_args[0][0] == Path(RAW_VIDEOS_FOLDER) / kept_video
        assert upserted_output_df(pipeline)["file_id"].tolist() == [
            file_id_from_path(Path(kept_video))
        ]
