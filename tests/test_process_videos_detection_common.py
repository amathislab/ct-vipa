"""
Common test utilities for both dense and sparse video detection scripts.

This module contains base test classes and helper functions shared between:
- test_process_videos_dense_detection.py
- test_process_videos_sparse_detection.py

This reduces duplication and ensures consistent testing across both scripts.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_processing.utils.db_utils import file_id_from_path, output_relative_path


def build_videos_df(raw_file_paths: List[str]) -> pd.DataFrame:
    """Build a Videos table as returned by the database, with the trimmed video path of each video"""
    return pd.DataFrame(
        {
            "file_id": [file_id_from_path(Path(path)) for path in raw_file_paths],
            "group_id": [Path(path).parent.as_posix() for path in raw_file_paths],
            "event_id": [f"event_{i}" for i in range(len(raw_file_paths))],
            "raw_file_path": raw_file_paths,
            "is_IR": [False] * len(raw_file_paths),
            "trimmed_video_file_path": [
                str(output_relative_path(path, Path(path).suffix))
                for path in raw_file_paths
            ],
        }
    )


def build_base_config(stage_name: str) -> Dict[str, Any]:
    return {
        "global": {
            "project_path": "/tmp/project",
            "database_path": "/tmp/test.db",
        },
        stage_name: {
            "model": "MDV6-yolov10-e",
            "json_confidence_threshold": 0.1,
            "output_json_folder_name": "detections",
            "tmp_frame_folder_name": "frames",
            "skip_existing": False,
            "force_frame_folder_deletion": False,
            "detector_label_map": {"0": "animal"},
        },
        "video_extensions": [".mp4"],
    }


class BaseConfigParsingTests:
    """Base class for config parsing tests - to be inherited by both dense and sparse tests"""

    # These should be overridden by subclasses
    MODULE_PATH = None
    STAGE_NAME = None
    FUNCTION_TO_TEST = None

    def test_missing_config_file_raises_error(self):
        """Test that missing config file raises an error"""
        with patch(f"{self.MODULE_PATH}.parse_config") as mock_parse_config:
            mock_parse_config.side_effect = FileNotFoundError(
                "Configuration file not found"
            )

            opt = argparse.Namespace(config_file="nonexistent.yaml")

            with pytest.raises(FileNotFoundError, match="Configuration file not found"):
                self.FUNCTION_TO_TEST(opt)


class BaseDatabaseIntegrationTests:
    """Base class for database integration tests"""

    MODULE_PATH = None
    STAGE_NAME = None
    FUNCTION_TO_TEST = None

    def test_database_connection_closed(self):
        """Test that database connection is properly closed"""
        mock_config = build_base_config(self.STAGE_NAME)

        with (
            patch(f"{self.MODULE_PATH}.parse_config") as mock_parse_config,
            patch(f"{self.MODULE_PATH}.load_database") as mock_load_db,
            patch(f"{self.MODULE_PATH}.get_previous_output") as mock_get_output,
            patch(f"{self.MODULE_PATH}.get_stage_parameters") as mock_get_stage,
            patch(
                f"{self.MODULE_PATH}.pd.read_sql_query",
                return_value=build_videos_df(["cam1/IMG_0001.MP4"]),
            ),
            patch(f"{self.MODULE_PATH}.VideoDetectionProcessor"),
            patch(f"{self.MODULE_PATH}.upsert_stage"),
        ):
            mock_parse_config.return_value = mock_config
            mock_conn = MagicMock()
            mock_load_db.return_value = mock_conn
            mock_get_output.return_value = "/tmp/videos"
            mock_get_stage.return_value = mock_config[self.STAGE_NAME]

            opt = argparse.Namespace(config_file="test_config.yaml")
            self.FUNCTION_TO_TEST(opt)

            mock_conn.close.assert_called_once()


class BaseStageUpsertTests:
    """Base class for stage upsert tests"""

    MODULE_PATH = None
    STAGE_NAME = None
    FUNCTION_TO_TEST = None

    def test_stage_upserted_with_correct_parameters(self):
        """Test that the stage is upserted with correct parameters"""
        mock_config = build_base_config(self.STAGE_NAME)

        with (
            patch(f"{self.MODULE_PATH}.parse_config") as mock_parse_config,
            patch(f"{self.MODULE_PATH}.load_database") as mock_load_db,
            patch(f"{self.MODULE_PATH}.get_previous_output") as mock_get_output,
            patch(f"{self.MODULE_PATH}.get_stage_parameters") as mock_get_stage,
            patch(
                f"{self.MODULE_PATH}.pd.read_sql_query",
                return_value=build_videos_df(["cam1/IMG_0001.MP4"]),
            ),
            patch(f"{self.MODULE_PATH}.VideoDetectionProcessor") as mock_processor,
            patch(f"{self.MODULE_PATH}.upsert_stage") as mock_upsert,
        ):
            mock_parse_config.return_value = mock_config
            mock_conn = MagicMock()
            mock_load_db.return_value = mock_conn
            mock_get_output.return_value = "/tmp/videos"
            mock_get_stage.return_value = mock_config[self.STAGE_NAME]

            mock_processor_instance = MagicMock()
            mock_processor_instance.output_json_folder = Path("/tmp/output_json")
            mock_processor.return_value = mock_processor_instance

            opt = argparse.Namespace(config_file="test_config.yaml")
            self.FUNCTION_TO_TEST(opt)

            mock_upsert.assert_called_once_with(
                conn=mock_conn,
                stage_name=self.STAGE_NAME,
                parameters_dict=mock_config[self.STAGE_NAME],
                output_json_folder=mock_processor_instance.output_json_folder,
                output_video_folder=None,
                examination_csv_path=None,
            )


class BaseVideosTableTests:
    """Base class for tests on how the Videos table is loaded and passed to the processor"""

    MODULE_PATH = None
    STAGE_NAME = None
    FUNCTION_TO_TEST = None
    # Column giving the path of each input video relative to the input folder
    INPUT_VIDEO_PATH_COLUMN = None

    def test_videos_table_passed_to_processor(self):
        """Test that the Videos table is queried and passed to VideoDetectionProcessor as raw_videos_df"""
        mock_config = build_base_config(self.STAGE_NAME)
        videos_df = build_videos_df(["cam1/IMG_0001.MP4", "cam2/IMG_0002.MP4"])

        with (
            patch(f"{self.MODULE_PATH}.parse_config") as mock_parse_config,
            patch(f"{self.MODULE_PATH}.load_database") as mock_load_db,
            patch(f"{self.MODULE_PATH}.get_previous_output") as mock_get_output,
            patch(f"{self.MODULE_PATH}.get_stage_parameters") as mock_get_stage,
            patch(
                f"{self.MODULE_PATH}.pd.read_sql_query", return_value=videos_df
            ) as mock_sql,
            patch(f"{self.MODULE_PATH}.VideoDetectionProcessor") as mock_processor,
            patch(f"{self.MODULE_PATH}.upsert_stage"),
        ):
            mock_parse_config.return_value = mock_config
            mock_conn = MagicMock()
            mock_load_db.return_value = mock_conn
            mock_get_output.return_value = "/tmp/videos"
            mock_get_stage.return_value = mock_config[self.STAGE_NAME]

            opt = argparse.Namespace(config_file="test_config.yaml")
            self.FUNCTION_TO_TEST(opt)

        mock_sql.assert_called_once()
        sql_query, sql_conn = mock_sql.call_args[0]
        assert "FROM Videos" in sql_query
        assert sql_conn is mock_conn

        mock_processor.assert_called_once()
        init_args = mock_processor.call_args[0]
        raw_videos_df = (
            init_args[3]
            if len(init_args) > 3
            else mock_processor.call_args[1]["raw_videos_df"]
        )
        assert raw_videos_df["file_id"].tolist() == ["cam1_IMG_0001", "cam2_IMG_0002"]
        assert raw_videos_df["video_path"].tolist() == raw_videos_df[
            self.INPUT_VIDEO_PATH_COLUMN
        ].tolist()

    def test_video_extensions_passed_to_processor(self):
        """Test that video_extensions from the config are forwarded to process_videos_folder"""
        mock_config = build_base_config(self.STAGE_NAME)
        mock_config["video_extensions"] = [".MP4", ".avi"]

        with (
            patch(f"{self.MODULE_PATH}.parse_config") as mock_parse_config,
            patch(f"{self.MODULE_PATH}.load_database"),
            patch(f"{self.MODULE_PATH}.get_previous_output") as mock_get_output,
            patch(f"{self.MODULE_PATH}.get_stage_parameters") as mock_get_stage,
            patch(
                f"{self.MODULE_PATH}.pd.read_sql_query",
                return_value=build_videos_df(["cam1/IMG_0001.MP4"]),
            ),
            patch(f"{self.MODULE_PATH}.VideoDetectionProcessor") as mock_processor,
            patch(f"{self.MODULE_PATH}.upsert_stage"),
        ):
            mock_parse_config.return_value = mock_config
            mock_get_output.return_value = "/tmp/videos"
            mock_get_stage.return_value = mock_config[self.STAGE_NAME]

            opt = argparse.Namespace(config_file="test_config.yaml")
            self.FUNCTION_TO_TEST(opt)

        mock_processor.return_value.process_videos_folder.assert_called_once_with(
            detection_categories={"0": "animal"},
            video_extensions=[".MP4", ".avi"],
        )
