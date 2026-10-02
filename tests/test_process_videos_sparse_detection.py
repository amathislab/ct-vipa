"""
Unit tests for process_videos_sparse_detection script.

Tests verify:
- Common tests (inherited from base classes)
- Videos table loaded from the database and passed to VideoDetectionProcessor
- Input folder taken from the raw_video_parser stage output
- Error handling for missing raw_video_parser output
- Error handling for empty videos dataframe
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest
from test_process_videos_detection_common import (
    BaseConfigParsingTests,
    BaseDatabaseIntegrationTests,
    BaseStageUpsertTests,
    BaseVideosTableTests,
    build_base_config,
    build_videos_df,
)

from data_processing.process_videos_sparse_detection import (
    process_videos_sparse_detection,
)

MODULE_PATH = "data_processing.process_videos_sparse_detection"
STAGE_NAME = "process_videos_sparse_detection"


class TestConfigParsing(BaseConfigParsingTests):
    """Test configuration file parsing - inherits common tests"""

    MODULE_PATH = MODULE_PATH
    STAGE_NAME = STAGE_NAME
    FUNCTION_TO_TEST = staticmethod(process_videos_sparse_detection)


class TestDatabaseIntegration(BaseDatabaseIntegrationTests):
    """Test database integration - inherits common tests"""

    MODULE_PATH = MODULE_PATH
    STAGE_NAME = STAGE_NAME
    FUNCTION_TO_TEST = staticmethod(process_videos_sparse_detection)


class TestStageUpsert(BaseStageUpsertTests):
    """Test stage upsert - inherits common tests"""

    MODULE_PATH = MODULE_PATH
    STAGE_NAME = STAGE_NAME
    FUNCTION_TO_TEST = staticmethod(process_videos_sparse_detection)


class TestVideosTable(BaseVideosTableTests):
    """Test Videos table loading - inherits common tests"""

    MODULE_PATH = MODULE_PATH
    STAGE_NAME = STAGE_NAME
    FUNCTION_TO_TEST = staticmethod(process_videos_sparse_detection)
    INPUT_VIDEO_PATH_COLUMN = "raw_file_path"


class TestInputFolder:
    """Test that sparse detection runs on the raw_video_parser output"""

    @patch(f"{MODULE_PATH}.upsert_stage")
    @patch(f"{MODULE_PATH}.VideoDetectionProcessor")
    @patch(f"{MODULE_PATH}.pd.read_sql_query")
    @patch(f"{MODULE_PATH}.get_stage_parameters")
    @patch(f"{MODULE_PATH}.get_previous_output")
    @patch(f"{MODULE_PATH}.load_database")
    @patch(f"{MODULE_PATH}.parse_config")
    def test_uses_raw_video_parser_output(
        self,
        mock_parse_config,
        mock_load_db,
        mock_get_output,
        mock_get_stage,
        mock_sql,
        mock_processor,
        mock_upsert,
    ):
        """Test that the processor receives the raw_video_parser video folder and the project path"""
        mock_config = build_base_config(STAGE_NAME)
        mock_parse_config.return_value = mock_config
        mock_conn = MagicMock()
        mock_load_db.return_value = mock_conn
        mock_get_output.return_value = "/tmp/raw_videos"
        mock_get_stage.return_value = mock_config[STAGE_NAME]
        mock_sql.return_value = build_videos_df(["cam1/IMG_0001.MP4"])

        opt = argparse.Namespace(config_file="test_config.yaml")
        process_videos_sparse_detection(opt)

        mock_get_output.assert_called_once_with(
            conn=mock_conn,
            stage_name="raw_video_parser",
            output_type="video",
        )
        init_args = mock_processor.call_args[0]
        assert init_args[0] == mock_config[STAGE_NAME]
        assert init_args[1] == "/tmp/raw_videos"
        assert init_args[2] == "/tmp/project"


class TestErrorHandling:
    """Test error handling scenarios"""

    @patch(f"{MODULE_PATH}.pd.read_sql_query")
    @patch(f"{MODULE_PATH}.get_stage_parameters")
    @patch(f"{MODULE_PATH}.get_previous_output")
    @patch(f"{MODULE_PATH}.load_database")
    @patch(f"{MODULE_PATH}.parse_config")
    def test_error_when_raw_video_parser_output_missing(
        self, mock_parse_config, mock_load_db, mock_get_output, mock_get_stage, mock_sql
    ):
        """Test that an error is raised when raw_video_parser has no output"""
        mock_config = build_base_config(STAGE_NAME)
        mock_parse_config.return_value = mock_config
        mock_load_db.return_value = MagicMock()
        mock_get_output.return_value = None
        mock_get_stage.return_value = mock_config[STAGE_NAME]
        mock_sql.return_value = build_videos_df(["cam1/IMG_0001.MP4"])

        opt = argparse.Namespace(config_file="test_config.yaml")

        with pytest.raises(ValueError, match="No output found from raw_video_parser"):
            process_videos_sparse_detection(opt)

    @patch(f"{MODULE_PATH}.upsert_stage")
    @patch(f"{MODULE_PATH}.VideoDetectionProcessor")
    @patch(f"{MODULE_PATH}.pd.read_sql_query")
    @patch(f"{MODULE_PATH}.get_stage_parameters")
    @patch(f"{MODULE_PATH}.get_previous_output")
    @patch(f"{MODULE_PATH}.load_database")
    @patch(f"{MODULE_PATH}.parse_config")
    def test_handles_empty_videos_dataframe(
        self,
        mock_parse_config,
        mock_load_db,
        mock_get_output,
        mock_get_stage,
        mock_sql,
        mock_processor,
        mock_upsert,
    ):
        """Test that an empty Videos table is handled gracefully"""
        mock_config = build_base_config(STAGE_NAME)
        mock_parse_config.return_value = mock_config
        mock_load_db.return_value = MagicMock()
        mock_get_output.return_value = "/tmp/videos"
        mock_get_stage.return_value = mock_config[STAGE_NAME]
        mock_sql.return_value = build_videos_df([])

        opt = argparse.Namespace(config_file="test_config.yaml")
        process_videos_sparse_detection(opt)

        mock_processor.assert_called_once()
        raw_videos_df = mock_processor.call_args[0][3]
        assert raw_videos_df.empty
        mock_upsert.assert_called_once()
