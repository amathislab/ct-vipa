"""
Unit tests for process_videos_dense_detection script.

Tests verify:
- Common tests (inherited from base classes)
- Folder fallback logic (trimmed_from_detection -> raw_video_parser)
- Error handling for missing stages
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

from data_processing.process_videos_dense_detection import (
    process_videos_dense_detection,
)

MODULE_PATH = "data_processing.process_videos_dense_detection"
STAGE_NAME = "process_videos_dense_detection"


class TestConfigParsing(BaseConfigParsingTests):
    """Test configuration file parsing - inherits common tests"""

    MODULE_PATH = MODULE_PATH
    STAGE_NAME = STAGE_NAME
    FUNCTION_TO_TEST = staticmethod(process_videos_dense_detection)


class TestDatabaseIntegration(BaseDatabaseIntegrationTests):
    """Test database integration - inherits common tests"""

    MODULE_PATH = MODULE_PATH
    STAGE_NAME = STAGE_NAME
    FUNCTION_TO_TEST = staticmethod(process_videos_dense_detection)


class TestStageUpsert(BaseStageUpsertTests):
    """Test stage upsert - inherits common tests"""

    MODULE_PATH = MODULE_PATH
    STAGE_NAME = STAGE_NAME
    FUNCTION_TO_TEST = staticmethod(process_videos_dense_detection)


class TestVideosTable(BaseVideosTableTests):
    """Test Videos table loading - inherits common tests"""

    MODULE_PATH = MODULE_PATH
    STAGE_NAME = STAGE_NAME
    FUNCTION_TO_TEST = staticmethod(process_videos_dense_detection)
    INPUT_VIDEO_PATH_COLUMN = "trimmed_video_file_path"


class TestFolderFallbackLogic:
    """Test input folder fallback from trimmed_from_detection to raw_video_parser"""

    @patch(f"{MODULE_PATH}.upsert_stage")
    @patch(f"{MODULE_PATH}.VideoDetectionProcessor")
    @patch(f"{MODULE_PATH}.pd.read_sql_query")
    @patch(f"{MODULE_PATH}.get_stage_parameters")
    @patch(f"{MODULE_PATH}.get_previous_output")
    @patch(f"{MODULE_PATH}.load_database")
    @patch(f"{MODULE_PATH}.parse_config")
    def test_uses_trimmed_from_detection_when_available(
        self,
        mock_parse_config,
        mock_load_db,
        mock_get_output,
        mock_get_stage,
        mock_sql,
        mock_processor,
        mock_upsert,
    ):
        """Test that trimmed_from_detection output is used when available"""
        mock_config = build_base_config(STAGE_NAME)
        mock_parse_config.return_value = mock_config
        mock_load_db.return_value = MagicMock()
        mock_get_output.return_value = "/tmp/trimmed_videos"
        mock_get_stage.return_value = mock_config[STAGE_NAME]
        mock_sql.return_value = build_videos_df(["cam1/IMG_0001.MP4"])

        opt = argparse.Namespace(config_file="test_config.yaml")
        process_videos_dense_detection(opt)

        mock_processor.assert_called_once()
        assert mock_processor.call_args[0][1] == "/tmp/trimmed_videos"
        assert mock_get_output.call_count == 1

    @patch(f"{MODULE_PATH}.upsert_stage")
    @patch(f"{MODULE_PATH}.VideoDetectionProcessor")
    @patch(f"{MODULE_PATH}.pd.read_sql_query")
    @patch(f"{MODULE_PATH}.get_stage_parameters")
    @patch(f"{MODULE_PATH}.get_previous_output")
    @patch(f"{MODULE_PATH}.load_database")
    @patch(f"{MODULE_PATH}.parse_config")
    def test_falls_back_to_raw_parser_when_trimmed_missing(
        self,
        mock_parse_config,
        mock_load_db,
        mock_get_output,
        mock_get_stage,
        mock_sql,
        mock_processor,
        mock_upsert,
    ):
        """Test that raw_video_parser output is used when trimmed_from_detection is not available"""
        mock_config = build_base_config(STAGE_NAME)
        mock_parse_config.return_value = mock_config
        mock_load_db.return_value = MagicMock()
        mock_get_stage.return_value = mock_config[STAGE_NAME]
        mock_sql.return_value = build_videos_df(["cam1/IMG_0001.MP4"])

        def get_output_side_effect(conn, stage_name: str, output_type: str):
            if stage_name == "raw_video_parser":
                return "/tmp/raw_videos"
            return None

        mock_get_output.side_effect = get_output_side_effect

        opt = argparse.Namespace(config_file="test_config.yaml")
        process_videos_dense_detection(opt)

        mock_processor.assert_called_once()
        assert mock_processor.call_args[0][1] == "/tmp/raw_videos"
        queried_stages = [
            call.kwargs["stage_name"] for call in mock_get_output.call_args_list
        ]
        assert queried_stages == ["trimmed_from_detection", "raw_video_parser"]

    @patch(f"{MODULE_PATH}.get_stage_parameters")
    @patch(f"{MODULE_PATH}.get_previous_output")
    @patch(f"{MODULE_PATH}.load_database")
    @patch(f"{MODULE_PATH}.parse_config")
    def test_error_when_no_input_folder_available(
        self, mock_parse_config, mock_load_db, mock_get_output, mock_get_stage
    ):
        """Test that an error is raised when neither stage has output"""
        mock_config = build_base_config(STAGE_NAME)
        mock_parse_config.return_value = mock_config
        mock_load_db.return_value = MagicMock()
        mock_get_output.return_value = None
        mock_get_stage.return_value = mock_config[STAGE_NAME]

        opt = argparse.Namespace(config_file="test_config.yaml")

        with pytest.raises(ValueError, match="No output found from either"):
            process_videos_dense_detection(opt)
