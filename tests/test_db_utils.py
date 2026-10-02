"""
Unit tests for data_processing/utils/db_utils.py

Tests database utility functions including:
- Database creation and schema verification
- Project registration
- Stage parameter handling
- Data upsert operations
- Query functions
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from data_processing.utils.db_utils import (
    build_database,
    ensure_project,
    get_parameters_from_config,
    get_previous_output,
    get_stage_parameters,
    load_database,
    upsert_stage,
    upsert_tracklets,
    upsert_video_data,
)


def build_test_database(tmpdir: str) -> sqlite3.Connection:
    db_path = Path(tmpdir) / "test.db"
    build_database(str(db_path))
    return sqlite3.connect(db_path)


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> set:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})")}


def trimmed_video_rows(
    file_ids: list, durations: list, start_frames: list = None, paths: list = None
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file_id": file_ids,
            "trimmed_duration": durations,
            "trimmed_start_frame": start_frames or [0] * len(file_ids),
            "trimmed_end_frame": [100] * len(file_ids),
            "trimmed_video_file_path": paths
            or [f"/{file_id}.mp4" for file_id in file_ids],
        }
    )


class TestBuildDatabase:
    """Test cases for the build_database function."""

    def test_build_database_creates_file(self):
        """Test that build_database creates a database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            build_database(str(db_path))

            assert db_path.exists()

    def test_build_database_creates_all_tables(self):
        """Test that all required tables are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            ]

            assert sorted(tables) == sorted(
                ["Project", "Stages", "Tracklets", "VideoData", "Videos"]
            )
            conn.close()

    def test_build_database_project_schema(self):
        """Test Project table has correct columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)

            assert get_table_columns(conn, "Project") == {
                "project_path",
                "config_path",
                "description",
                "created_at",
            }
            conn.close()

    def test_build_database_stages_schema(self):
        """Test Stages table has correct columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)

            assert get_table_columns(conn, "Stages") == {
                "stage_name",
                "parameters",
                "output_json_folder",
                "output_video_folder",
                "output_xml_folder",
                "examination_csv_path",
                "created_at",
            }
            conn.close()

    def test_build_database_videos_schema(self):
        """Test Videos table has correct columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)

            assert get_table_columns(conn, "Videos") == {
                "file_id",
                "group_id",
                "event_id",
                "raw_file_path",
                "date_time",
                "duration",
                "fps",
                "resolution",
                "is_IR",
            }
            conn.close()

    def test_build_database_video_data_schema(self):
        """Test VideoData table has correct columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)

            assert get_table_columns(conn, "VideoData") == {
                "file_id",
                "trimmed_duration",
                "trimmed_start_frame",
                "trimmed_end_frame",
                "trimmed_video_file_path",
                "checked",
                "individuals",
                "false_positive_video",
                "false_positive_track",
                "tracking_short",
                "segmented_track",
                "id_switch",
                "bbox_wrong",
                "tracking_pbm",
                "tracking_and_localization_correct",
                "tracking_correct",
                "description",
                "created_at",
            }
            conn.close()

    def test_build_database_tracklets_schema(self):
        """Test Tracklets table has correct columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)

            assert get_table_columns(conn, "Tracklets") == {
                "tublet_file_id",
                "file_id",
                "track_id",
                "segment_id",
                "tublet_date_time",
                "tublet_duration",
                "tublet_resolution",
                "tublet_fps",
                "tublet_start_frame",
                "tublet_end_frame",
                "tublet_file_path",
                "created_at",
            }
            conn.close()


class TestLoadDatabase:
    """Test cases for the load_database function."""

    def test_load_database_file_not_found(self):
        """Test that load_database raises error for non-existent database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_database(Path(tmpdir) / "nonexistent.db")

    def test_load_database_returns_connection_without_project(self):
        """Test that load_database returns a connection before the project is registered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            build_database(str(db_path))

            conn = load_database(db_path)

            assert isinstance(conn, sqlite3.Connection)
            conn.close()

    def test_load_database_updates_paths_when_moved(self):
        """Test that stored paths are updated when the project folder moves."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_project_path = Path(tmpdir) / "location1"
            new_project_path = Path(tmpdir) / "location2"
            old_project_path.mkdir()
            build_database(str(old_project_path / "database.db"))

            conn = sqlite3.connect(old_project_path / "database.db")
            conn.execute(
                "INSERT INTO Project (project_path, config_path, description) VALUES (?, ?, ?)",
                (str(old_project_path), "/path/to/config.yaml", "Test project"),
            )
            conn.execute(
                "INSERT INTO Stages (stage_name, parameters, output_json_folder) VALUES (?, ?, ?)",
                ("detection", "{}", str(old_project_path / "detections_json")),
            )
            conn.commit()
            conn.close()

            old_project_path.rename(new_project_path)
            conn = load_database(new_project_path / "database.db")

            stored_project_path = conn.execute(
                "SELECT project_path FROM Project"
            ).fetchone()[0]
            stored_json_folder = conn.execute(
                "SELECT output_json_folder FROM Stages WHERE stage_name = 'detection'"
            ).fetchone()[0]

            assert stored_project_path == str(new_project_path)
            assert stored_json_folder == str(new_project_path / "detections_json")
            conn.close()


class TestEnsureProject:
    """Test cases for the ensure_project function."""

    global_config = {
        "project_path": "/path/to/project",
        "config_path": "/path/to/config.yaml",
        "description": "Test project",
    }

    def test_ensure_project_inserts_project(self):
        """Test that ensure_project registers the project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)

            ensure_project(conn, self.global_config, MagicMock())

            row = conn.execute(
                "SELECT project_path, config_path, description FROM Project"
            ).fetchone()
            assert row == ("/path/to/project", "/path/to/config.yaml", "Test project")
            conn.close()

    def test_ensure_project_does_not_duplicate(self):
        """Test that ensure_project keeps a single project row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)

            ensure_project(conn, self.global_config, MagicMock())
            ensure_project(conn, self.global_config, MagicMock())

            assert conn.execute("SELECT COUNT(*) FROM Project").fetchone()[0] == 1
            conn.close()


class TestUpsertStage:
    """Test cases for the upsert_stage function."""

    def test_upsert_stage_inserts_new_stage(self):
        """Test that upsert_stage inserts a new stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)

            parameters = {"param1": "value1", "param2": 123}
            upsert_stage(
                conn,
                "detection",
                parameters,
                output_json_folder="/json",
                output_video_folder="/video",
            )

            row = conn.execute(
                "SELECT stage_name, parameters, output_json_folder, output_video_folder FROM Stages"
            ).fetchone()
            assert row[0] == "detection"
            assert json.loads(row[1]) == parameters
            assert row[2] == "/json"
            assert row[3] == "/video"
            conn.close()

    def test_upsert_stage_updates_existing_stage(self):
        """Test that upsert_stage updates an existing stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)

            upsert_stage(
                conn, "detection", {"threshold": 0.5}, output_json_folder="/json1"
            )
            upsert_stage(
                conn, "detection", {"threshold": 0.7}, output_json_folder="/json2"
            )

            rows = conn.execute(
                "SELECT parameters, output_json_folder FROM Stages WHERE stage_name = 'detection'"
            ).fetchall()
            assert len(rows) == 1
            assert json.loads(rows[0][0]) == {"threshold": 0.7}
            assert rows[0][1] == "/json2"
            conn.close()

    def test_upsert_stage_handles_none_values(self):
        """Test that upsert_stage handles None values correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)

            upsert_stage(
                conn, "detection", {}, output_json_folder=None, output_video_folder=None
            )

            row = conn.execute(
                "SELECT output_json_folder, output_video_folder FROM Stages"
            ).fetchone()
            assert row == (None, None)
            conn.close()


class TestUpsertVideoData:
    """Test cases for the upsert_video_data function."""

    def test_upsert_video_data_inserts_new_entries(self):
        """Test that upsert_video_data inserts new entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)

            upsert_video_data(
                conn,
                trimmed_video_rows(["video1", "video2"], [10.5, 20.3]),
                MagicMock(),
            )

            result_df = pd.read_sql("SELECT * FROM VideoData", conn)
            assert sorted(result_df["file_id"]) == ["video1", "video2"]
            conn.close()

    def test_upsert_video_data_overwrites_existing_entries_with_warning(self):
        """Test that upsert_video_data overwrites existing entries and logs a warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)
            mock_logger = MagicMock()

            upsert_video_data(conn, trimmed_video_rows(["video1"], [10.5]), mock_logger)
            upsert_video_data(
                conn,
                trimmed_video_rows(["video1", "video2"], [15.0, 20.0]),
                mock_logger,
            )

            result_df = pd.read_sql("SELECT * FROM VideoData", conn)
            assert len(result_df) == 2
            video1_duration = result_df[result_df["file_id"] == "video1"][
                "trimmed_duration"
            ].values[0]
            assert video1_duration == 15.0
            mock_logger.warning.assert_called_once()
            conn.close()

    def test_upsert_video_data_preserves_annotation_columns_on_overwrite(self):
        """Test that annotation columns are not overwritten when trimmed data is updated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)

            upsert_video_data(conn, trimmed_video_rows(["video1"], [10.5]), MagicMock())

            # Simulate manual annotation
            conn.execute("""
                UPDATE VideoData SET checked = 1, tracking_correct = 1, description = 'looks good'
                WHERE file_id = 'video1'
                """)
            conn.commit()

            upsert_video_data(
                conn,
                trimmed_video_rows(
                    ["video1"], [15.0], start_frames=[5], paths=["/path1_v2"]
                ),
                MagicMock(),
            )

            row = pd.read_sql(
                "SELECT * FROM VideoData WHERE file_id = 'video1'", conn
            ).iloc[0]
            # Trimmed columns should be updated
            assert row["trimmed_duration"] == 15.0
            assert row["trimmed_start_frame"] == 5
            assert row["trimmed_video_file_path"] == "/path1_v2"
            # Annotation columns should be preserved
            assert row["checked"] == 1
            assert row["tracking_correct"] == 1
            assert row["description"] == "looks good"
            conn.close()

    def test_upsert_video_data_removes_stale_entries(self):
        """Test that upsert_video_data removes entries not in new dataframe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)

            upsert_video_data(
                conn,
                trimmed_video_rows(["video1", "video2"], [10.5, 20.0]),
                MagicMock(),
            )
            upsert_video_data(conn, trimmed_video_rows(["video1"], [10.5]), MagicMock())

            result_df = pd.read_sql("SELECT * FROM VideoData", conn)
            assert list(result_df["file_id"]) == ["video1"]
            conn.close()

    def test_upsert_video_data_handles_empty_dataframe(self):
        """Test that upsert_video_data handles empty dataframe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)

            upsert_video_data(
                conn, pd.DataFrame(columns=["file_id", "trimmed_duration"]), MagicMock()
            )

            assert len(pd.read_sql("SELECT * FROM VideoData", conn)) == 0
            conn.close()


class TestUpsertTracklets:
    """Test cases for the upsert_tracklets function."""

    def test_upsert_tracklets_inserts_new_tracklets(self):
        """Test that upsert_tracklets inserts new tracklets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)

            df = pd.DataFrame(
                {
                    "tublet_file_id": ["track1", "track2"],
                    "file_id": ["video1", "video1"],
                    "track_id": [1, 2],
                    "segment_id": [0, 0],
                    "tublet_duration": [5.0, 3.5],
                }
            )
            upsert_tracklets(conn, df)

            result_df = pd.read_sql("SELECT * FROM Tracklets", conn)
            assert sorted(result_df["tublet_file_id"]) == ["track1", "track2"]
            conn.close()

    def test_upsert_tracklets_updates_existing_tracklets(self):
        """Test that upsert_tracklets updates existing tracklets on conflict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)

            def tracklet_row(duration: float) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "tublet_file_id": ["track1"],
                        "file_id": ["video1"],
                        "track_id": [1],
                        "segment_id": [0],
                        "tublet_duration": [duration],
                    }
                )

            upsert_tracklets(conn, tracklet_row(5.0))
            upsert_tracklets(conn, tracklet_row(7.5))

            result_df = pd.read_sql("SELECT * FROM Tracklets", conn)
            assert len(result_df) == 1
            assert result_df["tublet_duration"].values[0] == 7.5
            conn.close()


class TestGetPreviousOutput:
    """Test cases for the get_previous_output function."""

    @pytest.mark.parametrize(
        "column, output_type",
        [
            ("output_json_folder", "json"),
            ("output_video_folder", "video"),
            ("examination_csv_path", "csv"),
        ],
    )
    def test_get_previous_output_returns_stored_path(self, column, output_type):
        """Test that get_previous_output returns the stored path of each output type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)
            conn.execute(
                f"INSERT INTO Stages (stage_name, parameters, {column}) VALUES (?, ?, ?)",
                ("stage", "{}", "/path/to/output"),
            )
            conn.commit()

            assert get_previous_output(conn, "stage", output_type) == "/path/to/output"
            conn.close()

    def test_get_previous_output_returns_none_on_missing_stage(self):
        """Test that get_previous_output returns None when the stage is not in the database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)

            assert get_previous_output(conn, "nonexistent_stage", "json") is None
            conn.close()

    def test_get_previous_output_returns_none_on_missing_output(self):
        """Test that get_previous_output returns None when the stage has no output of that type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)
            conn.execute(
                "INSERT INTO Stages (stage_name, parameters, output_json_folder) VALUES (?, ?, ?)",
                ("detection", "{}", "/path/to/json"),
            )
            conn.commit()

            assert get_previous_output(conn, "detection", "video") is None
            conn.close()

    def test_get_previous_output_raises_on_invalid_type(self):
        """Test that get_previous_output raises ValueError for invalid output type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = build_test_database(tmpdir)

            with pytest.raises(ValueError):
                get_previous_output(conn, "detection", "invalid_type")
            conn.close()


class TestGetStageParameters:
    """Test cases for the get_stage_parameters function."""

    def test_get_stage_parameters_returns_correct_dict(self):
        """Test that get_stage_parameters returns the correct parameter dictionary."""
        config = {
            "global": {"project_path": "/path"},
            "detection": {"threshold": 0.5, "model": "mdv5"},
            "tracking": {"max_age": 30},
        }

        result = get_stage_parameters(config, "detection")

        assert result == {"threshold": 0.5, "model": "mdv5"}

    def test_get_stage_parameters_raises_on_missing_script(self):
        """Test that get_stage_parameters raises ValueError for missing script."""
        config = {
            "global": {"project_path": "/path"},
            "detection": {"threshold": 0.5},
        }

        with pytest.raises(ValueError, match="Script name 'nonexistent' not found"):
            get_stage_parameters(config, "nonexistent")

    def test_get_stage_parameters_handles_empty_params(self):
        """Test that get_stage_parameters handles empty parameter dict."""
        config = {"global": {"project_path": "/path"}, "detection": {}}

        result = get_stage_parameters(config, "detection")

        assert result == {}


class TestGetParametersFromConfig:
    """Test cases for the get_parameters_from_config function."""

    def test_get_parameters_from_config_returns_value(self):
        """Test that get_parameters_from_config returns the correct value."""
        config = {"threshold": 0.5, "model": "mdv5", "nested": {"param": "value"}}

        result = get_parameters_from_config(config, "threshold")

        assert result == 0.5

    def test_get_parameters_from_config_returns_nested_dict(self):
        """Test that get_parameters_from_config returns nested dictionary."""
        config = {"nested": {"param1": "value1", "param2": 123}}

        result = get_parameters_from_config(config, "nested")

        assert result == {"param1": "value1", "param2": 123}

    def test_get_parameters_from_config_raises_on_missing_key(self):
        """Test that get_parameters_from_config raises ValueError for missing key."""
        config = {"threshold": 0.5}

        with pytest.raises(ValueError, match="Key 'nonexistent' not found"):
            get_parameters_from_config(config, "nonexistent")
