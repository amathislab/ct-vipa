"""
Unit tests for RawVideoParser class.

Tests verify:
- file_id is the video's relative path flattened with "_", without extension
- Colliding file_ids are rejected
- group_id is the video's containing folder
- Unique event_id per group_id, split on inactivity gaps
- DataFrame has required columns
- Database insertion and update of videos
"""

import sqlite3
from pathlib import Path
from typing import Dict, Iterator
from unittest.mock import MagicMock, mock_open, patch

import cv2
import numpy as np
import pandas as pd
import pytest

from data_processing.raw_video_parser import RawVideoParser
from data_processing.utils.db_utils import build_database, file_id_from_path


def make_video_meta(
    raw_file_path: str, date_time: str, duration: float, is_IR: bool = False
) -> Dict:
    return {
        "raw_file_path": raw_file_path,
        "file_id": file_id_from_path(Path(raw_file_path)),
        "group_id": Path(raw_file_path).parent.as_posix(),
        "date_time": pd.Timestamp(date_time).timestamp(),
        "duration": duration,
        "fps": 30.0,
        "resolution": "1920x1080",
        "is_IR": is_IR,
    }


class TestRawVideoParserDataFrame:
    """Test suite for DataFrame generation and validation"""

    @pytest.fixture
    def sample_results(self) -> Dict[str, Dict]:
        """Create sample parsed results mimicking the flat dictionary keyed by relative path"""
        videos = [
            make_video_meta("Site1/CAM001/IMG_0100.MP4", "2021-01-01 00:00:00", 12.0),
            # different event (> 300s later)
            make_video_meta("Site1/CAM001/IMG_0101.MP4", "2021-01-01 00:36:40", 12.0),
            make_video_meta("Site1/CAM002/IMG_0100.MP4", "2021-01-01 00:00:00", 10.0),
            make_video_meta("Site3/CAM003/IMG_2452.MP4", "2021-01-01 00:00:00", 10.0),
            # 20 seconds later
            make_video_meta("Site3/CAM003/IMG_2453.MP4", "2021-01-01 00:00:20", 10.0),
            make_video_meta(
                "Site3/CAM002/IMG_0001.MP4", "2021-01-01 00:05:00", 15.0, is_IR=True
            ),
        ]
        return {video["raw_file_path"]: video for video in videos}

    @pytest.fixture
    def parser_with_results(self, sample_results: Dict[str, Dict]) -> RawVideoParser:
        """Create parser with sample results"""
        parser = RawVideoParser(event_time_period_s=300)
        parser.results = sample_results
        return parser

    def test_dataframe_has_required_columns(
        self, parser_with_results: RawVideoParser
    ) -> None:
        """Test that the final DataFrame contains all required columns"""
        df = parser_with_results.make_dataframe()

        required_columns = [
            "file_id",
            "group_id",
            "event_id",
            "raw_file_path",
            "date_time",
            "duration",
            "fps",
            "resolution",
            "is_IR",
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

    def test_file_id_is_flattened_raw_file_path(
        self, parser_with_results: RawVideoParser
    ) -> None:
        """Test that file_id is the video's relative path flattened with underscores"""
        df = parser_with_results.make_dataframe()

        row = df[df["raw_file_path"] == "Site3/CAM003/IMG_2452.MP4"].iloc[0]

        assert row["file_id"] == "Site3_CAM003_IMG_2452"
        assert row["group_id"] == "Site3/CAM003"

    def test_colliding_file_ids_raise(self) -> None:
        """Test that paths flattening to the same file_id are rejected"""
        videos = [
            make_video_meta("a_b/c.mp4", "2021-01-01 00:00:00", 10.0),
            make_video_meta("a/b_c.mp4", "2021-01-01 00:00:00", 10.0),
        ]
        parser = RawVideoParser()
        parser.results = {video["raw_file_path"]: video for video in videos}

        with pytest.raises(ValueError, match="a_b_c"):
            parser.make_dataframe()

    def test_same_event_id_same_group_within_time_threshold(
        self, parser_with_results: RawVideoParser
    ) -> None:
        """Test that videos from the same group within time threshold get same event_id"""
        df = parser_with_results.make_dataframe()

        # Videos IMG_2452 and IMG_2453 from Site3/CAM003 are 20s apart, should have same event_id
        video1 = df[df["raw_file_path"] == "Site3/CAM003/IMG_2452.MP4"].iloc[0]
        video2 = df[df["raw_file_path"] == "Site3/CAM003/IMG_2453.MP4"].iloc[0]

        assert (
            video1["event_id"] == video2["event_id"]
        ), "Videos within time threshold should have same event_id"

    def test_event_id_uniqueness_per_group(
        self, parser_with_results: RawVideoParser
    ) -> None:
        """Test that each event_id belongs to a single group"""
        df = parser_with_results.make_dataframe()

        groups_per_event = df.groupby("event_id")["group_id"].nunique()
        assert (groups_per_event == 1).all(), "An event_id spans several groups"

    def test_event_id_changes_with_time_gap(
        self, parser_with_results: RawVideoParser
    ) -> None:
        """Test that event_id changes when time gap > event_time_period_s"""
        df = parser_with_results.make_dataframe()

        # Site1/CAM001 has videos 2200 seconds apart (> 300s threshold)
        cam001_df = df[df["group_id"] == "Site1/CAM001"]

        event_ids = cam001_df["event_id"].tolist()
        assert (
            len(set(event_ids)) == 2
        ), "Videos > 300s apart should have different event_ids"

    def test_event_id_changes_with_new_group(
        self, parser_with_results: RawVideoParser
    ) -> None:
        """Test that simultaneous videos from different groups get different event_ids"""
        df = parser_with_results.make_dataframe()

        cam001_events = set(df[df["group_id"] == "Site1/CAM001"]["event_id"])
        cam002_events = set(df[df["group_id"] == "Site1/CAM002"]["event_id"])

        assert (
            len(cam001_events & cam002_events) == 0
        ), "Different groups should have different event_ids"

    def test_event_id_zero_padding(self, parser_with_results: RawVideoParser) -> None:
        """Test that event_id has correct zero padding (4 digits)"""
        df = parser_with_results.make_dataframe()

        for event_id in df["event_id"]:
            # Should be E followed by 4 digits
            assert event_id.startswith("E")
            assert len(event_id) == 5  # E + 4 digits
            assert event_id[1:].isdigit()

    def test_no_duplicate_file_ids(self, parser_with_results: RawVideoParser) -> None:
        """Test that all file_ids are unique"""
        df = parser_with_results.make_dataframe()

        file_ids = df["file_id"].tolist()
        assert len(file_ids) == len(set(file_ids)), "Duplicate file_ids found"

    def test_is_ir_column_type(self, parser_with_results: RawVideoParser) -> None:
        """Test that is_IR column contains boolean values"""
        df = parser_with_results.make_dataframe()

        for is_ir in df["is_IR"]:
            assert isinstance(is_ir, (bool, np.bool_))


class TestRawVideoParserFolderParsing:
    """Test suite for recursive folder parsing"""

    def test_parse_folder_keys_and_group_ids(self, tmp_path: Path) -> None:
        """Test that videos are keyed by relative path and grouped by containing folder"""
        for relative_path in [
            "cam1/IMG_0001.MP4",
            "cam1/IMG_0002.mp4",
            "site/cam2/clip.mp4",
            "root_clip.mp4",
            "cam1/notes.txt",
        ]:
            video_path = tmp_path / relative_path
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.touch()

        parser = RawVideoParser()
        with (
            patch("data_processing.raw_video_parser.open", mock_open(), create=True),
            patch("pickle.load"),
            patch.object(
                RawVideoParser,
                "parse_file",
                side_effect=lambda raw_video, svm_clf: {"date_time": 0.0},
            ),
        ):
            results = parser.parse_folder(tmp_path, video_extensions=[".mp4"])

        assert sorted(results) == [
            "cam1/IMG_0001.MP4",
            "cam1/IMG_0002.mp4",
            "root_clip.mp4",
            "site/cam2/clip.mp4",
        ]
        assert results["cam1/IMG_0001.MP4"]["group_id"] == "cam1"
        assert results["site/cam2/clip.mp4"]["group_id"] == "site/cam2"
        assert results["root_clip.mp4"]["group_id"] == "root"
        assert results["site/cam2/clip.mp4"]["raw_file_path"] == "site/cam2/clip.mp4"
        assert results["site/cam2/clip.mp4"]["file_id"] == "site_cam2_clip"


class TestRawVideoParserTimeCalculations:
    """Test suite for time-related calculations"""

    def test_get_event_id_single_group_continuous(self) -> None:
        """Test event ID generation for continuous videos in single group"""
        parser = RawVideoParser(event_time_period_s=300)

        df = pd.DataFrame(
            {
                "group_id": ["cam1", "cam1", "cam1"],
                "date_time": [100, 200, 280],  # All within 300s
                "end_time": [110, 210, 290],
            }
        )

        result = parser.get_event_id(df)

        # All should have same event_id
        event_ids = result["event_id"].unique()
        assert len(event_ids) == 1
        assert event_ids[0] == "E0001"

    def test_get_event_id_single_group_with_gap(self) -> None:
        """Test event ID generation with time gap > threshold"""
        parser = RawVideoParser(event_time_period_s=300)

        df = pd.DataFrame(
            {
                "group_id": ["cam1", "cam1", "cam1"],
                "date_time": [100, 200, 600],  # Last one > 300s after previous
                "end_time": [110, 210, 610],
            }
        )

        result = parser.get_event_id(df)

        # First two should have E0001, last should have E0002
        assert result.iloc[0]["event_id"] == "E0001"
        assert result.iloc[1]["event_id"] == "E0001"
        assert result.iloc[2]["event_id"] == "E0002"

    def test_get_event_id_gap_measured_from_previous_end_time(self) -> None:
        """Test that the inactivity gap starts when the previous video ends, not when it starts"""
        parser = RawVideoParser(event_time_period_s=300)

        df = pd.DataFrame(
            {
                "group_id": ["cam1", "cam1"],
                "date_time": [0, 450],  # 450s between starts
                "end_time": [200, 460],  # but only 250s of inactivity
            }
        )

        result = parser.get_event_id(df)

        assert result["event_id"].tolist() == ["E0001", "E0001"]

    def test_get_event_id_multiple_groups(self) -> None:
        """Test event ID generation across multiple groups"""
        parser = RawVideoParser(event_time_period_s=300)

        df = pd.DataFrame(
            {
                "group_id": ["cam1", "cam1", "cam2", "cam2"],
                "date_time": [100, 200, 150, 250],
                "end_time": [110, 210, 160, 260],
            }
        )

        result = parser.get_event_id(df)

        # cam1 should have E0001
        cam1_events = result[result["group_id"] == "cam1"]["event_id"].unique()
        assert len(cam1_events) == 1
        assert cam1_events[0] == "E0001"

        # cam2 should have E0002 (new group = new event)
        cam2_events = result[result["group_id"] == "cam2"]["event_id"].unique()
        assert len(cam2_events) == 1
        assert cam2_events[0] == "E0002"


class TestRawVideoParserDatabaseOperations:
    """Test suite for database operations"""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Iterator[sqlite3.Connection]:
        """Create a temporary database with the project schema"""
        db_path = tmp_path / "test.db"
        build_database(str(db_path))
        conn = sqlite3.connect(db_path)
        yield conn
        conn.close()

    @pytest.fixture
    def raw_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "file_id": ["cam1_IMG_0001", "cam1_IMG_0002"],
                "group_id": ["cam1", "cam1"],
                "event_id": ["E0001", "E0001"],
                "raw_file_path": ["cam1/IMG_0001.MP4", "cam1/IMG_0002.MP4"],
                "date_time": ["2021-01-01 00:00:00", "2021-01-01 00:00:20"],
                "duration": [10.0, 10.0],
                "fps": [30.0, 30.0],
                "resolution": ["1920x1080", "1920x1080"],
                "is_IR": [False, False],
            }
        )

    def test_save_to_db_new_videos(
        self, temp_db: sqlite3.Connection, raw_df: pd.DataFrame
    ) -> None:
        """Test saving new videos to database"""
        parser = RawVideoParser()
        parser.raw_df = raw_df

        parser.save_to_db(temp_db)

        result = pd.read_sql("SELECT * FROM Videos ORDER BY file_id", temp_db)
        assert len(result) == 2
        assert result.iloc[0]["file_id"] == "cam1_IMG_0001"
        assert result.iloc[0]["group_id"] == "cam1"
        assert result.iloc[0]["raw_file_path"] == "cam1/IMG_0001.MP4"

    def test_save_to_db_update_existing(
        self, temp_db: sqlite3.Connection, raw_df: pd.DataFrame
    ) -> None:
        """Test updating existing videos in database"""
        parser = RawVideoParser()
        parser.raw_df = raw_df.iloc[:1].copy()
        parser.save_to_db(temp_db)

        # Re-parse with new metadata
        parser.raw_df["is_IR"] = True
        parser.raw_df["duration"] = 15.0
        parser.raw_df["event_id"] = "E0002"

        parser.save_to_db(temp_db)

        result = pd.read_sql("SELECT * FROM Videos", temp_db)
        assert len(result) == 1
        assert result.iloc[0]["is_IR"] == 1  # True stored as 1
        assert result.iloc[0]["duration"] == 15.0
        assert result.iloc[0]["event_id"] == "E0002"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
