"""
Unit tests for prepare_videos_for_tracking_examination script.

Tests verify:
- generate_excel_ready_csv creates CSV from database VideoData
- CSV sorting by file_id
- Merging with existing CSV to preserve existing rows
- Duplicate handling based on file_id
"""

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from data_processing.prepare_videos_for_tracking_examination import (
    generate_excel_ready_csv,
)


class TestGenerateExcelReadyCsv:
    """Test Excel-ready CSV generation from database"""

    def test_creates_csv_from_database(self):
        """Test CSV is created with data from VideoData table"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE VideoData (
                    file_id TEXT,
                    site_cam_id TEXT                
                    )
                """
            )

            cursor.execute(
                "INSERT INTO VideoData VALUES ('video1', 'Site1_CAM001')"
            )
            cursor.execute(
                "INSERT INTO VideoData VALUES ('video2', 'Site1_CAM002')"
            )
            conn.commit()

            output_csv = Path(tmpdir) / "output.csv"
            generate_excel_ready_csv(conn, str(output_csv))

            assert output_csv.exists()

            df = pd.read_csv(output_csv)
            assert len(df) == 2
            assert "file_id" in df.columns
            assert "site_cam_id" in df.columns
            assert list(df["file_id"]) == ["video1", "video2"]

            conn.close()

    def test_sorts_by_file_id(self):
        """Test that CSV is sorted by file_id and not insertion order"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE VideoData (
                    file_id TEXT,
                    value INTEGER
                )
                """
            )

            # Insert data in random order
            cursor.execute(
                "INSERT INTO VideoData VALUES ('video3', 20)"
            )
            cursor.execute(
                "INSERT INTO VideoData VALUES ('video1', 10)"
            )
            cursor.execute(
                "INSERT INTO VideoData VALUES ('video2', 30)"
            )
            conn.commit()

            output_csv = Path(tmpdir) / "output.csv"
            generate_excel_ready_csv(conn, str(output_csv))

            df = pd.read_csv(output_csv)

            assert list(df["file_id"]) == ["video1", "video2", "video3"]
            assert list(df["value"]) == [10, 30, 20]

            conn.close()

    def test_merges_with_existing_csv(self):
        """Test that existing CSV is merged with new data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE VideoData (
                    file_id TEXT,
                    value INTEGER
                )
                """
            )

            # Insert new data
            cursor.execute(
                "INSERT INTO VideoData VALUES ('video2', 20)"
            )
            cursor.execute(
                "INSERT INTO VideoData VALUES ('video3', 30)"
            )
            conn.commit()

            # Create existing CSV
            output_csv = Path(tmpdir) / "output.csv"
            existing_df = pd.DataFrame(
                {"file_id": ["video1", "video2"], "value": [10, 200]}
            )
            existing_df.to_csv(output_csv, index=False)

            # Generate CSV (should merge)
            generate_excel_ready_csv(conn, str(output_csv))

            df = pd.read_csv(output_csv)

            # Should have 3 unique videos (video2 should not be duplicated)
            assert len(df) == 3
            assert set(df["file_id"]) == {"video1", "video2", "video3"}

            # Existing video1 should be preserved
            video1_row = df[df["file_id"] == "video1"]
            assert video1_row["value"].values[0] == 10

            conn.close()

    def test_removes_duplicates_by_file_id(self):
        """Test that duplicates are removed based on file_id"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE VideoData (
                    file_id TEXT,
                    value INTEGER
                )
                """
            )

            # Insert duplicate file_ids in database
            cursor.execute(
                "INSERT INTO VideoData VALUES ('video1', 10)"
            )
            cursor.execute(
                "INSERT INTO VideoData VALUES ('video1', 15)"
            )
            conn.commit()

            # Create existing CSV with same file_id
            output_csv = Path(tmpdir) / "output.csv"
            existing_df = pd.DataFrame({"file_id": ["video1"], "value": [5]})
            existing_df.to_csv(output_csv, index=False)

            # Generate CSV (should remove duplicates)
            generate_excel_ready_csv(conn, str(output_csv))

            df = pd.read_csv(output_csv)

            # Should have only 1 row (duplicates removed)
            assert len(df) == 1
            assert df["file_id"].values[0] == "video1"

            conn.close()

    def test_preserves_all_columns(self):
        """Test that all columns from VideoData are preserved"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE VideoData (
                    file_id TEXT,
                    site_cam_id TEXT,
                    detection_count INTEGER,
                    tracking_count INTEGER,
                    custom_field TEXT
                )
                """
            )

            cursor.execute(
                "INSERT INTO VideoData VALUES ('video1', 'Site1_CAM001', 5, 3, 'test')"
            )
            conn.commit()

            output_csv = Path(tmpdir) / "output.csv"
            generate_excel_ready_csv(conn, str(output_csv))

            df = pd.read_csv(output_csv)

            expected_columns = [
                "file_id",
                "site_cam_id",
                "detection_count",
                "tracking_count",
                "custom_field",
            ]
            for col in expected_columns:
                assert col in df.columns

            assert df["site_cam_id"].values[0] == "Site1_CAM001"
            assert df["detection_count"].values[0] == 5
            assert df["tracking_count"].values[0] == 3
            assert df["custom_field"].values[0] == "test"

            conn.close()

    def test_sorted_output_when_merging(self):
        """Test that output is sorted even after merging with existing CSV"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE VideoData (
                    file_id TEXT,
                    value INTEGER
                )
                """
            )

            # New data
            cursor.execute(
                "INSERT INTO VideoData VALUES ('video3', 30)"
            )
            cursor.execute(
                "INSERT INTO VideoData VALUES ('video1', 10)"
            )
            conn.commit()

            # Create existing CSV with different order
            output_csv = Path(tmpdir) / "output.csv"
            existing_df = pd.DataFrame(
                {"file_id": ["video4", "video2"], "value": [40, 20]}
            )
            existing_df.to_csv(output_csv, index=False)

            generate_excel_ready_csv(conn, str(output_csv))

            df = pd.read_csv(output_csv)

            # Should be sorted
            assert list(df["file_id"]) == ["video1", "video2", "video3", "video4"]
            assert list(df["value"]) == [10, 20, 30, 40]

            conn.close()
