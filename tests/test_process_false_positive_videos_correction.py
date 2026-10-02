"""
Unit tests for data_processing/process_false_positive_videos_correction.py

Tests verify:
- All videos of an event reviewed as true positive are selected
- Unchecked or false positive reviews are ignored
- Reintegrated JSONs have their false positive flags cleared
- process_reinsertion copies the true positive events into the filtered folder
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import List, Tuple
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from data_processing.process_false_positive_videos_correction import (
    load_true_positive_videos,
    process_reinsertion,
    reintegrate_jsons,
)
from data_processing.utils.db_utils import (
    build_database,
    file_id_from_path,
    output_relative_path,
)

MODULE = "data_processing.process_false_positive_videos_correction"

# (raw_file_path, event_id)
VIDEOS: List[Tuple[str, str]] = [
    ("cam1/v1.mp4", "10"),
    ("cam1/v2.mp4", "10"),
    ("cam1/v3.mp4", "11"),
    ("cam2/v1.mp4", "12"),
]


def write_sparse_json(sparse_folder: Path, raw_file_path: str) -> Path:
    file_id = file_id_from_path(Path(raw_file_path))
    json_path = sparse_folder / output_relative_path(raw_file_path, ".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "info": {"file_id": file_id, "false_positive_video": True},
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": [
                            {
                                "bbox": [0.1, 0.1, 0.2, 0.2],
                                "conf": 0.9,
                                "false_positive_detection": True,
                            }
                        ],
                    }
                ],
            }
        )
    )
    return json_path


def setup_project(
    tmpdir: Path, review_rows: List[Tuple[str, str, bool, bool]]
) -> sqlite3.Connection:
    """Build the Videos table, the sparse JSONs of VIDEOS and a filled review CSV."""
    build_database(str(tmpdir / "test.db"))
    conn = sqlite3.connect(tmpdir / "test.db")
    conn.executemany(
        "INSERT INTO Videos (file_id, group_id, event_id, raw_file_path) VALUES (?, ?, ?, ?)",
        [
            (
                file_id_from_path(Path(raw_file_path)),
                str(Path(raw_file_path).parent),
                event_id,
                raw_file_path,
            )
            for raw_file_path, event_id in VIDEOS
        ],
    )
    conn.commit()
    for raw_file_path, _ in VIDEOS:
        write_sparse_json(tmpdir / "sparse_json", raw_file_path)
    pd.DataFrame(
        review_rows, columns=["file_id", "event_id", "checked", "true_positive"]
    ).to_csv(tmpdir / "review.csv", index=False)
    return conn


class TestLoadTruePositiveVideos:
    def test_expands_to_all_videos_of_the_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            conn = setup_project(tmpdir, [("cam1_v1", "10", True, True)])

            result = load_true_positive_videos(
                tmpdir / "review.csv", conn, tmpdir / "sparse_json"
            )

            assert sorted(result["file_id"]) == ["cam1_v1", "cam1_v2"]
            assert all(Path(p).exists() for p in result["detection_file_path"])
            conn.close()

    def test_ignores_unchecked_and_false_positive_reviews(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            conn = setup_project(
                tmpdir,
                [
                    ("cam1_v1", "10", False, True),
                    ("cam1_v3", "11", True, False),
                    ("cam2_v1", "12", True, True),
                ],
            )

            result = load_true_positive_videos(
                tmpdir / "review.csv", conn, tmpdir / "sparse_json"
            )

            assert result["file_id"].tolist() == ["cam2_v1"]
            conn.close()

    def test_skips_videos_without_sparse_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            conn = setup_project(tmpdir, [("cam1_v1", "10", True, True)])
            (tmpdir / "sparse_json" / "cam1" / "cam1_v2.json").unlink()

            result = load_true_positive_videos(
                tmpdir / "review.csv", conn, tmpdir / "sparse_json"
            )

            assert result["file_id"].tolist() == ["cam1_v1"]
            conn.close()

    def test_missing_required_column_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "review.csv"
            pd.DataFrame(
                {"file_id": ["cam1_v1"], "event_id": ["10"], "checked": [True]}
            ).to_csv(csv_path, index=False)
            with pytest.raises(ValueError, match="true_positive"):
                load_true_positive_videos(csv_path, Mock(), Path(tmpdir))


class TestReintegrateJsons:
    def test_clears_flags_and_keeps_relative_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            json_src = write_sparse_json(tmpdir / "sparse_json", "cam1/v1.mp4")

            reintegrate_jsons(
                pd.DataFrame({"detection_file_path": [str(json_src)]}),
                json_folder=tmpdir / "sparse_json",
                filtered_json_output=tmpdir / "filtered_json",
            )

            data = json.loads(
                (tmpdir / "filtered_json" / "cam1" / "cam1_v1.json").read_text()
            )
            assert data["info"]["false_positive_video"] is False
            assert data["info"]["file_id"] == "cam1_v1"
            detection = data["frames"][0]["detections"][0]
            assert detection["false_positive_detection"] is False
            assert detection["conf"] == 0.9


class TestProcessReinsertionIntegration:
    def _run(
        self, tmpdir: Path, conn: sqlite3.Connection, review_csv_path: Path
    ) -> None:
        config = {
            "global": {
                "project_path": str(tmpdir),
                "database_path": str(tmpdir / "test.db"),
            }
        }
        with (
            patch(f"{MODULE}.parse_config", return_value=config),
            patch(f"{MODULE}.load_database", return_value=conn),
            patch(
                f"{MODULE}.get_previous_output",
                side_effect=[
                    str(tmpdir / "sparse_json"),
                    str(tmpdir / "filtered_json"),
                    str(review_csv_path),
                ],
            ),
            patch(f"{MODULE}.upsert_stage") as mock_upsert,
        ):
            process_reinsertion(Mock(config_file="config.yaml"))
        mock_upsert.assert_called_once()

    def test_true_positive_events_are_reintegrated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            conn = setup_project(
                tmpdir,
                [("cam1_v1", "10", True, True), ("cam1_v3", "11", True, False)],
            )

            self._run(tmpdir, conn, tmpdir / "review.csv")

            reintegrated = sorted(
                p.relative_to(tmpdir / "filtered_json").as_posix()
                for p in (tmpdir / "filtered_json").rglob("*.json")
            )
            assert reintegrated == ["cam1/cam1_v1.json", "cam1/cam1_v2.json"]

    def test_missing_csv_reintegrates_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            conn = setup_project(tmpdir, [])

            self._run(tmpdir, conn, tmpdir / "missing.csv")

            assert not (tmpdir / "filtered_json").exists()
