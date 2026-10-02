"""
Unit tests for data_processing/flag_false_positive_videos.py

Tests verify:
- False positive videos are flagged from the sparse detections
- JSON copies
- Only the first false positive video of each event is selected for review
- Examination CSV columns and content
- Kept JSONs, review videos and CSV produced by process_flagging
- Review videos are rendered from the raw videos to {file_id}.mp4
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
from unittest.mock import Mock, patch

import pandas as pd

from data_processing.flag_false_positive_videos import (
    copy_jsons,
    generate_review_csv,
    process_flagging,
    render_review_videos,
)
from data_processing.utils.db_utils import (
    build_database,
    file_id_from_path,
    output_relative_path,
)

MODULE = "data_processing.flag_false_positive_videos"

STATIC_BOX = [0.1, 0.1, 0.2, 0.2]
ANIMAL_BOX = [0.6, 0.5, 0.3, 0.3]

# (raw_file_path, event_id, date_time, bbox): the static box is seen in both events,
# and v2 is recorded before v1
VIDEOS: List[Tuple[str, str, str, List[float]]] = [
    ("cam1/v1.mp4", "e1", "2024-01-01 10:05:00", STATIC_BOX),
    ("cam1/v2.mp4", "e1", "2024-01-01 10:00:00", STATIC_BOX),
    ("cam1/v3.mp4", "e2", "2024-01-02 09:00:00", STATIC_BOX),
    ("cam1/v4.mp4", "e2", "2024-01-02 09:01:00", ANIMAL_BOX),
]


class TestCopyJsons:
    def test_copies_files_into_output_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            json_file = tmpdir / "test.json"
            json_file.write_text('{"test": "data"}')
            output_folder = tmpdir / "output"
            output_path = output_folder / "test.json"

            copy_jsons(
                output_folder,
                pd.DataFrame(
                    {
                        "detection_file_path": [str(json_file)],
                        "output_json_path": [str(output_path)],
                    }
                ),
            )

            assert output_path.read_text() == '{"test": "data"}'

    def test_creates_output_folder_when_nothing_to_copy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_folder = Path(tmpdir) / "output"
            copy_jsons(
                output_folder,
                pd.DataFrame(columns=["detection_file_path", "output_json_path"]),
            )
            assert output_folder.is_dir()


class TestRenderReviewVideos:
    def _render(self, tmpdir: Path, skip_existing: bool) -> Mock:
        df = pd.DataFrame(
            {
                "file_id": ["cam1_v1", "cam1_v2"],
                "raw_file_path": ["raw/cam1/v1.AVI", "raw/cam1/v2.AVI"],
                "detection_file_path": ["json/cam1_v1.json", "json/cam1_v2.json"],
                "output_video_path": [
                    f"{tmpdir}/cam1/cam1_v1.mp4",
                    f"{tmpdir}/cam1/cam1_v2.mp4",
                ],
            }
        )
        with (
            patch(f"{MODULE}.ProcessPoolExecutor", ThreadPoolExecutor),
            patch(f"{MODULE}.process_video") as mock_process_video,
            patch(f"{MODULE}.reencode_video"),
        ):
            render_review_videos(df, n_cores=1, skip_existing=skip_existing)
        return mock_process_video

    def test_renders_raw_videos_with_their_json_to_mp4(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_process_video = self._render(Path(tmpdir), skip_existing=False)
            assert sorted(call.args for call in mock_process_video.call_args_list) == [
                (
                    "raw/cam1/v1.AVI",
                    "json/cam1_v1.json",
                    f"{tmpdir}/cam1/cam1_v1.mp4",
                    "detection",
                ),
                (
                    "raw/cam1/v2.AVI",
                    "json/cam1_v2.json",
                    f"{tmpdir}/cam1/cam1_v2.mp4",
                    "detection",
                ),
            ]

    def test_skip_existing_skips_rendered_videos(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "cam1").mkdir()
            (Path(tmpdir) / "cam1" / "cam1_v1.mp4").touch()
            mock_process_video = self._render(Path(tmpdir), skip_existing=True)
            assert [call.args[0] for call in mock_process_video.call_args_list] == [
                "raw/cam1/v2.AVI"
            ]


class TestGenerateReviewCsv:
    def test_columns_flags_and_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "review.csv"
            df = pd.DataFrame(
                {
                    "file_id": ["cam1_v3", "cam1_v1"],
                    "event_id": ["e2", "e1"],
                    "raw_file_path": ["unused", "unused"],
                }
            )

            generate_review_csv(df, csv_path)

            result = pd.read_csv(csv_path)
            assert list(result.columns) == [
                "file_id",
                "event_id",
                "checked",
                "true_positive",
            ]
            assert result["file_id"].tolist() == ["cam1_v1", "cam1_v3"]
            assert not result["checked"].any()
            assert not result["true_positive"].any()


def setup_project(
    tmpdir: Path, videos: List[Tuple[str, str, str, List[float]]] = VIDEOS
) -> Tuple[Dict, sqlite3.Connection]:
    """Build a database, sparse detection JSONs and raw videos for VIDEOS."""
    build_database(str(tmpdir / "test.db"))
    conn = sqlite3.connect(tmpdir / "test.db")
    conn.executemany(
        "INSERT INTO Videos (file_id, group_id, event_id, raw_file_path, date_time, is_IR)"
        " VALUES (?, ?, ?, ?, ?, 0)",
        [
            (
                file_id_from_path(Path(raw_file_path)),
                "cam1",
                event_id,
                raw_file_path,
                date_time,
            )
            for raw_file_path, event_id, date_time, _ in videos
        ],
    )
    conn.commit()

    for raw_file_path, event_id, _, bbox in videos:
        file_id = file_id_from_path(Path(raw_file_path))
        json_path = tmpdir / "sparse_json" / output_relative_path(raw_file_path, ".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                {
                    "info": {
                        "file_id": file_id,
                        "group_id": "cam1",
                        "event_id": event_id,
                    },
                    "frames": [
                        {
                            "frame_id": 0,
                            "detections": [
                                {
                                    "bbox": bbox,
                                    "conf": 0.9,
                                    "category": "0",
                                }
                            ],
                        }
                    ],
                }
            )
        )
        video_path = tmpdir / "raw_videos" / raw_file_path
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_text(raw_file_path)

    config = {
        "global": {
            "project_path": str(tmpdir),
            "database_path": str(tmpdir / "test.db"),
        },
        "flag_false_positive_videos": {
            "iou_threshold": 0.8,
            "min_events": 2,
            "output_json_folder_name": "filtered_json",
            "output_video_folder_name": "review_videos",
            "output_csv_filename": "review.csv",
            "n_cores": 1,
            "skip_existing": False,
        },
    }
    return config, conn


class TestProcessFlaggingIntegration:
    def _run(
        self, tmpdir: Path, videos: List[Tuple[str, str, str, List[float]]] = VIDEOS
    ) -> Tuple[Mock, Mock]:
        config, conn = setup_project(tmpdir, videos)
        with (
            patch(f"{MODULE}.parse_config", return_value=config),
            patch(f"{MODULE}.load_database", return_value=conn),
            patch(
                f"{MODULE}.get_previous_output",
                side_effect=[str(tmpdir / "sparse_json"), str(tmpdir / "raw_videos")],
            ),
            patch(f"{MODULE}.upsert_stage") as mock_upsert,
            patch(f"{MODULE}.render_review_videos") as mock_render,
        ):
            process_flagging(Mock(config_file="config.yaml"))
        return mock_render, mock_upsert

    def test_sparse_jsons_are_flagged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._run(tmpdir)
            flags = {
                path.stem: json.loads(path.read_text())["info"]["false_positive_video"]
                for path in (tmpdir / "sparse_json").rglob("*.json")
            }
            assert flags == {
                "cam1_v1": True,
                "cam1_v2": True,
                "cam1_v3": True,
                "cam1_v4": False,
            }

    def test_only_non_false_positive_jsons_are_kept(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._run(tmpdir)
            kept = sorted(
                p.relative_to(tmpdir / "filtered_json").as_posix()
                for p in (tmpdir / "filtered_json").rglob("*.json")
            )
            assert kept == ["cam1/cam1_v4.json"]

    def test_earliest_false_positive_video_per_event_is_rendered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            mock_render, _ = self._run(tmpdir)

            mock_render.assert_called_once()
            rendered_df = mock_render.call_args[0][0]
            assert sorted(rendered_df["file_id"]) == ["cam1_v2", "cam1_v3"]
            assert sorted(rendered_df["output_video_path"]) == [
                str(tmpdir / "review_videos" / "cam1" / "cam1_v2.mp4"),
                str(tmpdir / "review_videos" / "cam1" / "cam1_v3.mp4"),
            ]
            assert all(Path(p).exists() for p in rendered_df["raw_file_path"])
            assert all(Path(p).exists() for p in rendered_df["detection_file_path"])

            review_df = pd.read_csv(tmpdir / "review.csv")
            assert review_df["file_id"].tolist() == ["cam1_v2", "cam1_v3"]
            assert review_df["event_id"].tolist() == ["e1", "e2"]

    def test_no_false_positives_records_no_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # A single event cannot reach min_events=2
            single_video = [("cam1/v1.mp4", "e1", "2024-01-01 10:00:00", STATIC_BOX)]
            mock_render, mock_upsert = self._run(tmpdir, single_video)

            mock_render.assert_not_called()
            assert not (tmpdir / "filtered_json").exists()
            assert not (tmpdir / "review.csv").exists()
            stage_outputs = mock_upsert.call_args.kwargs
            assert stage_outputs["output_json_folder"] is None
            assert stage_outputs["output_video_folder"] is None
            assert stage_outputs["examination_csv_path"] is None
