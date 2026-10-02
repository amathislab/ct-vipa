"""
Unit tests for MDVideoOutputParser class.

Tests verify:
- Parsing single JSON file
- Parsing folder recursively with nested structure
- Converting results to DataFrame
- Reading file_id/group_id/event_id from the "info" block
- false_positive_video flag propagation
- Detection expansion and normalization
- Handling empty detections
- Nested folder structure handling
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from data_processing.utils.md_output_parser import MDVideoOutputParser


class TestMDVideoOutputParser:
    """Test MDVideoOutputParser class"""

    def test_parse_single_file(self):
        """Test parsing a single JSON file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_json = {
                "info": {"video_file": "test.mp4", "detector": "MDV5a"},
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": [
                            {
                                "bbox": [0.1, 0.2, 0.3, 0.4],
                                "conf": 0.95,
                                "category": "1",
                            }
                        ],
                    }
                ],
            }

            json_file = Path(tmpdir) / "test.json"
            with open(json_file, "w") as f:
                json.dump(test_json, f)

            parser = MDVideoOutputParser()
            result = parser.parse_file(json_file)

            assert "info" in result
            assert "frames" in result
            assert result["info"]["video_file"] == "test.mp4"
            assert len(result["frames"]) == 1

    def test_parse_folder_flat_structure(self):
        """Test parsing folder with flat structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)

            for i in range(1, 3):
                test_json = {
                    "info": {"video_file": f"IMG_000{i}.MP4"},
                    "frames": [{"frame_id": 0, "detections": []}],
                }
                json_file = folder / f"IMG_000{i}.json"
                with open(json_file, "w") as f:
                    json.dump(test_json, f)

            parser = MDVideoOutputParser()
            results = parser.parse_folder(str(folder))

            # Results are keyed by the JSON path relative to the parsed folder
            assert set(results) == {"IMG_0001.json", "IMG_0002.json"}
            assert results["IMG_0001.json"]["info"]["video_file"] == "IMG_0001.MP4"

    def test_parse_folder_nested_structure(self):
        """Test that nested JSON files are flattened into relative-path keys"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)

            site1_cam1 = folder / "site1" / "cam1"
            site1_cam2 = folder / "site1" / "cam2"
            site2_cam1 = folder / "site2" / "cam1"

            for subfolder in [site1_cam1, site1_cam2, site2_cam1]:
                subfolder.mkdir(parents=True)

            test_json = {
                "info": {"video_file": "IMG_0001.MP4"},
                "frames": [{"frame_id": 0, "detections": []}],
            }

            for subfolder in [site1_cam1, site1_cam2, site2_cam1]:
                with open(subfolder / "IMG_0001.json", "w") as f:
                    json.dump(test_json, f)

            parser = MDVideoOutputParser()
            results = parser.parse_folder(str(folder))

            assert set(results) == {
                "site1/cam1/IMG_0001.json",
                "site1/cam2/IMG_0001.json",
                "site2/cam1/IMG_0001.json",
            }

    def test_parse_folder_stores_detection_file_path(self):
        """Test that detection_file_path is stored as relative path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            subfolder = folder / "site1" / "cam1"
            subfolder.mkdir(parents=True)

            test_json = {
                "info": {"video_file": "IMG_0001.MP4"},
                "frames": [{"frame_id": 0, "detections": []}],
            }

            json_file = subfolder / "IMG_0001.json"
            with open(json_file, "w") as f:
                json.dump(test_json, f)

            parser = MDVideoOutputParser()
            results = parser.parse_folder(str(folder))

            file_path = results["site1/cam1/IMG_0001.json"]["detection_file_path"]
            assert file_path == Path("site1") / "cam1" / "IMG_0001.json"

    def test_parse_folder_raises_error_for_invalid_path(self):
        """Test that parse_folder raises error for non-existent directory"""
        parser = MDVideoOutputParser()

        with pytest.raises(AssertionError, match="is not a directory"):
            parser.parse_folder("/nonexistent/folder")

    def test_to_dataframe_single_file(self):
        """Test conversion to DataFrame with single file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            site_cam = folder / "site1" / "cam1"
            site_cam.mkdir(parents=True)

            test_json = {
                "info": {
                    "video_file": "cam1/IMG_0001.MP4",
                    "file_id": "cam1_IMG_0001",
                },
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": [
                            {
                                "bbox": [0.1, 0.2, 0.3, 0.4],
                                "conf": 0.95,
                                "category": "1",
                            }
                        ],
                    },
                    {
                        "frame_id": 1,
                        "detections": [
                            {
                                "bbox": [0.5, 0.6, 0.2, 0.3],
                                "conf": 0.88,
                                "category": "2",
                            }
                        ],
                    },
                ],
            }

            with open(site_cam / "IMG_0001.json", "w") as f:
                json.dump(test_json, f)

            parser = MDVideoOutputParser()
            parser.parse_folder(str(folder))
            df = parser.to_dataframe()

            assert len(df) == 2  # 2 frames, 1 detection each
            assert "file_id" in df.columns
            assert "frame_id" in df.columns
            assert "detection_file_path" in df.columns
            assert "bbox" in df.columns
            assert "conf" in df.columns
            assert "category" in df.columns

            assert all(df["file_id"] == "cam1_IMG_0001")

            assert list(df["frame_id"]) == [0, 1]

            assert list(df["conf"]) == [0.95, 0.88]

    def test_to_dataframe_multiple_detections_per_frame(self):
        """Test DataFrame with multiple detections per frame"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            site_cam = folder / "site1" / "cam1"
            site_cam.mkdir(parents=True)

            test_json = {
                "info": {"video_file": "cam1/IMG_0001.MP4"},
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": [
                            {
                                "bbox": [0.1, 0.2, 0.3, 0.4],
                                "conf": 0.95,
                                "category": "1",
                            },
                            {
                                "bbox": [0.5, 0.6, 0.2, 0.3],
                                "conf": 0.88,
                                "category": "2",
                            },
                            {
                                "bbox": [0.7, 0.8, 0.1, 0.1],
                                "conf": 0.75,
                                "category": "1",
                            },
                        ],
                    }
                ],
            }

            with open(site_cam / "IMG_0001.json", "w") as f:
                json.dump(test_json, f)

            parser = MDVideoOutputParser()
            parser.parse_folder(str(folder))
            df = parser.to_dataframe()

            # Should have 3 rows (one per detection)
            assert len(df) == 3
            assert all(df["frame_id"] == 0)
            assert list(df["conf"]) == [0.95, 0.88, 0.75]
            assert list(df["category"]) == ["1", "2", "1"]

    def test_to_dataframe_multiple_files(self):
        """Test DataFrame with multiple files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            site_cam = folder / "site1" / "cam1"
            site_cam.mkdir(parents=True)

            for i in range(1, 4):
                test_json = {
                    "info": {
                        "video_file": f"cam1/IMG_000{i}.MP4",
                        "file_id": f"cam1_IMG_000{i}",
                    },
                    "frames": [
                        {
                            "frame_id": 0,
                            "detections": [
                                {
                                    "bbox": [0.1, 0.2, 0.3, 0.4],
                                    "conf": 0.9 + i * 0.01,
                                    "category": "1",
                                }
                            ],
                        }
                    ],
                }

                with open(site_cam / f"IMG_000{i}.json", "w") as f:
                    json.dump(test_json, f)

            parser = MDVideoOutputParser()
            parser.parse_folder(str(folder))
            df = parser.to_dataframe()

            # Should have 3 rows (one per file)
            assert len(df) == 3
            assert set(df["file_id"]) == {
                "cam1_IMG_0001",
                "cam1_IMG_0002",
                "cam1_IMG_0003",
            }

    def test_to_dataframe_with_false_positive_video_flag(self):
        """Test that false_positive_video flag is propagated to DataFrame"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            site_cam = folder / "site1" / "cam1"
            site_cam.mkdir(parents=True)

            test_json = {
                "info": {
                    "video_file": "cam1/IMG_0001.MP4",
                    "false_positive_video": True,
                },
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": [
                            {
                                "bbox": [0.1, 0.2, 0.3, 0.4],
                                "conf": 0.95,
                                "category": "1",
                            }
                        ],
                    }
                ],
            }

            with open(site_cam / "IMG_0001.json", "w") as f:
                json.dump(test_json, f)

            parser = MDVideoOutputParser()
            parser.parse_folder(str(folder))
            df = parser.to_dataframe()

            assert "false_positive_video" in df.columns
            assert bool(df["false_positive_video"].iloc[0]) is True

    def test_to_dataframe_without_false_positive_video_flag(self):
        """Test DataFrame when false_positive_video flag is not present"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            site_cam = folder / "site1" / "cam1"
            site_cam.mkdir(parents=True)

            test_json = {
                "info": {"video_file": "cam1/IMG_0001.MP4"},
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": [
                            {
                                "bbox": [0.1, 0.2, 0.3, 0.4],
                                "conf": 0.95,
                                "category": "1",
                            }
                        ],
                    }
                ],
            }

            with open(site_cam / "IMG_0001.json", "w") as f:
                json.dump(test_json, f)

            parser = MDVideoOutputParser()
            parser.parse_folder(str(folder))
            df = parser.to_dataframe()

            assert "false_positive_video" not in df.columns

    def test_to_dataframe_conf_converted_to_float(self):
        """Test that conf column is converted to float"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            site_cam = folder / "site1" / "cam1"
            site_cam.mkdir(parents=True)

            test_json = {
                "info": {"video_file": "cam1/IMG_0001.MP4"},
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": [
                            {
                                "bbox": [0.1, 0.2, 0.3, 0.4],
                                "conf": "0.95",
                                "category": "1",
                            }
                        ],
                    }
                ],
            }

            with open(site_cam / "IMG_0001.json", "w") as f:
                json.dump(test_json, f)

            parser = MDVideoOutputParser()
            parser.parse_folder(str(folder))
            df = parser.to_dataframe()

            assert df["conf"].dtype == float
            assert df["conf"].iloc[0] == 0.95

    def test_to_dataframe_with_extra_detection_fields(self):
        """Test DataFrame with extra detection fields (track_id, etc.)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            site_cam = folder / "site1" / "cam1"
            site_cam.mkdir(parents=True)

            test_json = {
                "info": {"video_file": "cam1/IMG_0001.MP4"},
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": [
                            {
                                "bbox": [0.1, 0.2, 0.3, 0.4],
                                "conf": 0.95,
                                "category": "1",
                                "track_id": 5,
                                "false_positive_detection": False,
                            }
                        ],
                    }
                ],
            }

            with open(site_cam / "IMG_0001.json", "w") as f:
                json.dump(test_json, f)

            parser = MDVideoOutputParser()
            parser.parse_folder(str(folder))
            df = parser.to_dataframe()

            assert "track_id" in df.columns
            assert "false_positive_detection" in df.columns
            assert df["track_id"].iloc[0] == 5
            assert bool(df["false_positive_detection"].iloc[0]) is False

    def test_to_dataframe_handles_empty_detections(self):
        """Test DataFrame with frames that have empty detections"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            site_cam = folder / "site1" / "cam1"
            site_cam.mkdir(parents=True)

            test_json = {
                "info": {"video_file": "cam1/IMG_0001.MP4"},
                "frames": [
                    {"frame_id": 0, "detections": []},
                    {
                        "frame_id": 1,
                        "detections": [
                            {
                                "bbox": [0.1, 0.2, 0.3, 0.4],
                                "conf": 0.95,
                                "category": "1",
                            }
                        ],
                    },
                    {"frame_id": 2, "detections": []},
                ],
            }

            with open(site_cam / "IMG_0001.json", "w") as f:
                json.dump(test_json, f)

            parser = MDVideoOutputParser()
            parser.parse_folder(str(folder))
            df = parser.to_dataframe()

            # Should handle empty detections (frames with no detections may not appear or have NaN)
            # The explode operation will drop frames with empty lists
            assert len(df) >= 1  # At least the frame with detection
            assert 1 in df["frame_id"].values

    def test_to_dataframe_reads_ids_from_info(self):
        """Test that file_id/group_id/event_id come from the "info" block, not the filename"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)

            test_json = {
                "info": {
                    "video_file": "cam1/IMG_0001.MP4",
                    "file_id": "cam1_IMG_0001",
                    "group_id": "cam1",
                    "event_id": "E1",
                },
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": [
                            {
                                "bbox": [0.1, 0.2, 0.3, 0.4],
                                "conf": 0.95,
                                "category": "1",
                            }
                        ],
                    }
                ],
            }

            with open(folder / "unrelated_name.json", "w") as f:
                json.dump(test_json, f)

            parser = MDVideoOutputParser()
            parser.parse_folder(str(folder))
            df = parser.to_dataframe()

            assert df["file_id"].iloc[0] == "cam1_IMG_0001"
            assert df["group_id"].iloc[0] == "cam1"
            assert df["event_id"].iloc[0] == "E1"

    def test_to_dataframe_file_id_falls_back_to_json_stem(self):
        """Test that JSONs without file_id in "info" use the JSON filename stem"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            site_cam = folder / "site1" / "cam1"
            site_cam.mkdir(parents=True)

            test_json = {
                "info": {"video_file": "IMG_0001.MP4"},
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": [
                            {
                                "bbox": [0.1, 0.2, 0.3, 0.4],
                                "conf": 0.95,
                                "category": "1",
                            }
                        ],
                    }
                ],
            }

            with open(site_cam / "IMG_0001.json", "w") as f:
                json.dump(test_json, f)

            parser = MDVideoOutputParser()
            parser.parse_folder(str(folder))
            df = parser.to_dataframe()

            assert df["file_id"].iloc[0] == "IMG_0001"
            assert df["group_id"].iloc[0] is None
            assert df["event_id"].iloc[0] is None

    def test_to_dataframe_nested_folders_multiple_sites(self):
        """Test DataFrame with complex nested structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)

            group_ids = ["site1/cam1", "site1/cam2", "site2/cam1"]

            for group_id in group_ids:
                group_folder = folder / group_id
                group_folder.mkdir(parents=True)

                test_json = {
                    "info": {
                        "video_file": f"{group_id}/IMG_0001.MP4",
                        "file_id": f"{group_id}_IMG_0001",
                        "group_id": group_id,
                    },
                    "frames": [
                        {
                            "frame_id": 0,
                            "detections": [
                                {
                                    "bbox": [0.1, 0.2, 0.3, 0.4],
                                    "conf": 0.95,
                                    "category": "1",
                                }
                            ],
                        }
                    ],
                }

                with open(group_folder / "IMG_0001.json", "w") as f:
                    json.dump(test_json, f)

            parser = MDVideoOutputParser()
            parser.parse_folder(str(folder))
            df = parser.to_dataframe()

            # Same video filename in different groups stays distinct
            assert len(df) == 3
            assert set(df["file_id"]) == {
                f"{group_id}_IMG_0001" for group_id in group_ids
            }
            assert set(df["group_id"]) == set(group_ids)

    def test_to_dataframe_bbox_as_list(self):
        """Test that bbox field remains as list in DataFrame"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            site_cam = folder / "site1" / "cam1"
            site_cam.mkdir(parents=True)

            test_json = {
                "info": {"video_file": "cam1/IMG_0001.MP4"},
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": [
                            {
                                "bbox": [0.1, 0.2, 0.3, 0.4],
                                "conf": 0.95,
                                "category": "1",
                            }
                        ],
                    }
                ],
            }

            with open(site_cam / "IMG_0001.json", "w") as f:
                json.dump(test_json, f)

            parser = MDVideoOutputParser()
            parser.parse_folder(str(folder))
            df = parser.to_dataframe()

            assert isinstance(df["bbox"].iloc[0], list)
            assert df["bbox"].iloc[0] == [0.1, 0.2, 0.3, 0.4]

    def test_parser_reusability(self):
        """Test that parser can be reused for multiple parse operations"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            site_cam = folder / "site1" / "cam1"
            site_cam.mkdir(parents=True)

            test_json = {
                "info": {"video_file": "cam1/IMG_0001.MP4"},
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": [
                            {
                                "bbox": [0.1, 0.2, 0.3, 0.4],
                                "conf": 0.95,
                                "category": "1",
                            }
                        ],
                    }
                ],
            }

            with open(site_cam / "IMG_0001.json", "w") as f:
                json.dump(test_json, f)

            # First parse
            parser = MDVideoOutputParser()
            parser.parse_folder(str(folder))
            df1 = parser.to_dataframe()

            # Add another file
            test_json2 = {
                "info": {"video_file": "cam1/IMG_0002.MP4"},
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": [
                            {
                                "bbox": [0.5, 0.6, 0.2, 0.3],
                                "conf": 0.88,
                                "category": "2",
                            }
                        ],
                    }
                ],
            }

            with open(site_cam / "IMG_0002.json", "w") as f:
                json.dump(test_json2, f)

            # Re-parsing replaces previous results with the folder's current content
            parser.parse_folder(str(folder))
            df2 = parser.to_dataframe()

            assert len(df1) == 1
            assert len(df2) == 2
