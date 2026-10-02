"""
Unit tests for megadetector_utils module.

Tests verify:
- Correct conversion between MegaDetector and supervision formats
- Correct XML/JSON conversion for video annotations and tracklets
- Proper video dimensions extraction
- Frame saving and deletion operations
- Results writing to file with correct structure
- Detection loading from JSON folders
"""

import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pandas as pd
import pytest

import supervision as sv
from data_processing.utils.megadetector_utils import (
    create_default_xml_tracklet,
    delete_frames_from_disk,
    from_json_to_video_xml,
    from_megadetector_to_supervision,
    from_supervision_to_megadetector,
    from_track_xml_to_json,
    from_video_xml_to_json,
    get_video_dimensions,
    load_all_detections,
    outputs_df,
    save_frames_to_disk,
    write_results_to_file,
)


class TestFromMegadetectorToSupervision:
    """Test conversion from MegaDetector format to supervision Detections"""

    def test_single_detection_conversion(self):
        """Test conversion of single detection with core fields"""
        frame_results = {
            "detections": [
                {"bbox": [0.1, 0.2, 0.3, 0.4], "conf": 0.95, "category": "1"}
            ]
        }
        width, height = 1920, 1080

        detections = from_megadetector_to_supervision(frame_results, width, height)

        assert len(detections) == 1

        # Check coordinates conversion (xywh relative -> xyxy absolute)
        expected_xyxy = np.array(
            [[0.1 * 1920, 0.2 * 1080, (0.1 + 0.3) * 1920, (0.2 + 0.4) * 1080]]
        )
        np.testing.assert_array_almost_equal(
            detections.xyxy, expected_xyxy.round().astype(int)
        )

        assert detections.confidence[0] == 0.95
        assert detections.class_id[0] == 1

    def test_multiple_detections_conversion(self):
        """Test conversion of multiple detections"""
        frame_results = {
            "detections": [
                {"bbox": [0.1, 0.2, 0.3, 0.4], "conf": 0.95, "category": "1"},
                {"bbox": [0.5, 0.6, 0.2, 0.3], "conf": 0.88, "category": "2"},
                {"bbox": [0.0, 0.0, 0.1, 0.1], "conf": 0.75, "category": "1"},
            ]
        }
        width, height = 1920, 1080

        detections = from_megadetector_to_supervision(frame_results, width, height)

        assert len(detections) == 3
        assert list(detections.confidence) == [0.95, 0.88, 0.75]
        assert list(detections.class_id) == [1, 2, 1]

    def test_conversion_with_track_id(self):
        """Test conversion preserves track_id field"""
        frame_results = {
            "detections": [
                {
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "conf": 0.95,
                    "category": "1",
                    "track_id": 5,
                },
                {
                    "bbox": [0.5, 0.6, 0.2, 0.3],
                    "conf": 0.88,
                    "category": "1",
                    "track_id": 7,
                },
            ]
        }
        width, height = 1920, 1080

        detections = from_megadetector_to_supervision(frame_results, width, height)

        assert detections.tracker_id is not None
        assert list(detections.tracker_id) == [5, 7]

    def test_conversion_with_extra_fields(self):
        """Test conversion preserves extra fields in data attribute"""
        frame_results = {
            "detections": [
                {
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "conf": 0.95,
                    "category": "1",
                    "track_id": 5,
                    "false_positive_detection": True,
                    "custom_field": "test_value",
                }
            ]
        }
        width, height = 1920, 1080

        detections = from_megadetector_to_supervision(frame_results, width, height)

        assert "false_positive_detection" in detections.data
        assert bool(detections.data["false_positive_detection"][0]) is True
        assert "custom_field" in detections.data
        assert detections.data["custom_field"][0] == "test_value"

    def test_empty_detections(self):
        """Test conversion of empty detections returns empty Detections object"""
        frame_results = {"detections": []}
        width, height = 1920, 1080

        detections = from_megadetector_to_supervision(frame_results, width, height)

        assert len(detections) == 0
        assert isinstance(detections, sv.Detections)

    def test_coordinate_rounding(self):
        """Test that coordinates are properly rounded to integers"""
        frame_results = {
            "detections": [
                {
                    "bbox": [0.12345, 0.67891, 0.11111, 0.22222],
                    "conf": 0.9,
                    "category": "1",
                }
            ]
        }
        width, height = 1000, 1000

        detections = from_megadetector_to_supervision(frame_results, width, height)

        # All coordinates should be integers
        assert detections.xyxy.dtype == np.int64 or detections.xyxy.dtype == np.int32


class TestFromSupervisionToMegadetector:
    """Test conversion from supervision Detections to MegaDetector format"""

    def test_single_detection_conversion(self):
        """Test conversion of single detection"""
        xyxy = np.array([[192, 216, 768, 648]])
        confidence = np.array([0.95])
        class_id = np.array([1])
        detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        width, height = 1920, 1080

        result = from_supervision_to_megadetector(detections, width, height)

        assert len(result) == 1
        det = result[0]
        assert "bbox" in det
        assert "conf" in det
        assert "category" in det
        assert det["conf"] == 0.95
        assert det["category"] == "1"

        # Check bbox conversion (xyxy absolute -> xywh relative)
        expected_bbox = [
            192 / 1920,
            216 / 1080,
            (768 - 192) / 1920,
            (648 - 216) / 1080,
        ]
        np.testing.assert_array_almost_equal(det["bbox"], expected_bbox, decimal=6)

    def test_multiple_detections_conversion(self):
        """Test conversion of multiple detections"""
        xyxy = np.array([[100, 200, 300, 400], [500, 600, 700, 800]])
        confidence = np.array([0.95, 0.88])
        class_id = np.array([1, 2])
        detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        width, height = 1920, 1080

        result = from_supervision_to_megadetector(detections, width, height)

        assert len(result) == 2
        assert result[0]["conf"] == 0.95
        assert result[1]["conf"] == 0.88
        assert result[0]["category"] == "1"
        assert result[1]["category"] == "2"

    def test_conversion_with_track_id(self):
        """Test conversion preserves track_id"""
        xyxy = np.array([[100, 200, 300, 400]])
        confidence = np.array([0.95])
        class_id = np.array([1])
        tracker_id = np.array([5])
        detections = sv.Detections(
            xyxy=xyxy, confidence=confidence, class_id=class_id, tracker_id=tracker_id
        )
        width, height = 1920, 1080

        result = from_supervision_to_megadetector(detections, width, height)

        assert result[0]["track_id"] == 5

    def test_conversion_with_extra_data_fields(self):
        """Test conversion preserves extra data fields"""
        xyxy = np.array([[100, 200, 300, 400]])
        confidence = np.array([0.95])
        class_id = np.array([1])
        tracker_id = np.array([5])
        data = {
            "false_positive_detection": np.array([True]),
            "custom_score": np.array([0.75]),
        }
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
            data=data,
        )
        width, height = 1920, 1080

        result = from_supervision_to_megadetector(detections, width, height)

        assert result[0]["false_positive_detection"] is True
        assert result[0]["custom_score"] == 0.75

    def test_conversion_without_tracker_id(self):
        """Test conversion when tracker_id is None"""
        xyxy = np.array([[100, 200, 300, 400]])
        confidence = np.array([0.95])
        class_id = np.array([1])
        detections = sv.Detections(
            xyxy=xyxy, confidence=confidence, class_id=class_id, tracker_id=None
        )
        width, height = 1920, 1080

        result = from_supervision_to_megadetector(detections, width, height)

        assert result[0]["track_id"] is None

    def test_roundtrip_conversion(self):
        """Test that converting MD -> SV -> MD preserves data"""
        original_md = {
            "detections": [
                {
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "conf": 0.95,
                    "category": "1",
                    "track_id": 5,
                    "false_positive_detection": True,
                }
            ]
        }
        width, height = 1920, 1080

        # MD -> SV
        sv_detections = from_megadetector_to_supervision(original_md, width, height)

        # SV -> MD
        back_to_md = from_supervision_to_megadetector(sv_detections, width, height)

        # Check core fields are preserved (with some rounding tolerance)
        assert len(back_to_md) == 1
        assert back_to_md[0]["conf"] == 0.95
        assert back_to_md[0]["category"] == "1"
        assert back_to_md[0]["track_id"] == 5
        assert back_to_md[0]["false_positive_detection"] is True
        np.testing.assert_array_almost_equal(
            back_to_md[0]["bbox"], original_md["detections"][0]["bbox"], decimal=3
        )


class TestFromJsonToVideoXml:
    """Test JSON to video XML conversion"""

    def test_single_track_conversion(self):
        """Test conversion of single track to XML"""
        input_json = {
            "info": {"width": 1920, "height": 1080},
            "frames": [
                {
                    "frame_id": 0,
                    "detections": [
                        {
                            "track_id": 1,
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
                            "track_id": 1,
                            "bbox": [0.15, 0.25, 0.3, 0.4],
                            "conf": 0.92,
                            "category": "1",
                        }
                    ],
                },
            ],
        }

        xml_str = from_json_to_video_xml(input_json)

        root = ET.fromstring(xml_str)
        tracks = root.findall("track")

        assert len(tracks) == 1
        assert tracks[0].attrib["id"] == "1"
        assert tracks[0].attrib["label"] == "Animal"

        boxes = tracks[0].findall("box")
        # Should have 2 boxes for frames + 1 outside box
        assert len(boxes) >= 2

    def test_multiple_tracks_conversion(self):
        """Test conversion of multiple tracks to XML"""
        input_json = {
            "info": {"width": 1920, "height": 1080},
            "frames": [
                {
                    "frame_id": 0,
                    "detections": [
                        {
                            "track_id": 1,
                            "bbox": [0.1, 0.2, 0.3, 0.4],
                            "conf": 0.95,
                            "category": "1",
                        },
                        {
                            "track_id": 2,
                            "bbox": [0.5, 0.6, 0.2, 0.3],
                            "conf": 0.88,
                            "category": "1",
                        },
                    ],
                }
            ],
        }

        xml_str = from_json_to_video_xml(input_json)

        root = ET.fromstring(xml_str)
        tracks = root.findall("track")

        assert len(tracks) == 2
        track_ids = [track.attrib["id"] for track in tracks]
        assert "1" in track_ids
        assert "2" in track_ids

    def test_bbox_coordinate_conversion(self):
        """Test that bbox coordinates are correctly converted to absolute"""
        input_json = {
            "info": {"width": 1000, "height": 1000},
            "frames": [
                {
                    "frame_id": 0,
                    "detections": [
                        {
                            "track_id": 1,
                            "bbox": [0.1, 0.2, 0.3, 0.4],
                            "conf": 0.95,
                            "category": "1",
                        }
                    ],
                }
            ],
        }

        xml_str = from_json_to_video_xml(input_json)

        root = ET.fromstring(xml_str)
        box = root.find("track").find("box")

        assert float(box.attrib["xtl"]) == 100.0  # 0.1 * 1000
        assert float(box.attrib["ytl"]) == 200.0  # 0.2 * 1000
        assert float(box.attrib["xbr"]) == 400.0  # (0.1 + 0.3) * 1000
        assert float(box.attrib["ybr"]) == 600.0  # (0.2 + 0.4) * 1000


class TestFromVideoXmlToJson:
    """Test video XML to JSON conversion"""

    def test_single_track_parsing(self):
        """Test parsing XML with single track"""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <meta>
    <task>
      <original_size>
        <width>1920</width>
        <height>1080</height>
      </original_size>
    </task>
  </meta>
  <track id="1" label="Animal" source="manual">
    <box frame="0" keyframe="1" outside="0" occluded="0" xtl="192.0" ytl="216.0" xbr="768.0" ybr="648.0" z_order="0">
      <attribute name="Species">deer</attribute>
    </box>
  </track>
</annotations>"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            temp_path = f.name

        try:
            result = from_video_xml_to_json(temp_path)

            assert 0 in result  # frame 0
            assert 1 in result[0]  # track 1
            assert "bbox" in result[0][1]
            assert "attributes" in result[0][1]
            assert "label" in result[0][1]
            assert result[0][1]["label"] == "Animal"
            assert result[0][1]["outside"] is False
            assert result[0][1]["occluded"] is False
            assert result[0][1]["attributes"]["Species"] == "deer"
        finally:
            Path(temp_path).unlink()

    def test_multiple_frames_parsing(self):
        """Test parsing XML with multiple frames"""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <meta>
    <task>
      <original_size>
        <width>1920</width>
        <height>1080</height>
      </original_size>
    </task>
  </meta>
  <track id="1" label="Animal" source="manual">
    <box frame="0" keyframe="1" outside="0" occluded="0" xtl="192.0" ytl="216.0" xbr="768.0" ybr="648.0" z_order="0"/>
    <box frame="1" keyframe="1" outside="0" occluded="0" xtl="200.0" ytl="220.0" xbr="770.0" ybr="650.0" z_order="0"/>
  </track>
</annotations>"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            temp_path = f.name

        try:
            result = from_video_xml_to_json(temp_path)

            assert 0 in result
            assert 1 in result
            assert 1 in result[0]  # track 1 in frame 0
            assert 1 in result[1]  # track 1 in frame 1
        finally:
            Path(temp_path).unlink()

    def test_bbox_coordinate_conversion_to_relative(self):
        """Test that absolute bbox coordinates are converted to relative"""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <meta>
    <task>
      <original_size>
        <width>1000</width>
        <height>1000</height>
      </original_size>
    </task>
  </meta>
  <track id="1" label="Animal" source="manual">
    <box frame="0" keyframe="1" outside="0" occluded="0" xtl="100.0" ytl="200.0" xbr="400.0" ybr="600.0" z_order="0"/>
  </track>
</annotations>"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            temp_path = f.name

        try:
            result = from_video_xml_to_json(temp_path)

            bbox = result[0][1]["bbox"]
            expected = [0.1, 0.2, 0.3, 0.4]  # Relative coordinates
            np.testing.assert_array_almost_equal(bbox, expected, decimal=6)
        finally:
            Path(temp_path).unlink()


class TestFromTrackXmlToJson:
    """Test track XML to JSON conversion for tracklet annotations"""

    def test_tracklet_annotation_parsing(self):
        """Test parsing tracklet annotations from XML"""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <track id="0" label="Animal" source="manual">
    <box frame="0" keyframe="1" outside="0" occluded="0" xtl="0" ytl="0" xbr="100" ybr="100" z_order="0">
      <attribute name="Species">deer</attribute>
      <attribute name="Action">walking</attribute>
    </box>
    <box frame="1" keyframe="1" outside="0" occluded="1" xtl="0" ytl="0" xbr="100" ybr="100" z_order="0">
      <attribute name="Species">deer</attribute>
      <attribute name="Action">running</attribute>
    </box>
  </track>
</annotations>"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            temp_path = f.name

        try:
            result = from_track_xml_to_json(temp_path)

            assert len(result) == 2
            assert result[0]["frame_id"] == 0
            assert result[0]["occluded"] is False
            assert result[0]["attributes"]["Species"] == "deer"
            assert result[0]["attributes"]["Action"] == "walking"

            assert result[1]["frame_id"] == 1
            assert result[1]["occluded"] is True
            assert result[1]["attributes"]["Action"] == "running"
        finally:
            Path(temp_path).unlink()

    def test_tracklet_includes_all_attributes(self):
        """Test that all attributes are included (description is NOT skipped in tracklet annotations)"""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <track id="0" label="Animal" source="manual">
    <box frame="0" keyframe="1" outside="0" occluded="0" xtl="0" ytl="0" xbr="100" ybr="100" z_order="0">
      <attribute name="Species">deer</attribute>
      <attribute name="description">This is a note</attribute>
    </box>
  </track>
</annotations>"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            temp_path = f.name

        try:
            result = from_track_xml_to_json(temp_path)

            # In tracklet annotations, description IS included
            assert "description" in result[0]["attributes"]
            assert "Species" in result[0]["attributes"]
        finally:
            Path(temp_path).unlink()


class TestCreateDefaultXmlTracklet:
    """Test creation of default XML tracklet"""

    def test_creates_valid_xml_structure(self):
        """Test that default XML tracklet has correct structure"""
        tublet_size = (1920, 1080)

        xml_str = create_default_xml_tracklet(tublet_size)

        root = ET.fromstring(xml_str)
        assert root.tag == "annotations"

        track = root.find("track")
        assert track is not None
        assert track.attrib["id"] == "0"
        assert track.attrib["label"] == "Animal"
        assert track.attrib["source"] == "manual"

        box = track.find("box")
        assert box is not None
        assert box.attrib["frame"] == "0"
        assert box.attrib["keyframe"] == "1"
        assert box.attrib["outside"] == "0"
        assert box.attrib["xtl"] == "0"
        assert box.attrib["ytl"] == "0"
        assert box.attrib["xbr"] == "1920"
        assert box.attrib["ybr"] == "1080"

    def test_uses_provided_dimensions(self):
        """Test that provided dimensions are used"""
        tublet_size = (640, 480)

        xml_str = create_default_xml_tracklet(tublet_size)

        root = ET.fromstring(xml_str)
        box = root.find("track").find("box")

        assert box.attrib["xbr"] == "640"
        assert box.attrib["ybr"] == "480"


class TestGetVideoDimensions:
    """Test video dimensions extraction"""

    @patch("cv2.VideoCapture")
    def test_get_dimensions_success(self, mock_capture):
        """Test successful dimension extraction"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }[prop]
        mock_capture.return_value = mock_cap

        width, height = get_video_dimensions("test.mp4")

        assert width == 1920
        assert height == 1080
        mock_cap.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_get_dimensions_video_not_opened(self, mock_capture):
        """Test error when video cannot be opened"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap

        with pytest.raises(ValueError, match="Cannot open video file"):
            get_video_dimensions("nonexistent.mp4")


class TestSaveFramesToDisk:
    """Test frame extraction and saving"""

    @patch("data_processing.utils.megadetector_utils.sv.get_video_frames_generator")
    @patch("cv2.imwrite")
    def test_saves_frames_with_stride(self, mock_imwrite, mock_generator):
        """Test that frames are saved with correct naming"""
        mock_frames = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(3)]
        mock_generator.return_value = iter(mock_frames)
        mock_imwrite.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            save_frames_to_disk("test.mp4", tmpdir, stride=10)

            assert mock_imwrite.call_count == 3

            calls = mock_imwrite.call_args_list
            expected_names = [f"frame_{i:05d}.jpg" for i in range(3)]
            for call, expected_name in zip(calls, expected_names):
                assert expected_name in call[0][0]

    @patch("data_processing.utils.megadetector_utils.sv.get_video_frames_generator")
    @patch("cv2.imwrite")
    def test_creates_output_folder(self, mock_imwrite, mock_generator):
        """Test that output folder is created if it doesn't exist"""
        mock_generator.return_value = iter([np.zeros((100, 100, 3))])
        mock_imwrite.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            output_folder = Path(tmpdir) / "frames" / "subfolder"
            save_frames_to_disk("test.mp4", str(output_folder), stride=1)

            assert output_folder.exists()


class TestDeleteFramesFromDisk:
    """Test frame deletion"""

    def test_deletes_jpg_files(self):
        """Test that only .jpg files are deleted"""
        with tempfile.TemporaryDirectory() as tmpdir:
            jpg1 = Path(tmpdir) / "frame_001.jpg"
            jpg2 = Path(tmpdir) / "frame_002.jpg"
            txt_file = Path(tmpdir) / "info.txt"

            jpg1.write_text("test")
            jpg2.write_text("test")
            txt_file.write_text("test")

            delete_frames_from_disk(tmpdir)

            # JPG files should be deleted
            assert not jpg1.exists()
            assert not jpg2.exists()
            # TXT file should remain
            assert txt_file.exists()

    def test_handles_nonexistent_folder(self):
        """Test that function handles non-existent folder gracefully"""
        # Should not raise error
        delete_frames_from_disk("/nonexistent/folder")

    def test_handles_empty_folder(self):
        """Test that function handles empty folder gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise error
            delete_frames_from_disk(tmpdir)


class TestWriteResultsToFile:
    """Test writing detection results to JSON file"""

    def test_writes_correct_structure(self):
        """Test that output JSON has correct structure"""
        results = [
            {
                "frame_id": 0,
                "detections": [
                    {"bbox": [0.1, 0.2, 0.3, 0.4], "conf": 0.95, "category": "1"}
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.json"

            result = write_results_to_file(
                results=results,
                input_video_file="test.mp4",
                output_file=str(output_file),
                file_id="cam1_IMG_0001",
                group_id="cam1",
                event_id="E1",
                width=1920,
                height=1080,
                detector="MDV5a",
                detection_categories={"0": "animal", "1": "person", "2": "vehicle"},
            )

            assert output_file.exists()

            with open(output_file) as f:
                data = json.load(f)

            assert "frames" in data
            assert "detection_categories" in data
            assert "info" in data
            assert data["info"]["video_file"] == "test.mp4"
            assert data["info"]["file_id"] == "cam1_IMG_0001"
            assert data["info"]["group_id"] == "cam1"
            assert data["info"]["event_id"] == "E1"
            assert data["info"]["width"] == 1920
            assert data["info"]["height"] == 1080
            assert data["info"]["detector"] == "MDV5a"

    def test_includes_detection_completion_time(self):
        """Test that detection_completion_time is included"""
        results = []

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.json"

            write_results_to_file(
                results=results,
                input_video_file="test.mp4",
                output_file=str(output_file),
                file_id="cam1_IMG_0001",
                group_id="cam1",
                event_id="E1",
                width=1920,
                height=1080,
                detector="MDV5a",
                detection_categories={},
            )

            with open(output_file) as f:
                data = json.load(f)

            assert "detection_completion_time" in data["info"]
            assert data["info"]["format_version"] == "1.3"

    def test_preserves_detection_data(self):
        """Test that detection data is preserved correctly"""
        results = [
            {
                "frame_id": 0,
                "detections": [
                    {"bbox": [0.1, 0.2, 0.3, 0.4], "conf": 0.95, "category": "1"},
                    {"bbox": [0.5, 0.6, 0.2, 0.3], "conf": 0.88, "category": "2"},
                ],
            },
            {
                "frame_id": 1,
                "detections": [
                    {"bbox": [0.15, 0.25, 0.3, 0.4], "conf": 0.92, "category": "1"}
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.json"

            write_results_to_file(
                results=results,
                input_video_file="test.mp4",
                output_file=str(output_file),
                file_id="cam1_IMG_0001",
                group_id="cam1",
                event_id="E1",
                width=1920,
                height=1080,
                detector="MDV5a",
                detection_categories={},
            )

            with open(output_file) as f:
                data = json.load(f)

            assert len(data["frames"]) == 2
            assert len(data["frames"][0]["detections"]) == 2
            assert len(data["frames"][1]["detections"]) == 1
            assert data["frames"][0]["detections"][0]["conf"] == 0.95


class TestLoadAllDetections:
    """Test loading all detections from JSON folder"""

    def test_loads_single_json_file(self):
        """Test loading detections from single JSON file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_folder = Path(tmpdir)

            test_json = {
                "info": {
                    "video_file": "cam1/IMG_0001.MP4",
                    "file_id": "cam1_IMG_0001",
                    "group_id": "cam1",
                    "width": 1920,
                    "height": 1080,
                    "false_positive_video": False,
                },
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

            json_file = json_folder / "IMG_0001.json"
            with open(json_file, "w") as f:
                json.dump(test_json, f)

            df = load_all_detections(str(json_folder))

            assert len(df) == 1
            assert df.iloc[0]["file_id"] == "cam1_IMG_0001"
            assert df.iloc[0]["group_id"] == "cam1"
            assert df.iloc[0]["frame_id"] == 0
            assert df.iloc[0]["tracking_id"] == 5
            assert df.iloc[0]["conf"] == 0.95
            assert bool(df.iloc[0]["false_positive_video"]) is False
            assert bool(df.iloc[0]["false_positive_detection"]) is False

    def test_loads_multiple_json_files(self):
        """Test loading detections from multiple JSON files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_folder = Path(tmpdir)

            for i in range(1, 3):
                test_json = {
                    "info": {
                        "video_file": f"cam1/IMG_000{i}.MP4",
                        "file_id": f"cam1_IMG_000{i}",
                        "group_id": "cam1",
                        "width": 1920,
                        "height": 1080,
                    },
                    "frames": [
                        {
                            "frame_id": 0,
                            "detections": [
                                {
                                    "bbox": [0.1, 0.2, 0.3, 0.4],
                                    "conf": 0.95,
                                    "category": "1",
                                    "track_id": i,
                                }
                            ],
                        }
                    ],
                }

                json_file = json_folder / f"IMG_000{i}.json"
                with open(json_file, "w") as f:
                    json.dump(test_json, f)

            df = load_all_detections(str(json_folder))

            assert len(df) == 2
            assert set(df["file_id"]) == {"cam1_IMG_0001", "cam1_IMG_0002"}

    def test_loads_recursive_json_files(self):
        """Test loading detections from nested folders"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_folder = Path(tmpdir)
            subfolder = json_folder / "site1" / "cam1"
            subfolder.mkdir(parents=True)

            test_json = {
                "info": {
                    "video_file": "site1/cam1/IMG_0001.MP4",
                    "file_id": "site1_cam1_IMG_0001",
                    "group_id": "site1/cam1",
                    "width": 1920,
                    "height": 1080,
                },
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": [
                            {
                                "bbox": [0.1, 0.2, 0.3, 0.4],
                                "conf": 0.95,
                                "category": "1",
                                "track_id": 1,
                            }
                        ],
                    }
                ],
            }

            json_file = subfolder / "IMG_0001.json"
            with open(json_file, "w") as f:
                json.dump(test_json, f)

            df = load_all_detections(str(json_folder))

            assert len(df) == 1
            assert df.iloc[0]["file_id"] == "site1_cam1_IMG_0001"
            assert df.iloc[0]["detection_file_path"] == str(json_file)

    def test_handles_multiple_detections_per_frame(self):
        """Test loading multiple detections from same frame"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_folder = Path(tmpdir)

            test_json = {
                "info": {
                    "video_file": "cam1/IMG_0001.MP4",
                    "file_id": "cam1_IMG_0001",
                    "group_id": "cam1",
                    "width": 1920,
                    "height": 1080,
                },
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": [
                            {
                                "bbox": [0.1, 0.2, 0.3, 0.4],
                                "conf": 0.95,
                                "category": "1",
                                "track_id": 1,
                            },
                            {
                                "bbox": [0.5, 0.6, 0.2, 0.3],
                                "conf": 0.88,
                                "category": "1",
                                "track_id": 2,
                            },
                        ],
                    }
                ],
            }

            json_file = json_folder / "IMG_0001.json"
            with open(json_file, "w") as f:
                json.dump(test_json, f)

            df = load_all_detections(str(json_folder))

            assert len(df) == 2
            assert list(df["tracking_id"]) == [1, 2]

    def test_reads_ids_from_info_regardless_of_filename(self):
        """Test that file_id/group_id come from the "info" block, not the JSON filename"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_folder = Path(tmpdir)

            test_json = {
                "info": {
                    "video_file": "any_name.mp4",
                    "file_id": "cam7_IMG_0042",
                    "group_id": "cam7",
                    "width": 1920,
                    "height": 1080,
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

            json_file = json_folder / "unrelated_name.json"
            with open(json_file, "w") as f:
                json.dump(test_json, f)

            df = load_all_detections(str(json_folder))

            assert df.iloc[0]["file_id"] == "cam7_IMG_0042"
            assert df.iloc[0]["group_id"] == "cam7"
            assert df.iloc[0]["tracking_id"] is None


class TestOutputsDf:
    """Test outputs_df function"""

    @patch("data_processing.utils.megadetector_utils.MDVideoOutputParser")
    def test_returns_dataframe(self, mock_parser_class):
        """Test that outputs_df returns a DataFrame"""
        mock_parser = MagicMock()
        mock_df = pd.DataFrame({"file_id": ["video1"], "frame_id": [0], "conf": [0.95]})
        mock_parser.to_dataframe.return_value = mock_df
        mock_parser_class.return_value = mock_parser

        result = outputs_df("test_folder")

        assert isinstance(result, pd.DataFrame)
        mock_parser.parse_folder.assert_called_once_with("test_folder")
        mock_parser.to_dataframe.assert_called_once()

    @patch("data_processing.utils.megadetector_utils.MDVideoOutputParser")
    def test_calls_parser_correctly(self, mock_parser_class):
        """Test that parser is called with correct folder"""
        mock_parser = MagicMock()
        mock_parser.to_dataframe.return_value = pd.DataFrame()
        mock_parser_class.return_value = mock_parser

        outputs_df("/path/to/md_folder")

        mock_parser.parse_folder.assert_called_once_with("/path/to/md_folder")
