"""
Unit tests for VideoDetectionProcessor class.

Tests verify:
- Proper folder creation when they don't exist
- Error handling for invalid inputs
- Missing frame_id generation and filling with empty results
- Correct JSON structure in output files
- Thread pooling functionality
- Correct model selection and configuration
- Video metadata looked up from raw_videos_df by the video path relative to the input folder
- JSON outputs keep the raw folder structure and are named by file_id
"""

import tempfile
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from data_processing.utils.db_utils import file_id_from_path
from data_processing.utils.video_detection_processor import VideoDetectionProcessor


def build_raw_videos_df(
    raw_file_paths: List[str], is_IR: Optional[List[bool]] = None
) -> pd.DataFrame:
    """Build a Videos table from paths relative to the raw videos folder, used as input videos"""
    return pd.DataFrame(
        {
            "file_id": [file_id_from_path(Path(path)) for path in raw_file_paths],
            "group_id": [Path(path).parent.as_posix() for path in raw_file_paths],
            "event_id": [f"event_{i}" for i in range(len(raw_file_paths))],
            "raw_file_path": raw_file_paths,
            "video_path": raw_file_paths,
            "is_IR": is_IR if is_IR is not None else [False] * len(raw_file_paths),
        }
    )


class TestVideoDetectionProcessorInitialization:
    """Test initialization and folder setup"""

    def test_folder_creation_when_missing(self):
        """Test that output and frame folders are created if they don't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input_videos"
            input_folder.mkdir()
            project_path = Path(tmpdir) / "project"
            raw_videos_df = build_raw_videos_df([])

            params = {
                "model": "MDV6-yolov10-e",
                "json_confidence_threshold": 0.1,
                "output_json_folder_name": "detections",
                "tmp_frame_folder_name": "frames",
                "skip_existing": False,
                "force_frame_folder_deletion": False,
                "n_cores": 1,
                "batch_size": 16,
                "stride": 1,
                "debug_max_videos": -1,
            }

            processor = VideoDetectionProcessor(
                params, str(input_folder), str(project_path), raw_videos_df
            )

            assert processor.output_json_folder.exists()
            assert processor.frame_folder.exists()

    def test_video_metadata_looked_up_by_video_path(self):
        """Test that video metadata is retrieved from raw_videos_df using the path relative to the input folder"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input_videos"
            input_folder.mkdir()
            project_path = Path(tmpdir) / "project"
            raw_videos_df = build_raw_videos_df(
                ["cam1/IMG_0001.MP4", "cam2/sub/IMG_0002.MP4"]
            )

            params = {
                "model": "MDV6-yolov10-e",
                "json_confidence_threshold": 0.1,
                "output_json_folder_name": "detections",
                "tmp_frame_folder_name": "frames",
                "skip_existing": False,
                "force_frame_folder_deletion": False,
            }

            processor = VideoDetectionProcessor(
                params, str(input_folder), str(project_path), raw_videos_df
            )

            assert processor._lookup_video_meta(input_folder / "cam1/IMG_0001.MP4") == {
                "file_id": "cam1_IMG_0001",
                "group_id": "cam1",
                "event_id": "event_0",
                "raw_file_path": "cam1/IMG_0001.MP4",
            }
            assert (
                processor._lookup_video_meta(input_folder / "cam2/sub/IMG_0002.MP4")[
                    "file_id"
                ]
                == "cam2_sub_IMG_0002"
            )


class TestModelSelection:
    """Test correct model loading"""

    @patch("data_processing.utils.video_detection_processor.pw_detection")
    @patch("data_processing.utils.video_detection_processor.torch")
    def test_mdv6_model_loading(self, mock_torch, mock_pw_detection):
        """Test that MDV6 model is correctly loaded"""
        mock_torch.cuda.is_available.return_value = True
        mock_device = Mock()
        mock_torch.device.return_value = mock_device

        mock_model = MagicMock()
        mock_model.predictor.args.verbose = True
        mock_pw_detection.MegaDetectorV6.return_value = mock_model

        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input_videos"
            input_folder.mkdir()
            project_path = Path(tmpdir) / "project"
            raw_videos_df = build_raw_videos_df([])

            params = {
                "model": "MDV6-yolov10-e",
                "json_confidence_threshold": 0.1,
                "output_json_folder_name": "detections",
                "tmp_frame_folder_name": "frames",
                "skip_existing": False,
                "force_frame_folder_deletion": False,
            }

            processor = VideoDetectionProcessor(
                params, str(input_folder), str(project_path), raw_videos_df
            )
            model, runner = processor._get_model_and_runner()

            mock_pw_detection.MegaDetectorV6.assert_called_once_with(
                device=mock_device,
                pretrained=True,
                version="MDV6-yolov10-e",
            )
            assert model.predictor.args.verbose is False

    @patch("data_processing.utils.video_detection_processor.pw_detection")
    @patch("data_processing.utils.video_detection_processor.torch")
    def test_mdv5_model_loading(self, mock_torch, mock_pw_detection):
        """Test that MDV5 model is correctly loaded"""
        mock_torch.cuda.is_available.return_value = False
        mock_device = Mock()
        mock_torch.device.return_value = mock_device

        mock_model = MagicMock()
        mock_pw_detection.MegaDetectorV5.return_value = mock_model

        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input_videos"
            input_folder.mkdir()
            project_path = Path(tmpdir) / "project"
            raw_videos_df = build_raw_videos_df([])

            params = {
                "model": "a",
                "json_confidence_threshold": 0.1,
                "output_json_folder_name": "detections",
                "tmp_frame_folder_name": "frames",
                "skip_existing": False,
                "force_frame_folder_deletion": False,
            }

            processor = VideoDetectionProcessor(
                params, str(input_folder), str(project_path), raw_videos_df
            )
            model, runner = processor._get_model_and_runner()

            mock_pw_detection.MegaDetectorV5.assert_called_once_with(
                device=mock_device,
                pretrained=True,
                version="a",
            )

    def test_invalid_model_raises_error(self):
        """Test that invalid model name raises ValueError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input_videos"
            input_folder.mkdir()
            project_path = Path(tmpdir) / "project"
            raw_videos_df = build_raw_videos_df([])

            params = {
                "model": "invalid_model",
                "json_confidence_threshold": 0.1,
                "output_json_folder_name": "detections",
                "tmp_frame_folder_name": "frames",
                "skip_existing": False,
                "force_frame_folder_deletion": False,
            }

            processor = VideoDetectionProcessor(
                params, str(input_folder), str(project_path), raw_videos_df
            )

            with pytest.raises(ValueError, match="Unsupported model version"):
                processor._get_model_and_runner()


class TestJSONOutputStructure:
    """Test JSON output format and structure"""

    @patch("data_processing.utils.video_detection_processor.md_utils")
    @patch("data_processing.utils.video_detection_processor.pw_detection")
    @patch("data_processing.utils.video_detection_processor.torch")
    def test_json_has_required_fields(
        self, mock_torch, mock_pw_detection, mock_md_utils
    ):
        """Test that generated JSON has all required fields"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = Mock()
        mock_torch.set_grad_enabled = Mock()

        mock_model = MagicMock()
        mock_model.batch_image_detection.return_value = []
        mock_pw_detection.MegaDetectorV6.return_value = mock_model

        mock_md_utils.get_video_dimensions.return_value = (1920, 1080)
        mock_md_utils.get_total_frames.return_value = 10
        mock_md_utils.save_frames_to_disk = Mock()
        mock_md_utils.delete_frames_from_disk = Mock()

        written_data = {}

        def capture_write(
            frames,
            video_path,
            output_path,
            file_id,
            group_id,
            event_id,
            width,
            height,
            detector,
            detection_categories,
        ):
            written_data["frames"] = frames
            written_data["output_path"] = output_path
            written_data["file_id"] = file_id
            written_data["group_id"] = group_id
            written_data["event_id"] = event_id
            written_data["width"] = width
            written_data["height"] = height
            written_data["detector"] = detector
            written_data["detection_categories"] = detection_categories

        mock_md_utils.write_results_to_file.side_effect = capture_write

        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input_videos" / "cam1"
            input_folder.mkdir(parents=True)
            video_file = input_folder / "test_video.mp4"
            video_file.touch()

            project_path = Path(tmpdir) / "project"
            raw_videos_df = build_raw_videos_df(["cam1/test_video.mp4"])

            params = {
                "model": "MDV6-yolov10-e",
                "json_confidence_threshold": 0.1,
                "output_json_folder_name": "detections",
                "tmp_frame_folder_name": "frames",
                "skip_existing": False,
                "force_frame_folder_deletion": True,
                "n_cores": 1,
                "batch_size": 16,
                "stride": 1,
                "debug_max_videos": 1,
            }

            processor = VideoDetectionProcessor(
                params, str(input_folder.parent), str(project_path), raw_videos_df
            )

            detection_categories = {"0": "animal", "1": "person", "2": "vehicle"}
            processor.process_videos_folder(detection_categories, [".mp4"])

            assert mock_md_utils.write_results_to_file.called
            assert written_data["file_id"] == "cam1_test_video"
            assert written_data["group_id"] == "cam1"
            assert written_data["event_id"] == "event_0"
            assert written_data["output_path"] == str(
                project_path / "detections" / "cam1" / "cam1_test_video.json"
            )
            assert written_data["width"] == 1920
            assert written_data["height"] == 1080
            assert written_data["detector"] == "MDV6-yolov10-e"
            assert written_data["detection_categories"] == detection_categories

    @patch("data_processing.utils.video_detection_processor.cv2")
    @patch("data_processing.utils.video_detection_processor.md_utils")
    @patch("data_processing.utils.video_detection_processor.pw_detection")
    @patch("data_processing.utils.video_detection_processor.torch")
    def test_missing_frames_filled_with_empty_detections(
        self, mock_torch, mock_pw_detection, mock_md_utils, mock_cv2
    ):
        """Test that frames without detections are included with empty detection lists.

        This test simulates a model that only returns results for frames with detections:
        - Model returns detections for frames [0, 4, 6, 8] (frames 0, 2, 3, 4 in image files)
        - Video has 10 frames total
        - Stride is 2 (so we process frames 0, 2, 4, 6, 8)
        - Frame 2 should be automatically added with empty detections
        """
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = Mock()
        mock_torch.set_grad_enabled = Mock()

        mock_model = MagicMock()
        # Return detections only for frames 0, 4, 6, 8 (skipping frame 2)
        # Note: frame_0000.jpg = frame 0, frame_0002.jpg = frame 4 (with stride 2), etc.
        mock_model.batch_image_detection.return_value = [
            {"img_id": "frame_0000.jpg", "detections": Mock()},  # frame 0
            {"img_id": "frame_0002.jpg", "detections": Mock()},  # frame 4
            {"img_id": "frame_0003.jpg", "detections": Mock()},  # frame 6
            {"img_id": "frame_0004.jpg", "detections": Mock()},  # frame 8
        ]
        mock_pw_detection.MegaDetectorV6.return_value = mock_model

        mock_md_utils.get_video_dimensions.return_value = (1920, 1080)
        mock_md_utils.save_frames_to_disk = Mock()
        mock_md_utils.delete_frames_from_disk = Mock()
        mock_md_utils.from_supervision_to_megadetector.return_value = [
            {"bbox": [0.1, 0.2, 0.3, 0.4], "conf": 0.95, "category": "0"}
        ]

        mock_cap = MagicMock()
        mock_cap.get.return_value = 10  # 10 total frames
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FRAME_COUNT = 7  # This is the actual cv2 constant value

        # Capture the frames passed to write_results_to_file
        written_frames = []

        def capture_write(frames, *args, **kwargs):
            written_frames.extend(frames)

        mock_md_utils.write_results_to_file.side_effect = capture_write

        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input_videos" / "cam1"
            input_folder.mkdir(parents=True)
            video_file = input_folder / "test_video.mp4"
            video_file.touch()

            project_path = Path(tmpdir) / "project"
            raw_videos_df = build_raw_videos_df(["cam1/test_video.mp4"])

            params = {
                "model": "MDV6-yolov10-e",
                "json_confidence_threshold": 0.1,
                "output_json_folder_name": "detections",
                "tmp_frame_folder_name": "frames",
                "skip_existing": False,
                "force_frame_folder_deletion": True,
                "n_cores": 1,
                "batch_size": 16,
                "stride": 2,  # Process every 2nd frame
                "debug_max_videos": 1,
            }

            processor = VideoDetectionProcessor(
                params, str(input_folder.parent), str(project_path), raw_videos_df
            )

            detection_categories = {"0": "animal"}
            processor.process_videos_folder(detection_categories, [".mp4"])

            # Check that frames are generated for all frames with stride
            # Total frames = 10, stride = 2, so we should have frames: 0, 2, 4, 6, 8
            frame_ids = [f["frame_id"] for f in written_frames]
            assert 0 in frame_ids, "Frame 0 should be present"
            assert (
                2 in frame_ids
            ), "Frame 2 should be present (filled with empty detections)"
            assert 4 in frame_ids, "Frame 4 should be present"
            assert 6 in frame_ids, "Frame 6 should be present"
            assert 8 in frame_ids, "Frame 8 should be present"

            # Verify we have exactly 5 frames (no duplicates)
            assert len(frame_ids) == 5, f"Expected 5 frames, got {len(frame_ids)}"

            assert frame_ids == sorted(frame_ids), "Frames should be sorted"

            # Check that frame 2 has empty detections (since model didn't return it)
            frame_2_data = next(f for f in written_frames if f["frame_id"] == 2)
            assert (
                frame_2_data["detections"] == []
            ), "Frame 2 should have empty detections list"

    @patch("data_processing.utils.video_detection_processor.cv2")
    @patch("data_processing.utils.video_detection_processor.md_utils")
    @patch("data_processing.utils.video_detection_processor.pw_detection")
    @patch("data_processing.utils.video_detection_processor.torch")
    def test_missing_multiple_consecutive_frames_filled(
        self, mock_torch, mock_pw_detection, mock_md_utils, mock_cv2
    ):
        """Test that multiple consecutive missing frames are filled with empty detections.

        This tests a more extreme case where the model only returns first and last frame:
        - Model returns detections for frames [0, 8] only
        - Video has 10 frames total
        - Stride is 2 (so we process frames 0, 2, 4, 6, 8)
        - Frames 2, 4, 6 should all be automatically added with empty detections
        """
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = Mock()
        mock_torch.set_grad_enabled = Mock()

        mock_model = MagicMock()
        # Return detections only for first and last frames
        mock_model.batch_image_detection.return_value = [
            {"img_id": "frame_0000.jpg", "detections": Mock()},  # frame 0
            {"img_id": "frame_0004.jpg", "detections": Mock()},  # frame 8
        ]
        mock_pw_detection.MegaDetectorV6.return_value = mock_model

        mock_md_utils.get_video_dimensions.return_value = (1920, 1080)
        mock_md_utils.save_frames_to_disk = Mock()
        mock_md_utils.delete_frames_from_disk = Mock()
        mock_md_utils.from_supervision_to_megadetector.return_value = [
            {"bbox": [0.1, 0.2, 0.3, 0.4], "conf": 0.95, "category": "0"}
        ]

        mock_cap = MagicMock()
        mock_cap.get.return_value = 10  # 10 total frames
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FRAME_COUNT = 7  # This is the actual cv2 constant value

        # Capture the frames passed to write_results_to_file
        written_frames = []

        def capture_write(frames, *args, **kwargs):
            written_frames.extend(frames)

        mock_md_utils.write_results_to_file.side_effect = capture_write

        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input_videos" / "cam1"
            input_folder.mkdir(parents=True)
            video_file = input_folder / "test_video.mp4"
            video_file.touch()

            project_path = Path(tmpdir) / "project"
            raw_videos_df = build_raw_videos_df(["cam1/test_video.mp4"])

            params = {
                "model": "MDV6-yolov10-e",
                "json_confidence_threshold": 0.1,
                "output_json_folder_name": "detections",
                "tmp_frame_folder_name": "frames",
                "skip_existing": False,
                "force_frame_folder_deletion": True,
                "n_cores": 1,
                "batch_size": 16,
                "stride": 2,
                "debug_max_videos": 1,
            }

            processor = VideoDetectionProcessor(
                params, str(input_folder.parent), str(project_path), raw_videos_df
            )

            detection_categories = {"0": "animal"}
            processor.process_videos_folder(detection_categories, [".mp4"])

            frame_ids = [f["frame_id"] for f in written_frames]
            assert frame_ids == [
                0,
                2,
                4,
                6,
                8,
            ], f"Expected [0, 2, 4, 6, 8], got {frame_ids}"

            # Check that frames 2, 4, 6 have empty detections (model didn't return them)
            for frame_id in [2, 4, 6]:
                frame_data = next(
                    f for f in written_frames if f["frame_id"] == frame_id
                )
                assert (
                    frame_data["detections"] == []
                ), f"Frame {frame_id} should have empty detections list"

            # Check that frames 0, 8 have detections (model returned them)
            for frame_id in [0, 8]:
                frame_data = next(
                    f for f in written_frames if f["frame_id"] == frame_id
                )
                assert (
                    len(frame_data["detections"]) > 0
                ), f"Frame {frame_id} should have detections"


class TestSkipExisting:
    """Test skip_existing functionality"""

    @patch("data_processing.utils.video_detection_processor.md_utils")
    @patch("data_processing.utils.video_detection_processor.pw_detection")
    @patch("data_processing.utils.video_detection_processor.torch")
    def test_skip_existing_videos(self, mock_torch, mock_pw_detection, mock_md_utils):
        """Test that existing JSON files are skipped when skip_existing=True"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = Mock()
        mock_torch.set_grad_enabled = Mock()

        mock_model = MagicMock()
        mock_pw_detection.MegaDetectorV6.return_value = mock_model

        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input_videos" / "cam1"
            input_folder.mkdir(parents=True)
            video_file = input_folder / "test_video.mp4"
            video_file.touch()

            project_path = Path(tmpdir) / "project"
            raw_videos_df = build_raw_videos_df(["cam1/test_video.mp4"])
            existing_json = project_path / "detections" / "cam1" / "cam1_test_video.json"
            existing_json.parent.mkdir(parents=True)
            existing_json.write_text("{}")

            params = {
                "model": "MDV6-yolov10-e",
                "json_confidence_threshold": 0.1,
                "output_json_folder_name": "detections",
                "tmp_frame_folder_name": "frames",
                "skip_existing": True,
                "force_frame_folder_deletion": True,
                "n_cores": 1,
                "batch_size": 16,
                "stride": 1,
                "debug_max_videos": 1,
            }

            processor = VideoDetectionProcessor(
                params, str(input_folder.parent), str(project_path), raw_videos_df
            )

            detection_categories = {"0": "animal"}
            processor.process_videos_folder(detection_categories, [".mp4"])

            # Verify that save_frames_to_disk was NOT called (video was skipped)
            mock_md_utils.save_frames_to_disk.assert_not_called()


class TestThreadPooling:
    """Test thread pooling functionality"""

    @patch("data_processing.utils.video_detection_processor.md_utils")
    @patch("data_processing.utils.video_detection_processor.pw_detection")
    @patch("data_processing.utils.video_detection_processor.torch")
    @patch(
        "data_processing.utils.video_detection_processor.concurrent.futures.ThreadPoolExecutor"
    )
    def test_thread_pool_created_with_n_cores(
        self, mock_executor_class, mock_torch, mock_pw_detection, mock_md_utils
    ):
        """Test that ThreadPoolExecutor is created with correct number of workers"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = Mock()
        mock_torch.set_grad_enabled = Mock()

        mock_model = MagicMock()
        mock_pw_detection.MegaDetectorV6.return_value = mock_model

        mock_executor = MagicMock()
        mock_executor.__enter__ = Mock(return_value=mock_executor)
        mock_executor.__exit__ = Mock(return_value=False)
        mock_executor.map.return_value = []
        mock_executor_class.return_value = mock_executor

        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input_videos" / "cam1"
            input_folder.mkdir(parents=True)

            # Create a dummy video file so the processing loop runs
            video_file = input_folder / "test_video.mp4"
            video_file.touch()

            project_path = Path(tmpdir) / "project"
            raw_videos_df = build_raw_videos_df(["cam1/test_video.mp4"])

            params = {
                "model": "MDV6-yolov10-e",
                "json_confidence_threshold": 0.1,
                "output_json_folder_name": "detections",
                "tmp_frame_folder_name": "frames",
                "skip_existing": False,
                "force_frame_folder_deletion": True,
                "n_cores": 4,
                "batch_size": 16,
                "stride": 1,
                "debug_max_videos": -1,
            }

            processor = VideoDetectionProcessor(
                params, str(input_folder.parent), str(project_path), raw_videos_df
            )

            detection_categories = {"0": "animal"}
            processor.process_videos_folder(detection_categories, [".mp4"])

            mock_executor_class.assert_called_once_with(max_workers=4)

    @patch("data_processing.utils.video_detection_processor.md_utils")
    @patch("data_processing.utils.video_detection_processor.pw_detection")
    @patch("data_processing.utils.video_detection_processor.torch")
    def test_thread_specific_frame_folders(
        self, mock_torch, mock_pw_detection, mock_md_utils
    ):
        """Test that each thread uses its own frame folder"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = Mock()
        mock_torch.set_grad_enabled = Mock()

        mock_model = MagicMock()
        mock_model.batch_image_detection.return_value = []
        mock_pw_detection.MegaDetectorV6.return_value = mock_model

        mock_md_utils.get_video_dimensions.return_value = (1920, 1080)
        mock_md_utils.get_total_frames.return_value = 10
        mock_md_utils.save_frames_to_disk = Mock()
        mock_md_utils.delete_frames_from_disk = Mock()
        mock_md_utils.write_results_to_file = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input_videos" / "cam1"
            input_folder.mkdir(parents=True)
            video_file = input_folder / "test_video.mp4"
            video_file.touch()

            project_path = Path(tmpdir) / "project"
            raw_videos_df = build_raw_videos_df(["cam1/test_video.mp4"])

            params = {
                "model": "MDV6-yolov10-e",
                "json_confidence_threshold": 0.1,
                "output_json_folder_name": "detections",
                "tmp_frame_folder_name": "frames",
                "skip_existing": False,
                "force_frame_folder_deletion": True,
                "n_cores": 2,
                "batch_size": 16,
                "stride": 1,
                "debug_max_videos": 1,
            }

            processor = VideoDetectionProcessor(
                params, str(input_folder.parent), str(project_path), raw_videos_df
            )

            detection_categories = {"0": "animal"}
            processor.process_videos_folder(detection_categories, [".mp4"])

            assert mock_md_utils.save_frames_to_disk.called
            call_args = mock_md_utils.save_frames_to_disk.call_args[0]
            thread_frame_folder = call_args[1]
            assert "thread_" in thread_frame_folder


class TestFrameFolderCleanup:
    """Test frame folder deletion behavior"""

    @patch("data_processing.utils.video_detection_processor.md_utils")
    @patch("data_processing.utils.video_detection_processor.pw_detection")
    @patch("data_processing.utils.video_detection_processor.torch")
    @patch("data_processing.utils.video_detection_processor.shutil")
    def test_frame_folder_deleted_when_force_deletion_true(
        self, mock_shutil, mock_torch, mock_pw_detection, mock_md_utils
    ):
        """Test that frame folder is deleted when force_frame_folder_deletion=True"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = Mock()
        mock_torch.set_grad_enabled = Mock()

        mock_model = MagicMock()
        mock_model.batch_image_detection.return_value = []
        mock_pw_detection.MegaDetectorV6.return_value = mock_model

        mock_md_utils.get_video_dimensions.return_value = (1920, 1080)
        mock_md_utils.get_total_frames.return_value = 10
        mock_md_utils.save_frames_to_disk = Mock()
        mock_md_utils.delete_frames_from_disk = Mock()
        mock_md_utils.write_results_to_file = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input_videos" / "cam1"
            input_folder.mkdir(parents=True)
            video_file = input_folder / "test_video.mp4"
            video_file.touch()

            project_path = Path(tmpdir) / "project"
            raw_videos_df = build_raw_videos_df(["cam1/test_video.mp4"])

            params = {
                "model": "MDV6-yolov10-e",
                "json_confidence_threshold": 0.1,
                "output_json_folder_name": "detections",
                "tmp_frame_folder_name": "frames",
                "skip_existing": False,
                "force_frame_folder_deletion": True,
                "n_cores": 1,
                "batch_size": 16,
                "stride": 1,
                "debug_max_videos": 1,
            }

            processor = VideoDetectionProcessor(
                params, str(input_folder.parent), str(project_path), raw_videos_df
            )

            detection_categories = {"0": "animal"}
            processor.process_videos_folder(detection_categories, [".mp4"])

            mock_shutil.rmtree.assert_called()


class TestVideoExtensions:
    """Test video file filtering by extension"""

    @patch("data_processing.utils.video_detection_processor.md_utils")
    @patch("data_processing.utils.video_detection_processor.pw_detection")
    @patch("data_processing.utils.video_detection_processor.torch")
    def test_only_specified_extensions_processed(
        self, mock_torch, mock_pw_detection, mock_md_utils
    ):
        """Test that only files with specified extensions are processed"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = Mock()
        mock_torch.set_grad_enabled = Mock()

        mock_model = MagicMock()
        mock_model.batch_image_detection.return_value = []
        mock_pw_detection.MegaDetectorV6.return_value = mock_model

        mock_md_utils.get_video_dimensions.return_value = (1920, 1080)
        mock_md_utils.get_total_frames.return_value = 10
        mock_md_utils.save_frames_to_disk = Mock()
        mock_md_utils.delete_frames_from_disk = Mock()
        mock_md_utils.write_results_to_file = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input_videos" / "cam1"
            input_folder.mkdir(parents=True)

            # Create files with different extensions
            (input_folder / "video1.mp4").touch()
            (input_folder / "video2.avi").touch()
            (input_folder / "video3.txt").touch()

            project_path = Path(tmpdir) / "project"
            raw_videos_df = build_raw_videos_df(
                ["cam1/video1.mp4", "cam1/video2.avi", "cam1/video3.txt"]
            )

            params = {
                "model": "MDV6-yolov10-e",
                "json_confidence_threshold": 0.1,
                "output_json_folder_name": "detections",
                "tmp_frame_folder_name": "frames",
                "skip_existing": False,
                "force_frame_folder_deletion": True,
                "n_cores": 1,
                "batch_size": 16,
                "stride": 1,
                "debug_max_videos": -1,
            }

            processor = VideoDetectionProcessor(
                params, str(input_folder.parent), str(project_path), raw_videos_df
            )

            detection_categories = {"0": "animal"}
            processor.process_videos_folder(detection_categories, [".mp4", ".avi"])

            # Verify save_frames_to_disk was called 2 times (for .mp4 and .avi only)
            assert mock_md_utils.save_frames_to_disk.call_count == 2


class TestDebugMaxVideos:
    """Test debug_max_videos parameter"""

    @patch("data_processing.utils.video_detection_processor.md_utils")
    @patch("data_processing.utils.video_detection_processor.pw_detection")
    @patch("data_processing.utils.video_detection_processor.torch")
    def test_debug_max_videos_limits_processing(
        self, mock_torch, mock_pw_detection, mock_md_utils
    ):
        """Test that debug_max_videos limits the number of videos processed"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = Mock()
        mock_torch.set_grad_enabled = Mock()

        mock_model = MagicMock()
        mock_model.batch_image_detection.return_value = []
        mock_pw_detection.MegaDetectorV6.return_value = mock_model

        mock_md_utils.get_video_dimensions.return_value = (1920, 1080)
        mock_md_utils.get_total_frames.return_value = 10
        mock_md_utils.save_frames_to_disk = Mock()
        mock_md_utils.delete_frames_from_disk = Mock()
        mock_md_utils.write_results_to_file = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input_videos" / "cam1"
            input_folder.mkdir(parents=True)

            for i in range(5):
                (input_folder / f"video{i}.mp4").touch()

            project_path = Path(tmpdir) / "project"
            raw_videos_df = build_raw_videos_df(
                [f"cam1/video{i}.mp4" for i in range(5)]
            )

            params = {
                "model": "MDV6-yolov10-e",
                "json_confidence_threshold": 0.1,
                "output_json_folder_name": "detections",
                "tmp_frame_folder_name": "frames",
                "skip_existing": False,
                "force_frame_folder_deletion": True,
                "n_cores": 1,
                "batch_size": 16,
                "stride": 1,
                "debug_max_videos": 2,  # Only process 2 videos
            }

            processor = VideoDetectionProcessor(
                params, str(input_folder.parent), str(project_path), raw_videos_df
            )

            detection_categories = {"0": "animal"}
            processor.process_videos_folder(detection_categories, [".mp4"])

            assert mock_md_utils.save_frames_to_disk.call_count == 2
