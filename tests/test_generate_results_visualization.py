"""
Unit tests for generate_results_visualization script.

Tests verify:
- Video processing with correct annotation types (detection vs tracking)
- Video re-encoding with correct codec
- ProcessPoolExecutor parallelization
- Skip existing videos functionality
- Proper output directory creation
- JSON detection file handling
- Frame annotation with labels and bounding boxes
"""

import json
import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest

from visualization.generate_results_visualization import (
    process_batch_video_folder,
    process_single_video,
    process_video,
    reencode_video,
)


class TestProcessVideo:
    """Test individual video processing with annotations"""

    @patch("visualization.generate_results_visualization.sv.LabelAnnotator")
    @patch("visualization.generate_results_visualization.sv.ColorAnnotator")
    @patch("visualization.generate_results_visualization.sv.BoxAnnotator")
    @patch("visualization.generate_results_visualization.sv.VideoSink")
    @patch("visualization.generate_results_visualization.sv.VideoInfo")
    @patch("visualization.generate_results_visualization.sv.get_video_frames_generator")
    @patch(
        "visualization.generate_results_visualization.from_megadetector_to_supervision"
    )
    def test_process_video_detection_type(
        self,
        mock_from_md,
        mock_frame_gen,
        mock_video_info_cls,
        mock_video_sink,
        mock_box_annotator,
        mock_color_annotator,
        mock_label_annotator,
    ):
        """Test that detection visualization type creates correct annotators"""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            json_path = Path(tmpdir) / "test.json"
            output_path = Path(tmpdir) / "output.mp4"

            video_path.touch()

            detection_data = {
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": [
                            {
                                "bbox": [0.1, 0.2, 0.3, 0.4],
                                "conf": 0.95,
                                "category": "0",
                            }
                        ],
                    }
                ],
                "info": {"false_positive_video": False},
            }
            with open(json_path, "w") as f:
                json.dump(detection_data, f)

            mock_video_info = Mock()
            mock_video_info.width = 1920
            mock_video_info.height = 1080
            mock_video_info_cls.from_video_path.return_value = mock_video_info

            mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            mock_frame_gen.return_value = [mock_frame]

            mock_detections = Mock()
            mock_detections.class_id = np.array([0])
            mock_detections.confidence = np.array([0.95])
            mock_detections.data = {}
            mock_from_md.return_value = mock_detections

            mock_box_inst = Mock()
            mock_box_inst.annotate.return_value = mock_frame
            mock_box_annotator.return_value = mock_box_inst

            mock_color_inst = Mock()
            mock_color_inst.annotate.return_value = mock_frame
            mock_color_annotator.return_value = mock_color_inst

            mock_label_inst = Mock()
            mock_label_inst.annotate.return_value = mock_frame
            mock_label_annotator.return_value = mock_label_inst

            mock_sink_instance = MagicMock()
            mock_video_sink.return_value.__enter__.return_value = mock_sink_instance

            process_video(
                str(video_path), str(json_path), str(output_path), "detection"
            )

            mock_video_sink.assert_called_once()
            assert mock_video_sink.call_args[1]["target_path"] == str(output_path)
            assert mock_video_sink.call_args[1]["video_info"] == mock_video_info

            mock_sink_instance.write_frame.assert_called_once()

    @patch("visualization.generate_results_visualization.sv.LabelAnnotator")
    @patch("visualization.generate_results_visualization.sv.ColorAnnotator")
    @patch("visualization.generate_results_visualization.sv.BoxAnnotator")
    @patch("visualization.generate_results_visualization.sv.VideoSink")
    @patch("visualization.generate_results_visualization.sv.VideoInfo")
    @patch("visualization.generate_results_visualization.sv.get_video_frames_generator")
    @patch(
        "visualization.generate_results_visualization.from_megadetector_to_supervision"
    )
    def test_process_video_tracking_type(
        self,
        mock_from_md,
        mock_frame_gen,
        mock_video_info_cls,
        mock_video_sink,
        mock_box_annotator,
        mock_color_annotator,
        mock_label_annotator,
    ):
        """Test that tracking visualization type creates correct annotators with tracker IDs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            json_path = Path(tmpdir) / "test.json"
            output_path = Path(tmpdir) / "output.mp4"

            video_path.touch()

            detection_data = {
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": [
                            {
                                "bbox": [0.1, 0.2, 0.3, 0.4],
                                "conf": 0.95,
                                "category": "0",
                                "tracker_id": 1,
                            }
                        ],
                    }
                ],
                "info": {},
            }
            with open(json_path, "w") as f:
                json.dump(detection_data, f)

            mock_video_info = Mock()
            mock_video_info.width = 1920
            mock_video_info.height = 1080
            mock_video_info_cls.from_video_path.return_value = mock_video_info

            mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            mock_frame_gen.return_value = [mock_frame]

            mock_detections = Mock()
            mock_detections.tracker_id = np.array([1])
            mock_detections.confidence = np.array([0.95])
            mock_detections.data = {}
            mock_from_md.return_value = mock_detections

            mock_box_inst = Mock()
            mock_box_inst.annotate.return_value = mock_frame
            mock_box_annotator.return_value = mock_box_inst

            mock_color_inst = Mock()
            mock_color_inst.annotate.return_value = mock_frame
            mock_color_annotator.return_value = mock_color_inst

            mock_label_inst = Mock()
            mock_label_inst.annotate.return_value = mock_frame
            mock_label_annotator.return_value = mock_label_inst

            mock_sink_instance = MagicMock()
            mock_video_sink.return_value.__enter__.return_value = mock_sink_instance

            process_video(str(video_path), str(json_path), str(output_path), "tracking")

            mock_sink_instance.write_frame.assert_called_once()

    def test_process_video_invalid_visualization_type(self):
        """Test that invalid visualization type raises NotImplementedError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            json_path = Path(tmpdir) / "test.json"
            output_path = Path(tmpdir) / "output.mp4"

            video_path.touch()
            detection_data = {"frames": [], "info": {}}
            with open(json_path, "w") as f:
                json.dump(detection_data, f)

            with pytest.raises(
                NotImplementedError, match="Support only detection and tracking"
            ):
                process_video(
                    str(video_path), str(json_path), str(output_path), "invalid_type"
                )

    @patch("visualization.generate_results_visualization.sv.LabelAnnotator")
    @patch("visualization.generate_results_visualization.sv.ColorAnnotator")
    @patch("visualization.generate_results_visualization.sv.BoxAnnotator")
    @patch("visualization.generate_results_visualization.sv.VideoSink")
    @patch("visualization.generate_results_visualization.sv.VideoInfo")
    @patch("visualization.generate_results_visualization.sv.get_video_frames_generator")
    def test_process_video_empty_detections(
        self,
        mock_frame_gen,
        mock_video_info_cls,
        mock_video_sink,
        mock_box_annotator,
        mock_color_annotator,
        mock_label_annotator,
    ):
        """Test processing video with no detections"""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            json_path = Path(tmpdir) / "test.json"
            output_path = Path(tmpdir) / "output.mp4"

            video_path.touch()

            detection_data = {"frames": [], "info": {}}
            with open(json_path, "w") as f:
                json.dump(detection_data, f)

            mock_video_info = Mock()
            mock_video_info.width = 1920
            mock_video_info.height = 1080
            mock_video_info_cls.from_video_path.return_value = mock_video_info

            mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            mock_frame_gen.return_value = [mock_frame]

            mock_sink_instance = MagicMock()
            mock_video_sink.return_value.__enter__.return_value = mock_sink_instance

            process_video(
                str(video_path), str(json_path), str(output_path), "detection"
            )

            # Verify frame was still written (unannotated)
            mock_sink_instance.write_frame.assert_called_once()


class TestReencodeVideo:
    """Test video re-encoding functionality"""

    @patch("visualization.generate_results_visualization.os.rename")
    @patch("visualization.generate_results_visualization.os.remove")
    @patch("visualization.generate_results_visualization.ffmpeg")
    def test_reencode_video_with_libx264(self, mock_ffmpeg, mock_remove, mock_rename):
        """Test video re-encoding with libx264 codec"""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            video_path.touch()

            mock_input = Mock()
            mock_output = Mock()
            mock_ffmpeg.input.return_value = mock_input
            mock_input.output.return_value = mock_output

            reencode_video(str(video_path), codec="libx264")

            mock_ffmpeg.input.assert_called_once_with(str(video_path))
            mock_input.output.assert_called_once()

            call_args = mock_input.output.call_args
            assert "libx264" in str(call_args)
            assert call_args[1]["vcodec"] == "libx264"
            assert call_args[1]["pix_fmt"] == "yuv420p"
            assert call_args[1]["video_bitrate"] == "12048k"

            mock_output.run.assert_called_once_with(quiet=True)

            mock_remove.assert_called_once()
            mock_rename.assert_called_once()


class TestProcessSingleVideo:
    """Test single video processing wrapper"""

    @patch("visualization.generate_results_visualization.reencode_video")
    @patch("visualization.generate_results_visualization.process_video")
    def test_process_single_video_success(self, mock_process, mock_reencode):
        """Test successful single video processing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input"
            json_folder = Path(tmpdir) / "json"
            output_folder = Path(tmpdir) / "output"

            input_folder.mkdir()
            json_folder.mkdir()
            output_folder.mkdir()

            video_file = input_folder / "test.mp4"
            video_file.touch()
            json_file = json_folder / "test.json"
            json_file.write_text("{}")

            video_info = (
                video_file,
                input_folder,
                json_folder,
                output_folder,
                False,  # skip_existing
                "detection",
            )

            result = process_single_video(video_info, reencode=True, codec="libx264")

            assert result is True
            mock_process.assert_called_once()
            mock_reencode.assert_called_once()

    @patch("visualization.generate_results_visualization.process_video")
    def test_process_single_video_skip_existing(self, mock_process):
        """Test that existing videos are skipped when skip_existing=True"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input"
            json_folder = Path(tmpdir) / "json"
            output_folder = Path(tmpdir) / "output"

            input_folder.mkdir()
            json_folder.mkdir()
            output_folder.mkdir()

            video_file = input_folder / "test.mp4"
            video_file.touch()
            json_file = json_folder / "test.json"
            json_file.write_text("{}")

            # Create existing output video
            output_video = output_folder / "test.mp4"
            output_video.touch()

            video_info = (
                video_file,
                input_folder,
                json_folder,
                output_folder,
                True,  # skip_existing=True
                "detection",
            )

            result = process_single_video(video_info, reencode=False)

            assert result is False
            mock_process.assert_not_called()

    @patch("visualization.generate_results_visualization.process_video")
    def test_process_single_video_no_json(self, mock_process):
        """Test that videos without JSON are skipped"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input"
            json_folder = Path(tmpdir) / "json"
            output_folder = Path(tmpdir) / "output"

            input_folder.mkdir()
            json_folder.mkdir()
            output_folder.mkdir()

            video_file = input_folder / "test.mp4"
            video_file.touch()

            video_info = (
                video_file,
                input_folder,
                json_folder,
                output_folder,
                False,
                "detection",
            )

            result = process_single_video(video_info, reencode=False)

            assert result is False
            mock_process.assert_not_called()

    @patch("visualization.generate_results_visualization.reencode_video")
    @patch("visualization.generate_results_visualization.process_video")
    def test_process_single_video_creates_output_directory(
        self, mock_process, mock_reencode
    ):
        """Test that output directory is created if it doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input"
            json_folder = Path(tmpdir) / "json"
            output_folder = Path(tmpdir) / "output"

            input_folder.mkdir()
            json_folder.mkdir()
            # Don't create output_folder - it should be created automatically

            sub_input = input_folder / "subdir"
            sub_input.mkdir()
            video_file = sub_input / "test.mp4"
            video_file.touch()

            sub_json = json_folder / "subdir"
            sub_json.mkdir()
            json_file = sub_json / "test.json"
            json_file.write_text("{}")

            video_info = (
                video_file,
                input_folder,
                json_folder,
                output_folder,
                False,
                "detection",
            )

            result = process_single_video(video_info, reencode=False)

            assert result is True
            assert (output_folder / "subdir").exists()


class TestProcessBatchVideoFolder:
    """Test batch video processing with parallel execution"""

    @staticmethod
    def _mock_executor(results: List[bool]) -> MagicMock:
        mock_executor = MagicMock()
        mock_executor.__enter__ = Mock(return_value=mock_executor)
        mock_executor.__exit__ = Mock(return_value=False)
        mock_executor.map.return_value = results
        return mock_executor

    @patch("visualization.generate_results_visualization.ProcessPoolExecutor")
    def test_process_batch_video_folder_parallel_execution(self, mock_executor_cls):
        """Test that videos are processed in parallel with ProcessPoolExecutor"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input"
            json_folder = Path(tmpdir) / "json"
            output_folder = Path(tmpdir) / "output"

            (input_folder / "cam1").mkdir(parents=True)
            json_folder.mkdir()

            video1 = input_folder / "cam1" / "video1.mp4"
            video2 = input_folder / "cam1" / "video2.MP4"
            video1.touch()
            video2.touch()
            (input_folder / "cam1" / "notes.txt").touch()

            mock_executor = self._mock_executor([True, True])
            mock_executor_cls.return_value = mock_executor

            process_batch_video_folder(
                str(input_folder),
                str(json_folder),
                str(output_folder),
                skip_existing=False,
                visualization_type="detection",
                num_workers=2,
            )

            mock_executor_cls.assert_called_once_with(max_workers=2)

            # Each video found under the input folder (any extension case) is
            # mapped with the folders and options it needs
            mapped_function, mapped_args = mock_executor.map.call_args[0]
            assert mapped_function is process_single_video
            assert sorted(mapped_args) == sorted(
                [
                    (
                        video,
                        input_folder,
                        json_folder,
                        output_folder,
                        False,
                        "detection",
                    )
                    for video in [video1, video2]
                ]
            )

    @patch("visualization.generate_results_visualization.ProcessPoolExecutor")
    @patch(
        "visualization.generate_results_visualization.list_video_files",
        return_value=[],
    )
    def test_process_batch_video_folder_default_workers(
        self, mock_list_video_files, mock_executor_cls
    ):
        """Test that default number of workers is CPU count - 1"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input"
            json_folder = Path(tmpdir) / "json"
            output_folder = Path(tmpdir) / "output"

            input_folder.mkdir()
            json_folder.mkdir()

            mock_executor_cls.return_value = self._mock_executor([])

            with patch(
                "visualization.generate_results_visualization.multiprocessing.cpu_count"
            ) as mock_cpu_count:
                mock_cpu_count.return_value = 8

                # num_workers left to its default
                process_batch_video_folder(
                    str(input_folder),
                    str(json_folder),
                    str(output_folder),
                    skip_existing=False,
                    visualization_type="detection",
                )

                mock_executor_cls.assert_called_once_with(max_workers=7)

    @patch(
        "visualization.generate_results_visualization.list_video_files",
        return_value=[],
    )
    def test_process_batch_video_folder_passes_video_extensions(
        self, mock_list_video_files
    ):
        """Test that the requested video extensions are used to list input videos"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input"
            json_folder = Path(tmpdir) / "json"
            input_folder.mkdir()
            json_folder.mkdir()

            with patch(
                "visualization.generate_results_visualization.ProcessPoolExecutor"
            ):
                process_batch_video_folder(
                    str(input_folder),
                    str(json_folder),
                    str(Path(tmpdir) / "output"),
                    num_workers=1,
                    video_extensions=[".avi", ".mov"],
                )

            mock_list_video_files.assert_called_once_with(
                str(input_folder), [".avi", ".mov"]
            )

    @patch(
        "visualization.generate_results_visualization.list_video_files",
        return_value=[],
    )
    def test_process_batch_video_folder_creates_output_directory(
        self, mock_list_video_files
    ):
        """Test that output directory is created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input"
            json_folder = Path(tmpdir) / "json"
            output_folder = Path(tmpdir) / "output"

            input_folder.mkdir()
            json_folder.mkdir()
            # Don't create output_folder

            with patch(
                "visualization.generate_results_visualization.ProcessPoolExecutor"
            ):
                process_batch_video_folder(
                    str(input_folder),
                    str(json_folder),
                    str(output_folder),
                    skip_existing=False,
                    visualization_type="detection",
                    num_workers=1,
                )

                assert output_folder.exists()

    def test_process_batch_video_folder_invalid_input_folder(self):
        """Test that invalid input folder raises assertion error"""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "nonexistent"

            with pytest.raises(AssertionError):
                process_batch_video_folder(
                    str(nonexistent),
                    str(tmpdir),
                    str(tmpdir),
                    skip_existing=False,
                    visualization_type="detection",
                )

    def test_process_batch_video_folder_invalid_json_folder(self):
        """Test that invalid JSON folder raises assertion error"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input"
            input_folder.mkdir()
            nonexistent = Path(tmpdir) / "nonexistent"

            with pytest.raises(AssertionError):
                process_batch_video_folder(
                    str(input_folder),
                    str(nonexistent),
                    str(tmpdir),
                    skip_existing=False,
                    visualization_type="detection",
                )

    @patch("visualization.generate_results_visualization.ProcessPoolExecutor")
    @patch("visualization.generate_results_visualization.list_video_files")
    def test_process_batch_video_folder_counts_processed_videos(
        self, mock_list_video_files, mock_executor_cls
    ):
        """Test that processed video count is accurate"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "input"
            json_folder = Path(tmpdir) / "json"
            output_folder = Path(tmpdir) / "output"

            input_folder.mkdir()
            json_folder.mkdir()

            mock_list_video_files.return_value = [
                input_folder / "video1.mp4",
                input_folder / "video2.mp4",
                input_folder / "video3.mp4",
            ]

            # Simulate 2 processed, 1 skipped
            mock_executor_cls.return_value = self._mock_executor([True, True, False])

            with patch("builtins.print") as mock_print:
                process_batch_video_folder(
                    str(input_folder),
                    str(json_folder),
                    str(output_folder),
                    skip_existing=False,
                    visualization_type="detection",
                    num_workers=2,
                )

                print_calls = [str(call) for call in mock_print.call_args_list]
                assert any("Found 3 videos to process" in call for call in print_calls)
                assert any(
                    "Completed processing 2 videos out of 3" in call
                    for call in print_calls
                )
