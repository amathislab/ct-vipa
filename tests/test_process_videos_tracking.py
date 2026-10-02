"""Tests for process_videos_tracking module.

This comprehensive test suite validates the video tracking functionality including:
- ByteTrack (vendored in data_processing.utils.byte_tracker) integration
- DataFrame transformations for tracking data
- Track interpolation for missing detections
- Track smoothing using rolling median
- Complete tracking pipeline with various configurations

Coverage: ~66% of the module, focusing on core tracking logic.
The uncovered portions are primarily I/O and database integration code
(process_batch_video_folder, main) which require extensive mocking of
file systems and database connections.

Test Structure:
1. get_tracks_df: DataFrame normalization and type conversion (2 tests)
2. update_track_results: JSON format conversion (3 tests)
3. interpolate_missing_detections: Linear interpolation (3 tests)
4. smooth_tracks: Rolling median smoothing (3 tests)
5. tracks_post_processing: Callback orchestration (3 tests)
6. process_video_tracking: Main tracking pipeline (4 tests)

Total: 18 tests covering all core tracking functions.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from data_processing.process_videos_tracking import (
    extend_tracks_backward,
    get_tracks_df,
    interpolate_missing_detections,
    process_video_tracking,
    smooth_tracks,
    tracks_post_processing,
    update_track_results,
)


@pytest.fixture
def mock_detection_results():
    """Create mock detection results for testing."""
    return [
        {
            "frame_id": 0,
            "detections": [
                {"bbox": [10, 20, 50, 60], "conf": 0.9, "category": 1, "track_id": 1}
            ],
        },
        {
            "frame_id": 1,
            "detections": [
                {"bbox": [15, 25, 55, 65], "conf": 0.85, "category": 1, "track_id": 1}
            ],
        },
        {
            "frame_id": 2,
            "detections": [
                {"bbox": [20, 30, 60, 70], "conf": 0.88, "category": 1, "track_id": 1}
            ],
        },
    ]


@pytest.fixture
def mock_detections_df():
    """Create mock detections DataFrame for testing."""
    return pd.DataFrame(
        {
            "frame_id": [0, 1, 2],
            "detections": [
                [{"bbox": [10, 20, 50, 60], "conf": 0.9, "category": 1, "track_id": 1}],
                [
                    {
                        "bbox": [15, 25, 55, 65],
                        "conf": 0.85,
                        "category": 1,
                        "track_id": 1,
                    }
                ],
                [
                    {
                        "bbox": [20, 30, 60, 70],
                        "conf": 0.88,
                        "category": 1,
                        "track_id": 1,
                    }
                ],
            ],
        }
    )


@pytest.fixture
def mock_tracks_df():
    """Create mock tracks DataFrame for testing."""
    return pd.DataFrame(
        {
            "frame_id": [0, 1, 2],
            "x1": [10.0, 15.0, 20.0],
            "y1": [20.0, 25.0, 30.0],
            "x2": [50.0, 55.0, 60.0],
            "y2": [60.0, 65.0, 70.0],
            "conf": [0.9, 0.85, 0.88],
            "category": [1, 1, 1],
            "track_id": [1, 1, 1],
        }
    )


@pytest.fixture
def mock_tracking_params():
    """Create mock tracking parameters."""
    return {
        "track_activation_threshold": 0.5,
        "detection_threshold": 0.3,
        "minimum_matching_threshold_first_associations": 0.7,
        "minimum_matching_threshold_second_associations": 0.5,
        "lost_track_buffer": 30,
        "buffer_scale": 0.2,
        "backward_extension_max_gap": None,
        "interpolate_missing": False,
        "smooth_tracks": False,
        "smoothing_period": None,
    }


def test_get_tracks_df_basic(mock_detections_df):
    """Test get_tracks_df converts detections correctly."""
    # Explode and prepare the dataframe similar to tracks_post_processing
    detections_df = (
        mock_detections_df.explode("detections").dropna().reset_index(drop=True)
    )

    # get_tracks_df does the normalization itself
    result = get_tracks_df(detections_df)

    assert "frame_id" in result.columns
    assert "x1" in result.columns
    assert "y1" in result.columns
    assert "x2" in result.columns
    assert "y2" in result.columns
    assert "conf" in result.columns
    assert "category" in result.columns
    assert "track_id" in result.columns

    assert result["track_id"].dtype == int
    assert result["category"].dtype == int
    assert result["conf"].dtype == float

    assert result.iloc[0]["x1"] == 10
    assert result.iloc[0]["y1"] == 20
    assert result.iloc[0]["x2"] == 50
    assert result.iloc[0]["y2"] == 60
    assert result.iloc[0]["conf"] == 0.9
    assert result.iloc[0]["category"] == 1
    assert result.iloc[0]["track_id"] == 1


def test_get_tracks_df_multiple_tracks():
    """Test get_tracks_df handles multiple tracks correctly."""
    detections_df = pd.DataFrame(
        {
            "frame_id": [0, 0, 1, 1],
            "detections": [
                [{"bbox": [10, 20, 50, 60], "conf": 0.9, "category": 1, "track_id": 1}],
                [
                    {
                        "bbox": [100, 200, 150, 250],
                        "conf": 0.8,
                        "category": 2,
                        "track_id": 2,
                    }
                ],
                [
                    {
                        "bbox": [15, 25, 55, 65],
                        "conf": 0.85,
                        "category": 1,
                        "track_id": 1,
                    }
                ],
                [
                    {
                        "bbox": [105, 205, 155, 255],
                        "conf": 0.75,
                        "category": 2,
                        "track_id": 2,
                    }
                ],
            ],
        }
    )
    detections_df = detections_df.explode("detections").dropna().reset_index(drop=True)

    # get_tracks_df does the normalization itself
    result = get_tracks_df(detections_df)

    assert len(result) == 4

    assert set(result["track_id"].unique()) == {1, 2}

    assert set(result["category"].unique()) == {1, 2}


def test_update_track_results_basic(mock_tracks_df):
    """Test update_track_results converts DataFrame back to JSON format."""
    detection_results = [
        {"frame_id": 0, "detections": []},
        {"frame_id": 1, "detections": []},
        {"frame_id": 2, "detections": []},
    ]

    result = update_track_results(mock_tracks_df, detection_results)

    assert len(result) == 3
    assert all("detections" in frame for frame in result)

    frame_0_dets = result[0]["detections"]
    assert len(frame_0_dets) == 1
    assert frame_0_dets[0]["bbox"] == [10.0, 20.0, 50.0, 60.0]
    assert frame_0_dets[0]["conf"] == 0.9
    assert frame_0_dets[0]["category"] == 1
    assert frame_0_dets[0]["track_id"] == 1

    assert isinstance(frame_0_dets[0]["track_id"], int)
    assert isinstance(frame_0_dets[0]["category"], int)


def test_update_track_results_no_detections():
    """Test update_track_results handles frames with no detections."""
    tracks_df = pd.DataFrame(
        {
            "frame_id": [0, 2],  # Frame 1 has no detections
            "x1": [10.0, 20.0],
            "y1": [20.0, 30.0],
            "x2": [50.0, 60.0],
            "y2": [60.0, 70.0],
            "conf": [0.9, 0.88],
            "category": [1, 1],
            "track_id": [1, 1],
        }
    )

    detection_results = [
        {"frame_id": 0, "detections": []},
        {"frame_id": 1, "detections": []},
        {"frame_id": 2, "detections": []},
    ]

    result = update_track_results(tracks_df, detection_results)

    # Frame 0 should have detections
    assert len(result[0]["detections"]) == 1
    # Frame 1 should have no detections
    assert len(result[1]["detections"]) == 0
    # Frame 2 should have detections
    assert len(result[2]["detections"]) == 1


def test_update_track_results_sorted_by_frame():
    """Test update_track_results sorts by frame_id."""
    # Create unsorted tracks_df
    tracks_df = pd.DataFrame(
        {
            "frame_id": [2, 0, 1],
            "x1": [20.0, 10.0, 15.0],
            "y1": [30.0, 20.0, 25.0],
            "x2": [60.0, 50.0, 55.0],
            "y2": [70.0, 60.0, 65.0],
            "conf": [0.88, 0.9, 0.85],
            "category": [1, 1, 1],
            "track_id": [1, 1, 1],
        }
    )

    detection_results = [
        {"frame_id": 0, "detections": []},
        {"frame_id": 1, "detections": []},
        {"frame_id": 2, "detections": []},
    ]

    result = update_track_results(tracks_df, detection_results)

    assert result[0]["detections"][0]["bbox"][0] == 10.0  # Frame 0
    assert result[1]["detections"][0]["bbox"][0] == 15.0  # Frame 1
    assert result[2]["detections"][0]["bbox"][0] == 20.0  # Frame 2


def test_interpolate_missing_detections_basic():
    """Test interpolate_missing_detections fills gaps correctly."""
    # Create track with a missing frame
    track_df = pd.DataFrame(
        {
            "frame_id": [0, 2],  # Frame 1 is missing
            "x1": [10.0, 30.0],
            "y1": [20.0, 40.0],
            "x2": [50.0, 70.0],
            "y2": [60.0, 80.0],
            "conf": [0.9, 0.88],
            "category": [1, 1],
            "track_id": [1, 1],
        }
    )

    frame_ids_df = pd.DataFrame({"frame_id": [0, 1, 2]})

    result = interpolate_missing_detections(track_df, frame_ids_df)

    assert len(result) == 3

    frame_1 = result[result["frame_id"] == 1].iloc[0]
    assert frame_1["x1"] == 20.0  # (10 + 30) / 2
    assert frame_1["y1"] == 30.0  # (20 + 40) / 2
    assert frame_1["x2"] == 60.0  # (50 + 70) / 2
    assert frame_1["y2"] == 70.0  # (60 + 80) / 2


def test_interpolate_missing_detections_no_gaps():
    """Test interpolate_missing_detections with no gaps."""
    track_df = pd.DataFrame(
        {
            "frame_id": [0, 1, 2],
            "x1": [10.0, 15.0, 20.0],
            "y1": [20.0, 25.0, 30.0],
            "x2": [50.0, 55.0, 60.0],
            "y2": [60.0, 65.0, 70.0],
            "conf": [0.9, 0.85, 0.88],
            "category": [1, 1, 1],
            "track_id": [1, 1, 1],
        }
    )

    frame_ids_df = pd.DataFrame({"frame_id": [0, 1, 2]})

    result = interpolate_missing_detections(track_df, frame_ids_df)

    # Should be unchanged
    assert len(result) == 3
    assert result["x1"].tolist() == [10.0, 15.0, 20.0]


def test_interpolate_missing_detections_multiple_gaps():
    """Test interpolate_missing_detections with multiple gaps."""
    track_df = pd.DataFrame(
        {
            "frame_id": [0, 3, 6],  # Frames 1, 2, 4, 5 are missing
            "x1": [10.0, 40.0, 70.0],
            "y1": [20.0, 50.0, 80.0],
            "x2": [50.0, 80.0, 110.0],
            "y2": [60.0, 90.0, 120.0],
            "conf": [0.9, 0.85, 0.88],
            "category": [1, 1, 1],
            "track_id": [1, 1, 1],
        }
    )

    frame_ids_df = pd.DataFrame({"frame_id": [0, 1, 2, 3, 4, 5, 6]})

    result = interpolate_missing_detections(track_df, frame_ids_df)

    assert len(result) == 7

    frame_1 = result[result["frame_id"] == 1].iloc[0]
    assert np.isclose(frame_1["x1"], 20.0, atol=0.1)  # Linear interpolation


def test_smooth_tracks_basic():
    """Test smooth_tracks applies rolling median correctly."""
    track_df = pd.DataFrame(
        {
            "frame_id": [0, 1, 2, 3, 4],
            "x1": [10.0, 100.0, 20.0, 30.0, 40.0],  # Frame 1 is an outlier
            "y1": [20.0, 25.0, 30.0, 35.0, 40.0],
            "x2": [50.0, 55.0, 60.0, 65.0, 70.0],
            "y2": [60.0, 65.0, 70.0, 75.0, 80.0],
            "conf": [0.9, 0.85, 0.88, 0.87, 0.9],
            "category": [1, 1, 1, 1, 1],
            "track_id": [1, 1, 1, 1, 1],
        }
    )

    frame_ids_df = pd.DataFrame({"frame_id": [0, 1, 2, 3, 4]})
    smoothing_period = 3

    result = smooth_tracks(track_df, frame_ids_df, smoothing_period)

    assert len(result) == 5

    # The outlier at frame 1 (x1=100) should be smoothed towards nearby values
    # With a window of 3 centered on frame 1, median of [10, 100, 20] = 20
    assert result.iloc[1]["x1"] == 20.0


def test_smooth_tracks_window_size():
    """Test smooth_tracks with different window sizes."""
    track_df = pd.DataFrame(
        {
            "frame_id": list(range(10)),
            "x1": [10.0, 12.0, 11.0, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5, 16.0],
            "y1": [20.0] * 10,
            "x2": [50.0] * 10,
            "y2": [60.0] * 10,
            "conf": [0.9] * 10,
            "category": [1] * 10,
            "track_id": [1] * 10,
        }
    )

    frame_ids_df = pd.DataFrame({"frame_id": list(range(10))})

    # Small window
    result_small = smooth_tracks(track_df.copy(), frame_ids_df, smoothing_period=3)
    # Large window
    result_large = smooth_tracks(track_df.copy(), frame_ids_df, smoothing_period=7)

    # Larger window should produce smoother results (less variance)
    variance_small = result_small["x1"].var()
    variance_large = result_large["x1"].var()
    assert variance_large <= variance_small


def test_smooth_tracks_preserves_non_coordinate_columns():
    """Test smooth_tracks only modifies coordinate columns."""
    track_df = pd.DataFrame(
        {
            "frame_id": [0, 1, 2],
            "x1": [10.0, 15.0, 20.0],
            "y1": [20.0, 25.0, 30.0],
            "x2": [50.0, 55.0, 60.0],
            "y2": [60.0, 65.0, 70.0],
            "conf": [0.9, 0.85, 0.88],
            "category": [1, 1, 1],
            "track_id": [1, 1, 1],
        }
    )

    frame_ids_df = pd.DataFrame({"frame_id": [0, 1, 2]})

    result = smooth_tracks(track_df, frame_ids_df, smoothing_period=3)

    # Non-coordinate columns should be unchanged
    assert result["conf"].tolist() == [0.9, 0.85, 0.88]
    assert result["category"].tolist() == [1, 1, 1]
    assert result["track_id"].tolist() == [1, 1, 1]


def test_tracks_post_processing_with_callback(mock_detection_results):
    """Test tracks_post_processing applies callback to each track."""
    callback_calls = []

    def mock_callback(track_df, frame_ids_df):
        callback_calls.append(track_df["track_id"].iloc[0])
        return track_df

    result = tracks_post_processing(mock_detection_results, mock_callback)

    assert len(callback_calls) > 0
    assert isinstance(result, list)
    assert len(result) == len(mock_detection_results)


def test_tracks_post_processing_empty_detections():
    """Test tracks_post_processing handles empty detections."""
    results = [
        {"frame_id": 0, "detections": []},
        {"frame_id": 1, "detections": []},
    ]

    def mock_callback(track_df, frame_ids_df):
        return track_df

    result = tracks_post_processing(results, mock_callback)

    # Should return unchanged
    assert len(result) == 2
    assert result[0]["detections"] == []
    assert result[1]["detections"] == []


def test_tracks_post_processing_multiple_tracks():
    """Test tracks_post_processing handles multiple tracks correctly."""
    results = [
        {
            "frame_id": 0,
            "detections": [
                {"bbox": [10, 20, 50, 60], "conf": 0.9, "category": 1, "track_id": 1},
                {
                    "bbox": [100, 200, 150, 250],
                    "conf": 0.8,
                    "category": 2,
                    "track_id": 2,
                },
            ],
        },
        {
            "frame_id": 1,
            "detections": [
                {"bbox": [15, 25, 55, 65], "conf": 0.85, "category": 1, "track_id": 1},
                {
                    "bbox": [105, 205, 155, 255],
                    "conf": 0.75,
                    "category": 2,
                    "track_id": 2,
                },
            ],
        },
    ]

    callback_calls = []

    def mock_callback(track_df, frame_ids_df):
        callback_calls.append(track_df["track_id"].iloc[0])
        return track_df

    result = tracks_post_processing(results, mock_callback)

    # Should call callback for both tracks
    assert len(callback_calls) == 2
    assert set(callback_calls) == {1, 2}


@patch("data_processing.process_videos_tracking.ByteTrack")
@patch("data_processing.process_videos_tracking.md_utils")
def test_process_video_tracking_basic(
    mock_md_utils, mock_bytetrack, mock_tracking_params
):
    """Test process_video_tracking performs tracking on video."""
    mock_tracker_instance = Mock()
    mock_bytetrack.return_value = mock_tracker_instance

    mock_detections = Mock()
    mock_md_utils.from_megadetector_to_supervision.return_value = mock_detections
    mock_tracker_instance.update_with_detections.return_value = mock_detections
    # The converter returns relative float bboxes and string categories
    mock_md_utils.from_supervision_to_megadetector.return_value = [
        {"bbox": [0.1, 0.2, 0.3, 0.4], "conf": 0.9, "category": "1", "track_id": 1}
    ]

    results = [
        {
            "frame_id": 0,
            "detections": [{"bbox": [10, 20, 50, 60], "conf": 0.9, "category": 1}],
        },
        {
            "frame_id": 1,
            "detections": [{"bbox": [15, 25, 55, 65], "conf": 0.85, "category": 1}],
        },
    ]

    output = process_video_tracking(results, mock_tracking_params, 1920, 1080)

    mock_bytetrack.assert_called_once()
    call_kwargs = mock_bytetrack.call_args[1]
    assert call_kwargs["track_activation_threshold"] == 0.5
    assert call_kwargs["detection_threshold"] == 0.3
    assert call_kwargs["lost_track_buffer"] == 30

    mock_tracker_instance.reset.assert_called_once()

    assert isinstance(output, list)
    assert len(output) == 2


@patch("data_processing.process_videos_tracking.ByteTrack")
@patch("data_processing.process_videos_tracking.md_utils")
def test_process_video_tracking_with_interpolation(
    mock_md_utils, mock_bytetrack, mock_tracking_params
):
    """Test process_video_tracking with interpolation enabled."""
    mock_tracker_instance = Mock()
    mock_bytetrack.return_value = mock_tracker_instance

    mock_detections = Mock()
    mock_md_utils.from_megadetector_to_supervision.return_value = mock_detections
    mock_tracker_instance.update_with_detections.return_value = mock_detections
    # The converter returns relative float bboxes and string categories
    mock_md_utils.from_supervision_to_megadetector.return_value = [
        {"bbox": [0.1, 0.2, 0.3, 0.4], "conf": 0.9, "category": "1", "track_id": 1}
    ]

    results = [
        {
            "frame_id": 0,
            "detections": [
                {"bbox": [10, 20, 50, 60], "conf": 0.9, "category": 1, "track_id": 1}
            ],
        },
        {"frame_id": 1, "detections": []},  # Missing detection
        {
            "frame_id": 2,
            "detections": [
                {"bbox": [20, 30, 60, 70], "conf": 0.88, "category": 1, "track_id": 1}
            ],
        },
    ]

    params = mock_tracking_params.copy()
    params["interpolate_missing"] = True

    output = process_video_tracking(results, params, 1920, 1080)

    # Check that interpolation was applied (function should complete without error)
    assert isinstance(output, list)


@patch("data_processing.process_videos_tracking.ByteTrack")
@patch("data_processing.process_videos_tracking.md_utils")
def test_process_video_tracking_with_smoothing(
    mock_md_utils, mock_bytetrack, mock_tracking_params
):
    """Test process_video_tracking with smoothing enabled."""
    mock_tracker_instance = Mock()
    mock_bytetrack.return_value = mock_tracker_instance

    mock_detections = Mock()
    mock_md_utils.from_megadetector_to_supervision.return_value = mock_detections
    mock_tracker_instance.update_with_detections.return_value = mock_detections
    # The converter returns relative float bboxes and string categories
    mock_md_utils.from_supervision_to_megadetector.return_value = [
        {"bbox": [0.1, 0.2, 0.3, 0.4], "conf": 0.9, "category": "1", "track_id": 1}
    ]

    results = [
        {
            "frame_id": i,
            "detections": [
                {
                    "bbox": [10 + i * 5, 20 + i * 5, 50 + i * 5, 60 + i * 5],
                    "conf": 0.9,
                    "category": 1,
                    "track_id": 1,
                }
            ],
        }
        for i in range(5)
    ]

    params = mock_tracking_params.copy()
    params["smooth_tracks"] = True
    params["smoothing_period"] = 3

    output = process_video_tracking(results, params, 1920, 1080)

    # Smoothing a constant track leaves its boxes unchanged
    assert isinstance(output, list)
    assert len(output) == 5
    for frame_result in output:
        assert frame_result["detections"][0]["bbox"] == [0.1, 0.2, 0.3, 0.4]


@patch("data_processing.process_videos_tracking.ByteTrack")
@patch("data_processing.process_videos_tracking.md_utils")
def test_process_video_tracking_smoothing_without_period_raises_error(
    mock_md_utils, mock_bytetrack, mock_tracking_params
):
    """Test process_video_tracking raises error when smoothing enabled without period."""
    mock_tracker_instance = Mock()
    mock_bytetrack.return_value = mock_tracker_instance

    mock_detections = Mock()
    mock_md_utils.from_megadetector_to_supervision.return_value = mock_detections
    mock_tracker_instance.update_with_detections.return_value = mock_detections
    mock_md_utils.from_supervision_to_megadetector.return_value = []

    results = [{"frame_id": 0, "detections": []}]

    # Enable smoothing without period
    params = mock_tracking_params.copy()
    params["smooth_tracks"] = True
    params["smoothing_period"] = None

    with pytest.raises(ValueError, match="Smoothing period must be specified"):
        process_video_tracking(results, params, 1920, 1080)


def test_extend_tracks_backward_adds_earlier_untracked_detections():
    """A track starting at frame 2 is extended with the nearby detections of frames 0 and 1,
    not with a far-away one."""
    near = {"bbox": [0.10, 0.10, 0.1, 0.1], "conf": 0.2, "category": "0"}
    far = {"bbox": [0.80, 0.80, 0.1, 0.1], "conf": 0.2, "category": "0"}
    tracked = {"bbox": [0.11, 0.10, 0.1, 0.1], "conf": 0.9, "category": "0", "track_id": 1}
    animal_detections = [[near, far], [near], [tracked]]
    results = [
        {"frame_id": 0, "detections": []},
        {"frame_id": 1, "detections": []},
        {"frame_id": 2, "detections": [tracked]},
    ]

    output = extend_tracks_backward(
        results,
        animal_detections,
        max_gap=30,
        matching_threshold=0.9,
        buffer_scale=0.2,
        video_width=1920,
        video_height=1080,
    )

    assert [[d["bbox"] for d in frame["detections"]] for frame in output] == [
        [near["bbox"]],
        [near["bbox"]],
        [tracked["bbox"]],
    ]
    assert all(d["track_id"] == 1 for frame in output for d in frame["detections"])


def test_extend_tracks_backward_stops_after_max_gap():
    near = {"bbox": [0.10, 0.10, 0.1, 0.1], "conf": 0.2, "category": "0"}
    tracked = {"bbox": [0.10, 0.10, 0.1, 0.1], "conf": 0.9, "category": "0", "track_id": 1}
    animal_detections = [[near], [], [], [tracked]]
    results = [{"frame_id": i, "detections": []} for i in range(3)] + [
        {"frame_id": 3, "detections": [tracked]}
    ]

    output = extend_tracks_backward(results, animal_detections, 2, 0.9, 0.2, 1920, 1080)

    assert output[0]["detections"] == []
