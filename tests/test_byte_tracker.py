"""Tests for the optional behaviours of the vendored ByteTrack (data_processing.utils.byte_tracker)."""

from typing import List, Tuple

import numpy as np
import supervision as sv

from data_processing.utils.byte_tracker import ByteTrack

Box = Tuple[float, float, float, float]


def make_detections(boxes: List[Box], confidences: List[float]) -> sv.Detections:
    if not boxes:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(boxes, dtype=float),
        confidence=np.array(confidences, dtype=float),
        class_id=np.zeros(len(boxes), dtype=int),
    )


def shifted(box: Box, dx: float) -> Box:
    return (box[0] + dx, box[1], box[2] + dx, box[3])


BOX: Box = (100.0, 100.0, 200.0, 200.0)
FAR_BOX: Box = shifted(BOX, 1000.0)


def test_returns_the_associated_detections():
    """Output boxes are the input detections, not the Kalman filter boxes."""
    tracker = ByteTrack(
        detection_threshold=0.5,
        track_activation_threshold=0.5,
        minimum_matching_threshold_first_associations=0.9,
    )
    tracker.update_with_detections(make_detections([BOX], [0.9]))
    jumped_box = shifted(BOX, 60.0)
    output = tracker.update_with_detections(make_detections([jumped_box], [0.9]))

    np.testing.assert_array_equal(output.xyxy, [jumped_box])
    assert output.tracker_id.tolist() == [1]


def test_each_detection_keeps_its_own_track():
    tracker = ByteTrack(detection_threshold=0.5, track_activation_threshold=0.5)
    other_box = shifted(BOX, 400.0)
    tracker.update_with_detections(make_detections([BOX, other_box], [0.9, 0.9]))
    output = tracker.update_with_detections(
        make_detections([other_box, BOX], [0.9, 0.9])
    )

    np.testing.assert_array_equal(output.xyxy, [other_box, BOX])
    assert output.tracker_id.tolist() == [2, 1]


def test_buffered_iou_rejects_far_matches():
    """Boxes that do not overlap once enlarged are at distance 1 and never match."""
    tracker = ByteTrack(
        detection_threshold=0.5,
        track_activation_threshold=0.5,
        minimum_matching_threshold_first_associations=0.9,
        buffer_scale=0.3,
    )
    tracker.update_with_detections(make_detections([BOX], [0.9]))
    output = tracker.update_with_detections(make_detections([FAR_BOX], [0.9]))

    assert output.tracker_id.tolist() == [2]


def test_buffered_iou_matches_close_non_overlapping_boxes():
    """A box moved by more than its width is still matched thanks to the buffer."""
    tracker = ByteTrack(
        detection_threshold=0.5,
        track_activation_threshold=0.5,
        minimum_matching_threshold_first_associations=0.9,
        buffer_scale=0.5,
    )
    tracker.update_with_detections(make_detections([BOX], [0.9]))
    output = tracker.update_with_detections(make_detections([shifted(BOX, 110.0)], [0.9]))

    assert output.tracker_id.tolist() == [1]
