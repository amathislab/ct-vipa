"""
ByteTrack vendored from roboflow/supervision (MIT License,
https://github.com/roboflow/supervision), via the gabeva/supervision fork.

Modifications for sparse MegaDetector detections:
- Buffered IoU distance (C-BIoU, Yang et al. 2023) instead of IoU in both
  associations (`matching.buffered_iou_distance`): boxes are enlarged by
  `buffer_scale` times their size, more for lost tracks.
- Explicit `detection_threshold` and per-association matching thresholds.
- Tracks are activated (and get an ID) as soon as they are created.
- Each output track carries the index of the detection it was matched to.
"""

from .core import ByteTrack

__all__ = ["ByteTrack"]
