import numpy as np

from supervision import Detections
from . import matching
from .kalman_filter import KalmanFilter
from .single_object_track import STrack, TrackState
from .utils import IdCounter


class ByteTrack:
    """
    ByteTrack adapted from supervision, with the buffered IoU distance and separate
    matching thresholds for each association step.

    Parameters:
        track_activation_threshold (float): Minimum confidence of an unmatched
            high-confidence detection to start a new track.
        detection_threshold (float): Confidence splitting the detections between the
            first association (at or above) and the second association (below).
        lost_track_buffer (int): Number of frames to buffer when a track is lost.
            Increasing lost_track_buffer enhances occlusion handling, significantly
            reducing the likelihood of track fragmentation or disappearance caused
            by brief detection gaps.
        minimum_matching_threshold_first_associations (float): Maximum buffered IoU
            distance between tracks and high-confidence detections. Higher values are
            more permissive.
        minimum_matching_threshold_second_associations (float): Maximum buffered IoU
            distance between the remaining tracks and low-confidence detections.
        buffer_scale (float): Boxes are enlarged on each side by buffer_scale times
            their size before computing the IoU, more for lost tracks.
        frame_rate (int): The frame rate of the video.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        detection_threshold: float = 0.2,
        lost_track_buffer: int = 30,
        minimum_matching_threshold_first_associations: float = 0.8,
        minimum_matching_threshold_second_associations: float = 0.5,
        buffer_scale: float = 0.2,
        frame_rate: int = 30,
    ):
        self.track_activation_threshold = track_activation_threshold
        self.minimum_matching_threshold_first_associations = (
            minimum_matching_threshold_first_associations
        )
        self.minimum_matching_threshold_second_associations = (
            minimum_matching_threshold_second_associations
        )
        self.det_thresh = detection_threshold
        self.buffer_scale = buffer_scale

        self.frame_id = 0
        self.max_time_lost = int(frame_rate / 30.0 * lost_track_buffer)
        self.kalman_filter = KalmanFilter()
        self.shared_kalman = KalmanFilter()

        self.tracked_tracks: list[STrack] = []
        self.lost_tracks: list[STrack] = []

        # Warning, possible bug: If you also set internal_id to start at 1,
        # all traces will be connected across objects.
        self.internal_id_counter = IdCounter()
        self.external_id_counter = IdCounter(start_id=1)

    def update_with_detections(self, detections: Detections) -> Detections:
        """
        Updates the tracker with the provided detections and returns the detections
        associated with a track, with their tracker_id.

        Args:
            detections (Detections): The detections to pass through the tracker.
        """
        tensors = np.hstack(
            (
                detections.xyxy,
                detections.confidence[:, np.newaxis],
            )
        )
        tracks = self.update_with_tensors(tensors=tensors)

        detections.tracker_id = np.full(len(detections), -1, dtype=int)
        for track in tracks:
            detections.tracker_id[track.detection_index] = track.external_track_id

        return detections[detections.tracker_id != -1]

    def reset(self) -> None:
        """
        Resets the internal state of the ByteTrack tracker.

        This method clears the tracking data, including tracked and lost tracks,
        as well as resetting the frame counter. It's particularly useful when
        processing multiple videos sequentially, ensuring the tracker starts with
        a clean state for each new video.
        """
        self.frame_id = 0
        self.internal_id_counter.reset()
        self.external_id_counter.reset()
        self.tracked_tracks = []
        self.lost_tracks = []

    def _to_stracks(self, tensors: np.ndarray, detection_indices: np.ndarray) -> list[STrack]:
        return [
            STrack(
                STrack.tlbr_to_tlwh(tensor[:4]),
                tensor[4],
                detection_index,
                self.shared_kalman,
                self.internal_id_counter,
                self.external_id_counter,
            )
            for tensor, detection_index in zip(tensors, detection_indices)
        ]

    def _associate(
        self,
        tracks: list[STrack],
        detections: list[STrack],
        thresh: float,
        fuse_score: bool,
    ) -> tuple[np.ndarray, tuple[int], tuple[int]]:
        dists = matching.buffered_iou_distance(
            tracks, detections, self.buffer_scale, self.frame_id
        )
        if fuse_score:
            dists = matching.fuse_score(dists, detections)
        return matching.linear_assignment(dists, thresh=thresh)

    def _update_matched_tracks(
        self,
        matches: np.ndarray,
        tracks: list[STrack],
        detections: list[STrack],
        activated_tracks: list[STrack],
        refound_tracks: list[STrack],
    ) -> None:
        for i_track, i_detection in matches:
            track = tracks[i_track]
            if track.state == TrackState.Tracked:
                track.update(detections[i_detection], self.frame_id)
                activated_tracks.append(track)
            else:
                track.re_activate(detections[i_detection], self.frame_id)
                refound_tracks.append(track)

    def update_with_tensors(self, tensors: np.ndarray) -> list[STrack]:
        """
        Updates the tracker with the provided tensors and returns the tracks
        matched to a detection in this frame.

        Parameters:
            tensors: The new tensors to update with, one [x1, y1, x2, y2, conf] row
                per detection.

        Returns:
            List[STrack]: Updated tracks.
        """
        self.frame_id += 1
        activated_tracks: list[STrack] = []
        refound_tracks: list[STrack] = []
        new_lost_tracks: list[STrack] = []
        removed_tracks: list[STrack] = []

        is_high = tensors[:, 4] >= self.det_thresh
        detection_indices = np.arange(len(tensors))
        detections_high = self._to_stracks(tensors[is_high], detection_indices[is_high])
        detections_low = self._to_stracks(tensors[~is_high], detection_indices[~is_high])

        """ First association, tracked and lost tracks with high score detection boxes"""
        track_pool = joint_tracks(self.tracked_tracks, self.lost_tracks)
        # Predict the current location with KF
        STrack.multi_predict(track_pool, self.shared_kalman)
        matches, unmatched_tracks, unmatched_high = self._associate(
            track_pool,
            detections_high,
            self.minimum_matching_threshold_first_associations,
            fuse_score=True,
        )
        self._update_matched_tracks(
            matches, track_pool, detections_high, activated_tracks, refound_tracks
        )

        """ Second association, remaining tracks with low score detection boxes"""
        remaining_tracks = [
            track_pool[i]
            for i in unmatched_tracks
            if track_pool[i].state == TrackState.Tracked
        ]
        matches, unmatched_tracks, _ = self._associate(
            remaining_tracks,
            detections_low,
            self.minimum_matching_threshold_second_associations,
            fuse_score=False,
        )
        self._update_matched_tracks(
            matches, remaining_tracks, detections_low, activated_tracks, refound_tracks
        )
        for i_track in unmatched_tracks:
            track = remaining_tracks[i_track]
            if track.state != TrackState.Lost:
                track.state = TrackState.Lost
                new_lost_tracks.append(track)

        """ Init new tracks from the unmatched high score detections"""
        for track in [detections_high[i] for i in unmatched_high]:
            if track.score < self.track_activation_threshold:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_tracks.append(track)

        """ Update state"""
        for track in self.lost_tracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.state = TrackState.Removed
                removed_tracks.append(track)

        self.tracked_tracks = [
            t for t in self.tracked_tracks if t.state == TrackState.Tracked
        ]
        self.tracked_tracks = joint_tracks(self.tracked_tracks, activated_tracks)
        self.tracked_tracks = joint_tracks(self.tracked_tracks, refound_tracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(new_lost_tracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, removed_tracks)
        self.tracked_tracks, self.lost_tracks = remove_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks
        )

        return self.tracked_tracks


def joint_tracks(
    track_list_a: list[STrack], track_list_b: list[STrack]
) -> list[STrack]:
    """
    Joins two lists of tracks, ensuring that the resulting list does not
    contain tracks with duplicate internal_track_id values.

    Parameters:
        track_list_a: First list of tracks (with internal_track_id attribute).
        track_list_b: Second list of tracks (with internal_track_id attribute).

    Returns:
        Combined list of tracks from track_list_a and track_list_b
            without duplicate internal_track_id values.
    """
    seen_track_ids = set()
    result = []

    for track in track_list_a + track_list_b:
        if track.internal_track_id not in seen_track_ids:
            seen_track_ids.add(track.internal_track_id)
            result.append(track)

    return result


def sub_tracks(track_list_a: list[STrack], track_list_b: list[STrack]) -> list[int]:
    """
    Returns a list of tracks from track_list_a after removing any tracks
    that share the same internal_track_id with tracks in track_list_b.

    Parameters:
        track_list_a: List of tracks (with internal_track_id attribute).
        track_list_b: List of tracks (with internal_track_id attribute) to
            be subtracted from track_list_a.
    Returns:
        List of remaining tracks from track_list_a after subtraction.
    """
    tracks = {track.internal_track_id: track for track in track_list_a}
    track_ids_b = {track.internal_track_id for track in track_list_b}

    for track_id in track_ids_b:
        tracks.pop(track_id, None)

    return list(tracks.values())


def remove_duplicate_tracks(
    tracks_a: list[STrack], tracks_b: list[STrack]
) -> tuple[list[STrack], list[STrack]]:
    pairwise_distance = matching.iou_distance(tracks_a, tracks_b)
    matching_pairs = np.where(pairwise_distance < 0.15)

    duplicates_a, duplicates_b = set(), set()
    for track_index_a, track_index_b in zip(*matching_pairs):
        time_a = tracks_a[track_index_a].frame_id - tracks_a[track_index_a].start_frame
        time_b = tracks_b[track_index_b].frame_id - tracks_b[track_index_b].start_frame
        if time_a > time_b:
            duplicates_b.add(track_index_b)
        else:
            duplicates_a.add(track_index_a)

    result_a = [
        track for index, track in enumerate(tracks_a) if index not in duplicates_a
    ]
    result_b = [
        track for index, track in enumerate(tracks_b) if index not in duplicates_b
    ]

    return result_a, result_b
