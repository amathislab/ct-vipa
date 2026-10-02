"""
Process a batch of videos for tracking (ByteTrack) and optionally interpolate and smooth tracks.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from supervision import box_iou_batch
from tqdm.auto import tqdm

from data_processing.utils import megadetector_utils as md_utils
from data_processing.utils.byte_tracker import ByteTrack
from data_processing.utils.byte_tracker.matching import (
    buffered_box_iou_batch,
    linear_assignment,
)
from data_processing.utils.db_utils import (
    get_previous_output,
    get_stage_parameters,
    load_database,
    parse_config,
    upsert_stage,
)

## Logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed output
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def process_video_tracking(
    results: List[Dict],
    params: Dict,
    video_width: int,
    video_height: int,
) -> List[Dict]:
    """
    Process a single video for tracking with ByteTrack (adapted)

    Args:
        - results: List of results dictionaries for all frames
        - params: other params
        - video_width: video width
        - video_height: video height

    Returns:
        The same results object with "track_id" for each detection
    """
    tracker = ByteTrack(
        track_activation_threshold=params["track_activation_threshold"],
        detection_threshold=params["detection_threshold"],
        minimum_matching_threshold_first_associations=params[
            "minimum_matching_threshold_first_associations"
        ],
        minimum_matching_threshold_second_associations=params[
            "minimum_matching_threshold_second_associations"
        ],
        lost_track_buffer=params["lost_track_buffer"],
        buffer_scale=params["buffer_scale"],
        frame_rate=30,
    )
    tracker.reset()

    animal_detections = []
    for frame_result in results:
        frame_result["detections"] = [
            detection
            for detection in frame_result["detections"]
        ]
        animal_detections.append(frame_result["detections"])
        detections = md_utils.from_megadetector_to_supervision(
            frame_result, video_width, video_height
        )
        detections = tracker.update_with_detections(
            detections
        )  # Updating detection to Bytetracker

        # Adds track_ids to detections of the frame_result
        frame_result["detections"] = md_utils.from_supervision_to_megadetector(
            detections, video_width, video_height
        )

    if params["backward_extension_max_gap"] is not None:
        results = extend_tracks_backward(
            results,
            animal_detections,
            params["backward_extension_max_gap"],
            params["minimum_matching_threshold_second_associations"],
            params["buffer_scale"],
            video_width,
            video_height,
        )

    # Apply Linear Interpolation
    if params["interpolate_missing"]:
        results = tracks_post_processing(
            results, callback=interpolate_missing_detections
        )

    # Apply Smoothing
    if params["smooth_tracks"]:
        smoothing_period = params["smoothing_period"]
        if smoothing_period is None:
            raise ValueError("Smoothing period must be specified in the parameters.")
        results = tracks_post_processing(
            results, callback=smooth_tracks, smoothing_period=smoothing_period
        )

    return results


def relative_xywh_to_xyxy(bboxes: List[List[float]], width: int, height: int) -> np.ndarray:
    xywh = np.array(bboxes, dtype=float).reshape(-1, 4) * [width, height, width, height]
    return np.column_stack([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]])


def extend_tracks_backward(
    results: List[Dict],
    animal_detections: List[List[Dict]],
    max_gap: int,
    matching_threshold: float,
    buffer_scale: float,
    video_width: int,
    video_height: int,
) -> List[Dict]:
    """
    Extend every track backwards from its first frame with the detections left untracked, of any
    confidence, so that a track starts with the first detections of its animal instead of the first
    confident one. A track keeps extending while it finds a match within max_gap frames, with the
    matching distance of the tracker.

    Args:
        - results: Tracked results for all frames
        - animal_detections: Animal detections of each frame given to the tracker
        - max_gap: Maximum number of frames between two matches of the extension
        - matching_threshold: Maximum buffered IoU distance between the track box and a detection
        - buffer_scale: Buffer scale of the buffered IoU distance
        - video_width: video width
        - video_height: video height
    """
    first_frame_of_track: Dict[int, int] = {}
    reference_box_of_track: Dict[int, np.ndarray] = {}
    for frame_id, frame_result in enumerate(results):
        for detection in frame_result["detections"]:
            if detection["track_id"] not in first_frame_of_track:
                first_frame_of_track[detection["track_id"]] = frame_id
                reference_box_of_track[detection["track_id"]] = relative_xywh_to_xyxy(
                    [detection["bbox"]], video_width, video_height
                )[0]
    last_match_of_track = dict(first_frame_of_track)

    for frame_id in range(max(first_frame_of_track.values(), default=0) - 1, -1, -1):
        track_ids = [
            track_id
            for track_id, first_frame in first_frame_of_track.items()
            if first_frame > frame_id and last_match_of_track[track_id] - frame_id <= max_gap
        ]
        candidate_detections = animal_detections[frame_id]
        if not track_ids or not candidate_detections:
            continue
        candidate_boxes = relative_xywh_to_xyxy(
            [detection["bbox"] for detection in candidate_detections], video_width, video_height
        )
        # Detections already tracked in this frame are not candidates
        tracked_boxes = relative_xywh_to_xyxy(
            [detection["bbox"] for detection in results[frame_id]["detections"]],
            video_width,
            video_height,
        )
        if len(tracked_boxes) > 0:
            is_untracked = box_iou_batch(candidate_boxes, tracked_boxes).max(axis=1) < 0.9
            candidate_detections = [d for d, keep in zip(candidate_detections, is_untracked) if keep]
            candidate_boxes = candidate_boxes[is_untracked]
        if not candidate_detections:
            continue

        track_boxes = np.array([reference_box_of_track[track_id] for track_id in track_ids])
        frames_since_match = np.array(
            [last_match_of_track[track_id] - frame_id for track_id in track_ids]
        )
        cost_matrix = 1 - buffered_box_iou_batch(
            track_boxes, candidate_boxes, buffer_scale * np.sqrt(frames_since_match)
        )
        matches, _, _ = linear_assignment(cost_matrix, matching_threshold)
        for i_track, i_candidate in matches:
            track_id = track_ids[i_track]
            detection = candidate_detections[i_candidate]
            results[frame_id]["detections"].append(
                {
                    "bbox": detection["bbox"],
                    "conf": detection["conf"],
                    "category": detection["category"],
                    "track_id": track_id,
                }
            )
            reference_box_of_track[track_id] = candidate_boxes[i_candidate]
            last_match_of_track[track_id] = frame_id

    return results


def tracks_post_processing(
    results: List[Dict], callback, **callback_kwargs
) -> List[Dict]:

    detections_df = pd.DataFrame(results)
    frame_ids_df = detections_df[["frame_id"]].copy()
    detections_df = detections_df.explode("detections").dropna().reset_index(drop=True)

    if len(detections_df) > 0:

        detections_df = get_tracks_df(detections_df)
        tracks_df = []
        track_ids = detections_df["track_id"].unique()

        for track_id in track_ids:
            # For every track, we get the corresponding detections for every frame within this track
            track_df = detections_df[detections_df["track_id"] == track_id]
            updated_track_df = callback(track_df, frame_ids_df, **callback_kwargs)
            tracks_df.append(updated_track_df)

        # We collect the result in a dataframe and update the original detections dict
        tracks_df = pd.concat(tracks_df, axis=0)
        results = update_track_results(tracks_df, results)

    return results


def get_tracks_df(detections_df):
    # We create separate columns for every element of the detection dictionary
    detections_df = pd.concat(
        [
            detections_df.drop("detections", axis=1),
            pd.json_normalize(detections_df["detections"]),
        ],
        axis=1,
    )
    detections_df = pd.concat(
        [
            detections_df.drop("bbox", axis=1),
            pd.DataFrame(
                detections_df["bbox"].to_list(), columns=["x1", "y1", "x2", "y2"]
            ),
        ],
        axis=1,
    )

    detections_df["track_id"] = detections_df["track_id"].astype(int)
    detections_df["category"] = detections_df["category"].astype(int)
    detections_df["conf"] = detections_df["conf"].astype(float)

    return detections_df


def update_track_results(tracks_df, detection_results):
    # We convert this dataframe back to the expected json results
    tracks_df["bbox"] = tracks_df.apply(lambda d: [d.x1, d.y1, d.x2, d.y2], axis=1)
    tracks_df = tracks_df.drop(["x1", "y1", "x2", "y2"], axis=1).sort_values("frame_id")
    tracks_df["track_id"] = tracks_df["track_id"].astype(int)
    tracks_df["category"] = tracks_df["category"].astype(int)

    for frame_id, frame_result in enumerate(detection_results):
        updated_detections_df = tracks_df[tracks_df["frame_id"] == frame_id][
            ["conf", "category", "track_id", "bbox"]
        ]
        if len(updated_detections_df) > 0:
            frame_result["detections"] = updated_detections_df.to_dict(orient="records")

    return detection_results


def interpolate_missing_detections(
    track_df: pd.DataFrame, frame_ids_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Linearly interpolate the detections of a track on the frames where it was not detected.

    Args:
        track_df: Detections of a single track, one row per frame.
        frame_ids_df: All frame ids of the video.

    Returns:
        Detections of the track on every frame between its first and last detection.
    """
    track_df = frame_ids_df.merge(
        track_df,
        on="frame_id",
        how="outer",
    )

    track_df = track_df.interpolate(method="linear", limit_area="inside").dropna()

    return track_df


def smooth_tracks(
    track_df: pd.DataFrame, frame_ids_df: pd.DataFrame, smoothing_period: int
) -> pd.DataFrame:
    """Apply a centered rolling median of smoothing_period frames to the boxes of a track."""

    track_df.loc[:, ["x1", "y1", "x2", "y2"]] = (
        track_df[["frame_id", "x1", "y1", "x2", "y2"]]
        .rolling(
            window=smoothing_period,
            center=True,
            min_periods=1,
            closed="both",
            on="frame_id",
        )
        .median()
    )

    return track_df


def process_batch_video_folder(params):
    """Process a batch of videos for tracking."""
    # Get parameters and setup
    config = parse_config(params.config_file, logger=logger)
    params = get_stage_parameters(config, Path(__file__).stem)
    project_path = config["global"]["project_path"]
    output_json_folder = Path(project_path) / params["output_json_folder_name"]

    conn = load_database(config["global"]["database_path"])

    # All json files in the folder
    input_json_folder = get_previous_output(
        conn=conn,
        stage_name="process_videos_dense_detection",
        output_type="json",
    )

    input_json_files = list(Path(input_json_folder).rglob("*.json"))

    start_time = time.time()
    for input_json_file in tqdm(
        input_json_files, leave=True, position=0, desc="Videos processed"
    ):

        # Load json results from input_json_file_rel
        logger.debug("Loading results from %s", input_json_file)
        with open(input_json_file, "r") as f:
            detection_annotations = json.load(f)

        width = int(detection_annotations["info"]["width"])
        height = int(detection_annotations["info"]["height"])

        # Processing the video and saving the result with annotated detections and classifications
        output_json_file = output_json_folder / input_json_file.relative_to(
            input_json_folder
        )

        if output_json_file.exists() and params.get("skip_existing", False):
            continue
        os.makedirs(output_json_file.parent, exist_ok=True)

        # Apply ByteTrack
        detection_results = detection_annotations["frames"]
        tracking_results = process_video_tracking(
            detection_results, params, width, height
        )

        # Write results to file
        tracking_annotations = detection_annotations.copy()
        tracking_annotations["frames"] = tracking_results
        with open(output_json_file, "w") as f:
            json.dump(tracking_annotations, f, indent=2)

    elapsed = time.time() - start_time
    logger.info(
        "Processed %d videos in %.2f seconds",
        len(input_json_files),
        elapsed,
    )
    # Merge md results
    upsert_stage(
        conn=conn,
        stage_name=str(Path(__file__).stem),
        parameters_dict=params,
        output_json_folder=output_json_folder,
        output_video_folder=None,
        examination_csv_path=None,
    )

    conn.close()

    return


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, required=True, help="Path to the configuration file."
    )

    params = parser.parse_args()

    process_batch_video_folder(params)


if __name__ == "__main__":
    main()
