"""
Generate video tublets along with corresponding dataframe from input videos and tracking results
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from tqdm import tqdm

from data_processing.utils.crop_processing import (
    adjust_crop_dimensions,
    dim_padding,
    normalize_bbox,
)
from data_processing.utils.db_utils import (
    get_previous_output,
    get_stage_parameters,
    load_database,
    parse_config,
    upsert_stage,
    upsert_tracklets,
)
from data_processing.utils.md_output_parser import MDVideoOutputParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def write_video(
    input_video: str,
    tublet: Dict[int, List[int]],
    target_size: Tuple[int, int],
    target_video_path: str,
    fps: float,
) -> None:
    """
    Crops every frame of input_video covered by the tublet and writes the crops
    to target_video_path with a given FPS

    Args:
        input_video: Input video file path
        tublet: Dictionary of crop boxes {frame_id: (x1, y1, x2, y2)} in pixels
        target_size: Target size (width, height) for output video frames
        target_video_path: Output video path
        fps: Frames per second for output video
    """
    if not os.path.exists(os.path.dirname(target_video_path)):
        os.makedirs(os.path.dirname(target_video_path))

    start_frame = min(list(tublet.keys()))
    end_frame = max(list(tublet.keys()))
    frame_generator = sv.get_video_frames_generator(
        source_path=str(input_video), start=start_frame, end=end_frame + 1
    )
    tublet_width, tublet_height = target_size

    video_info = sv.VideoInfo(
        width=tublet_width,
        height=tublet_height,
        fps=fps,
        total_frames=len(tublet.keys()),
    )

    with sv.VideoSink(
        target_path=str(target_video_path), video_info=video_info
    ) as sink:
        for idx, frame in enumerate(frame_generator):
            frame_id = idx + start_frame
            bbox_img = tublet[frame_id]

            current_w = bbox_img[2] - bbox_img[0]
            current_h = bbox_img[3] - bbox_img[1]
            if (current_h < tublet_height) or (current_w < tublet_width):
                interpolation = cv2.INTER_CUBIC
            else:
                interpolation = cv2.INTER_AREA

            crop_dims_up = [c + 4 for c in target_size]
            crop_up = frame[bbox_img[1] : bbox_img[3], bbox_img[0] : bbox_img[2], :]
            crop_up = cv2.resize(crop_up, crop_dims_up, interpolation=interpolation)

            bbox_img_down = [2, 2, crop_up.shape[1] - 2, crop_up.shape[0] - 2]
            crop = crop_up[
                bbox_img_down[1] : bbox_img_down[3],
                bbox_img_down[0] : bbox_img_down[2],
                :,
            ]

            sink.write_frame(frame=crop.astype("uint8"))


def form_tublet(
    bboxes: Dict[int, np.ndarray],
    target_size: Tuple[int, int],
    img_size: Tuple[int, int],
    padding: bool,
    boundary_frac: float,
) -> Dict[int, np.ndarray]:
    """
    Form tublet from bounding box coordinates

    Args:
        bboxes: Dictionary of bounding box coordinates {frame_id: (xmin, ymin, width, height)}
        target_size: Target size (tublet_width, tublet_height) for tublet frames
        img_size: Original video size (width, height)
        padding: Whether to pad with background
        boundary_frac: Fraction of the box to add as extra padding

    Returns:
        Dictionary of {frame_id: (x1, y1, x2, y2)} for the tublet
    """

    tublet = {}
    for frame_id, bbox_img in bboxes.items():
        bbox_img = normalize_bbox(bbox_img, img_size)

        bbox_img_w = bbox_img[2] - bbox_img[0]
        bbox_img_h = bbox_img[3] - bbox_img[1]

        bbox_aug_w = bbox_img_w + bbox_img_w * boundary_frac
        bbox_aug_h = bbox_img_h + bbox_img_h * boundary_frac
        bbox_img = adjust_crop_dimensions(bbox_img, (bbox_aug_w, bbox_aug_h), img_size)

        if padding:
            bbox_img = dim_padding(bbox_img, target_size, img_size)

        tublet[frame_id] = bbox_img

    return tublet


def process_video(
    file_id: str,
    input_video_path: str,
    video_annotation_df: pd.DataFrame,
    output_video_folder: Path,
    output_folder: Path,
    video_fps: float,
    frame_size: Tuple[int, int],
    tublet_height: int,
    tublet_width: int,
    background_padding: bool,
    add_boundary_fraction: float,
    min_tublet_duration_s: float,
    skip_existing: bool,
    raw_video_date_time: Optional[str] = None,
    trimmed_start_frame: int = 0,
) -> pd.DataFrame:
    """
    Process a video to generate a tublet video for each track based on annotations

    Args:
        file_id: Video file id
        input_video_path: Path of the video file to process
        video_annotation_df: DataFrame containing annotations for the video
        output_video_folder: Output folder path for tublet videos of this video
        output_folder: Root output folder of all tublet videos
        video_fps: FPS for the video
        frame_size: video width and height
        tublet_height: Desired height of the tublet
        tublet_width: Desired width of the tublet
        background_padding: Whether to pad with background
        add_boundary_fraction: Fraction of the box to add as extra padding
        min_tublet_duration_s: Minimum duration of a tublet in seconds
        skip_existing: Whether to skip processing if the output video file already exists
        raw_video_date_time: Original datetime of the raw video
        trimmed_start_frame: Start frame of the trimmed video relative to the raw video

    Returns:
        video_df: Dataframe with all relevant information about tublets generated from video_path.
    """
    track_ids = video_annotation_df["track_id"].unique()
    track_ids = pd.Series(track_ids).dropna().unique()
    frame_w, frame_h = frame_size

    tublets_metadata = []

    for track_id in track_ids:
        track_id_df = video_annotation_df[
            video_annotation_df["track_id"] == track_id
        ].reset_index(drop=True)

        # Trimmed video frame ids for the given track
        track_frames = track_id_df["frame_id"].sort_values().to_numpy()

        # Whenever a track was annotated as "outside", we make a new segment (tracklet)
        segment_end_idx = track_id_df[track_id_df["outside"] == 1].index.tolist()
        segment_end_idx = [*segment_end_idx, len(track_frames)]
        start_idx = 0
        segment_count = 1

        # Process each segment
        for segment_end_id in segment_end_idx:
            output_segment_name = (
                f"{file_id}_ID_{str(int(track_id))}_T{str(int(segment_count))}.mp4"
            )
            output_segment_path = Path(output_video_folder) / output_segment_name
            tublet_file_id = Path(output_segment_name).stem

            # Check that the segment is sufficiently long
            segment_frames = track_frames[start_idx:segment_end_id]
            if len(segment_frames) < (video_fps * min_tublet_duration_s):
                start_idx = min(segment_end_id + 1, len(track_frames))
                continue

            segment_track_id_df = track_id_df[
                track_id_df["frame_id"].isin(segment_frames)
            ]
            segment_bboxes = segment_track_id_df.set_index("frame_id")["bbox"].to_dict()

            process_segment(
                segment_bboxes,
                video_fps,
                tublet_height,
                tublet_width,
                background_padding,
                add_boundary_fraction,
                skip_existing,
                str(output_segment_path),
                input_video_path,
                frame_w,
                frame_h,
            )

            # Calculate tublet datetime if raw video datetime is available
            tublet_date_time = None
            if raw_video_date_time is not None:
                try:
                    # Tublet start (in s) w.r.t raw video start
                    tublet_frame_offset = np.round(
                        (trimmed_start_frame + track_frames[start_idx]) / video_fps, 3
                    )
                    tublet_date_time = pd.to_datetime(
                        raw_video_date_time
                    ) + pd.to_timedelta(tublet_frame_offset, unit="s")
                except (ValueError, AttributeError) as e:
                    # If datetime calculation fails, leave as None
                    logger.warning(
                        f"Could not calculate tublet datetime for {tublet_file_id}"
                    )
            else:
                logger.warning(
                    f"Could not calculate tublet datetime for {tublet_file_id}"
                )

            segment_info_dict = {
                "file_id": file_id,
                "track_id": track_id,
                "segment_id": segment_count,
                "tublet_file_id": tublet_file_id,
                "tublet_duration": len(segment_frames) / video_fps,
                "tublet_resolution": str(tublet_width) + "x" + str(tublet_height),
                "tublet_fps": video_fps,
                "tublet_start_frame": track_frames[start_idx],
                "tublet_end_frame": track_frames[segment_end_id - 1],
                "tublet_file_path": str(
                    output_segment_path.relative_to(output_folder)
                ),
                "tublet_date_time": str(tublet_date_time),
            }

            tublets_metadata.append(segment_info_dict)
            start_idx = min(segment_end_id + 1, len(track_frames))
            segment_count += 1

    video_df = pd.DataFrame(tublets_metadata)

    return video_df


def process_segment(
    segment_bboxes: Dict,
    fps: float,
    tublet_height: int,
    tublet_width: int,
    background_padding: bool,
    add_boundary_fraction: float,
    skip_existing: bool,
    output_video_path: str,
    input_video_path: str,
    frame_w: int,
    frame_h: int,
) -> None:
    """
    Writes the tublet video of one track segment.

    Args:
        segment_bboxes: Dictionary of bounding boxes {frame_id: (xmin, ymin, width, height)} in relative coordinates
        fps: Frames per second of the video
        tublet_height: Desired height of the tublet
        tublet_width: Desired width of the tublet
        background_padding: Whether to pad with background
        add_boundary_fraction: Fraction of the box to add as extra padding
        skip_existing: Whether to skip processing if the output video file already exists
        output_video_path: Output path of the tublet video
        input_video_path: Path of the video file to crop
        frame_w: Width of the video frames
        frame_h: Height of the video frames
    """
    tublet = form_tublet(
        bboxes=segment_bboxes,
        target_size=(tublet_width, tublet_height),
        img_size=(frame_w, frame_h),
        padding=background_padding,
        boundary_frac=add_boundary_fraction,
    )

    if not (Path(output_video_path).exists() and skip_existing):
        write_video(
            input_video=input_video_path,
            tublet=tublet,
            target_size=(tublet_width, tublet_height),
            target_video_path=str(output_video_path),
            fps=fps,
        )


def process_video_folder(
    trimmed_video_folder: str,
    input_json_folder: str,
    output_folder: str,
    videos_info_df: pd.DataFrame,
    video_data_df: pd.DataFrame,
    tublet_height: int,
    tublet_width: int,
    background_padding: bool,
    add_boundary_fraction: float,
    min_tublet_duration_s: float,
    skip_existing: bool,
) -> pd.DataFrame:
    """
    Generate the tublets of all videos with tracking results in input_json_folder

    Args:
        trimmed_video_folder: Folder path containing trimmed videos
        input_json_folder: Folder containing the tracking JSONs of the trimmed videos
        output_folder: Output folder path for tublet videos
        videos_info_df: Dataframe containing videos metadata
        video_data_df: Dataframe with the VideoData rows, including trimmed frame info
        tublet_height: Desired height of the tublet
        tublet_width: Desired width of the tublet
        background_padding: Whether to pad with background
        add_boundary_fraction: Fraction of the box to add as extra padding
        min_tublet_duration_s: Minimum duration of a tublet in seconds
        skip_existing: Whether to skip processing if the output video file already exists

    Returns:
        tublets_df: Dataframe with the metadata of all generated tublets
    """
    # Load track annotations
    logger.info("Loading track annotations")
    parser = MDVideoOutputParser()
    parser.parse_folder(input_json_folder)
    annotations_df = parser.to_dataframe()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df_list = []

    for file_id in tqdm(annotations_df["file_id"].unique(), position=0, leave=True):

        # Get corresponding annotations
        video_info_df = videos_info_df[videos_info_df["file_id"] == file_id]
        video_annotations_df = annotations_df[annotations_df["file_id"] == file_id]
        video_row_df = video_data_df[video_data_df["file_id"] == file_id]

        if len(video_info_df) == 0:
            logger.warning(
                f"Could not find associated metadata for video {file_id}, skipping..."
            )
            continue

        if len(video_annotations_df) == 0:
            logger.warning(
                f"Could not find associated json annotation for video {file_id}, skipping..."
            )
            continue

        if len(video_row_df) == 0:
            logger.warning(f"Video {file_id} not found in VideoData, skipping...")
            continue

        video_path = Path(trimmed_video_folder) / video_row_df[
            "trimmed_video_file_path"
        ].values[0]
        video_fps = video_info_df["fps"].values[0]
        video_width, video_height = tuple(
            [int(r) for r in str(video_info_df["resolution"].values[0]).split("x")]
        )
        # Get raw video datetime and trimmed start frame if available
        raw_video_date_time = video_info_df["date_time"].values[0]
        trimmed_start_frame = video_row_df["trimmed_start_frame"].values[0]

        # Tublets are saved in the same subfolder as their trimmed video
        output_video_folder = Path(output_folder) / video_path.relative_to(
            trimmed_video_folder
        ).parent

        # Process video
        video_df = process_video(
            file_id=file_id,
            input_video_path=str(video_path),
            video_annotation_df=video_annotations_df,
            output_video_folder=output_video_folder,
            output_folder=output_folder,
            video_fps=video_fps,
            frame_size=(video_width, video_height),
            tublet_height=tublet_height,
            tublet_width=tublet_width,
            background_padding=background_padding,
            add_boundary_fraction=add_boundary_fraction,
            min_tublet_duration_s=min_tublet_duration_s,
            skip_existing=skip_existing,
            raw_video_date_time=raw_video_date_time,
            trimmed_start_frame=trimmed_start_frame,
        )
        df_list.append(video_df)

    tublets_df = pd.concat(df_list, ignore_index=True)

    return tublets_df


def main(opt):
    # Load config
    config = parse_config(opt.config_file, logger=logger)
    params = get_stage_parameters(config, Path(__file__).stem)
    project_path = config["global"]["project_path"]
    db_file = config["global"]["database_path"]

    # Script parameters
    tublet_height = params["tublet_height"]
    tublet_width = params["tublet_width"]
    background_padding = params["background_padding"]
    add_boundary_fraction = params["add_boundary_fraction"]
    skip_existing = params["skip_existing"]
    min_tublet_duration_s = params["min_tublet_duration_s"]
    output_video_folder = Path(project_path) / params["output_video_folder_name"]

    # Get files from video trimming
    conn = load_database(db_file)
    input_video_folder = get_previous_output(
        conn=conn,
        stage_name="trimmed_from_detection",
        output_type="video",
    )
    input_json_folder = get_previous_output(
        conn=conn,
        stage_name="process_tracking_corrections_from_cvat",
        output_type="json",
    )
    videos_info_df = pd.read_sql("SELECT * FROM Videos", con=conn)
    video_data_df = pd.read_sql("SELECT * FROM VideoData", con=conn)

    # Process videos to generate tublets (will only process non-false-positive videos in VideoData)
    tublets_df = process_video_folder(
        input_video_folder,
        input_json_folder,
        output_video_folder,
        videos_info_df,
        video_data_df,
        tublet_height,
        tublet_width,
        background_padding,
        add_boundary_fraction,
        min_tublet_duration_s,
        skip_existing,
    )

    # Insert data into Tracklets table in the database
    logger.info(f"Inserting {len(tublets_df)} tracklets into database")

    # We insert tracklets in the database, and update existing ones if they already exist
    upsert_tracklets(conn=conn, output_df=tublets_df)

    # Update stage in database
    upsert_stage(
        conn=conn,
        stage_name=Path(__file__).stem,
        parameters_dict=params,
        output_video_folder=str(output_video_folder),
    )

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate tublets from the tracking results of the trimmed videos."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )

    opt = parser.parse_args()

    main(opt)
