"""
From MegaDetector output, discard false positive videos, videos with humans and videos without
a long enough animal segment. The kept videos are trimmed to their last animal detection, or
linked at full length into the output folder when skip_trimming is set.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import ffmpeg
import pandas as pd
from tqdm import tqdm

from data_processing.utils.db_utils import (
    get_previous_output,
    get_stage_parameters,
    load_database,
    output_relative_path,
    parse_config,
    upsert_stage,
    upsert_video_data,
)
from data_processing.utils.megadetector_utils import load_all_detections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def trim_video(
    input_path,
    start_frame,
    end_frame,
    output_path,
    write_to_disk,
    skip_existing,
    frame_padding,
    fps,
) -> float:
    """
    Trim the video stored at input_path from start_frame to end_frame and save to the output path.
    Returns the duration of the trimmed video.

    Args:
        input_path (str): Path to the input video.
        start_frame (int): Frame number to start trimming.
        end_frame (int): Frame number to end trimming.
        output_path (str): Path to save the trimmed video.
        write_to_disk (bool): Whether to write the output video to disk.
        skip_existing (bool): Whether to skip trimming if the output video already exists.
        frame_padding (int): Number of frames added before start_frame and after end_frame.
        fps (float): Frame rate of the input video.

    Returns:
        Duration of the trimmed video in seconds.
    """
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_frame = max(start_frame - frame_padding, 0)
    end_frame = end_frame + frame_padding

    start_time = start_frame / fps
    end_time = end_frame / fps
    trimmed_duration = (end_frame - start_frame) / fps
    skip = skip_existing and os.path.exists(output_path)
    if write_to_disk and not skip:
        try:
            stream = ffmpeg.input(str(input_path), ss=start_time)
            stream = ffmpeg.output(
                stream,
                str(output_path),
                **{
                    "t": trimmed_duration,
                    "c:v": "copy",
                    "c:a": "aac",
                },
                avoid_negative_ts="make_zero",
            )

            cmd = " ".join(stream.compile())
            logger.debug(f"FFmpeg command: {cmd}")

            out, err = stream.run(
                overwrite_output=True, capture_stdout=True, capture_stderr=True
            )

        except ffmpeg.Error as e:
            logger.error(f"FFmpeg stderr:\n{e.stderr.decode()}")
            raise
        except Exception as e:
            logger.error(f"Error trimming video: {str(e)}")
            raise
    else:
        logger.debug(
            f"Skipping video {input_path} as it already exists at {output_path} or write_to_disk is False."
        )
    return trimmed_duration


def link_video(
    input_path: Path,
    output_path: Path,
    write_to_disk: bool,
    skip_existing: bool,
) -> None:
    """
    Symlink the full video stored at input_path to the output path, used instead of trimming.

    Args:
        input_path (Path): Path to the input video.
        output_path (Path): Path of the symlink to create.
        write_to_disk (bool): Whether to create the symlink on disk.
        skip_existing (bool): Whether to skip linking if the output video already exists.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    skip = skip_existing and output_path.exists()
    if write_to_disk and not skip:
        output_path.unlink(missing_ok=True)
        os.symlink(Path(input_path).resolve(), output_path)
    else:
        logger.debug(
            f"Skipping video {input_path} as it already exists at {output_path} or write_to_disk is False."
        )


def get_animal_segments(
    video_detections_df, min_frames, min_detection_threshold
) -> Optional[Tuple[int, int]]:
    """
    Given the animal detections of a video, returns (start_frame, end_frame) corresponding to the
    trimmed segment with animal activity detected with a confidence threshold above min_detection_threshold
    and of a length greater than min_frames.

    Args:
        video_detections_df (pd.DataFrame): DataFrame containing video detection information.
        min_frames (int): Minimum number of frames required for a valid segment.
        min_detection_threshold (float): Minimum confidence threshold for detections.

    Returns:
        Tuple of start frame and end frame if a valid segment is found, else None.
    """
    start_frame = (
        0  # always start from the beginning of the video and only trim the end
    )

    video_detections_df_conf = video_detections_df[
        video_detections_df["conf"] >= min_detection_threshold
    ]
    end_frame = video_detections_df_conf["frame_id"].max()

    if end_frame - start_frame > min_frames:
        return start_frame, int(end_frame)
    else:
        return None


def contains_human(video_detections_df, human_conf_threshold, human_key) -> bool:
    """
    Check if human detections are present in video detections.

    Args:
        video_detections_df (pd.DataFrame): DataFrame containing video detection information.
        human_conf_threshold (float): Confidence threshold for human detections.
        human_key (str): Detector category of humans.

    Returns:
        bool: True if human detections are present, False otherwise.
    """
    human_detections = video_detections_df[
        (video_detections_df["category"] == human_key)
        & (video_detections_df["conf"] >= human_conf_threshold)
    ]

    return len(human_detections) > 0


def get_input_folder(conn, stage_name, output_type):
    input_json_folder = get_previous_output(
        conn=conn,
        stage_name="flag_false_positive_videos",
        output_type="json",
    )
    if input_json_folder is None:
        # Fall back to raw_parser if flag_false_positive_videos doesn't exist or has no output
        logger.info(
            f"Could not find flag_false_positive_videos output. Falling back to output from process_videos_sparse_detection."
        )
        input_json_folder = get_previous_output(
            conn=conn,
            stage_name="process_videos_sparse_detection",
            output_type="json",
        )
        if input_json_folder is None:
            raise ValueError(
                "No output found from either flag_false_positive_videos or process_videos_sparse_detection stages"
            )
        else:
            logger.info(
                f"Using videos from process_videos_sparse_detection stage: {input_json_folder}"
            )
    else:
        logger.info(
            f"Using videos from flag_false_positive_videos stage: {input_json_folder}"
        )
    return input_json_folder


def process_trimming(opt):
    """
    Main function to process and trim videos based on MegaDetector output.

    Args:
        opt: command line arguments
    """

    # Load config
    config = parse_config(opt.config_file, logger=logger)
    params = get_stage_parameters(config, Path(__file__).stem)
    project_path = config["global"]["project_path"]
    db_file = config["global"]["database_path"]

    # Load detections from database
    conn = load_database(db_file)
    # First try to get output from flag_false_positive_videos stage
    input_json_folder = get_input_folder(
        conn=conn,
        stage_name="flag_false_positive_videos",
        output_type="json",
    )

    md_outputs_df = load_all_detections(input_json_folder)

    # Remove false positive detections
    md_outputs_df["false_positive_video"] = (
        md_outputs_df["false_positive_video"].astype(bool).fillna(False)
    )
    md_outputs_df = md_outputs_df[~md_outputs_df["false_positive_video"]].reset_index(
        drop=True
    )

    # Detector categories used by the sparse detection
    cursor = conn.cursor()
    cursor.execute(
        "SELECT parameters FROM Stages WHERE stage_name = 'process_videos_sparse_detection'"
    )
    detector_params = cursor.fetchone()
    if detector_params is None or detector_params[0] is None:
        logger.error(
            "No parameters found for stage 'process_videos_sparse_detection' in the database."
        )
        conn.close()
        return
    detector_params = detector_params[0]
    detector_label_map = json.loads(detector_params).get("detector_label_map")
    # get fps from database from the Videos Table within the file_id column of md_outputs_df and parse_file_path
    md_outputs_df["fps"] = md_outputs_df["file_id"].map(
        lambda file_id: cursor.execute(
            "SELECT fps FROM Videos WHERE file_id = ?",
            (file_id,),
        ).fetchone()[0]
    )

    md_outputs_df["duration"] = md_outputs_df["file_id"].map(
        lambda file_id: cursor.execute(
            "SELECT duration FROM Videos WHERE file_id = ?",
            (file_id,),
        ).fetchone()[0]
    )

    md_outputs_df["raw_file_path"] = md_outputs_df["file_id"].map(
        lambda file_id: cursor.execute(
            "SELECT raw_file_path FROM Videos WHERE file_id = ?",
            (file_id,),
        ).fetchone()[0]
    )

    # Get unique video paths from the DataFrame
    video_rel_paths = list(md_outputs_df["raw_file_path"].unique())
    sample = params["debug_n_videos"]
    output_video_folder = Path(project_path) / params["output_video_folder_name"]
    input_video_folder = get_previous_output(
        conn=conn,
        stage_name="raw_video_parser",
        output_type="video",
    )

    if sample is not None:
        video_rel_paths = video_rel_paths[:sample]

    logger.info(
        "Processing %d videos with animal detections from %s",
        len(video_rel_paths),
        input_video_folder,
    )
    skip_trimming = params["skip_trimming"]
    if skip_trimming:
        logger.info("Trimming skipped: kept videos are linked at full length")
    human_key = [k for k, v in detector_label_map.items() if v == "person"][0]
    animal_key = [k for k, v in detector_label_map.items() if v == "animal"][0]
    for column in [
        "trimmed_video_file_path",
        "trimmed_duration",
        "trimmed_start_frame",
        "trimmed_end_frame",
    ]:
        md_outputs_df[column] = None
    # Trimming videos
    for video_rel_path in tqdm(
        video_rel_paths, position=0, leave=True, desc="Videos processed"
    ):
        video_detections_df = md_outputs_df[
            md_outputs_df["raw_file_path"] == video_rel_path
        ]
        file_id = video_detections_df["file_id"].values[0]

        if contains_human(
            video_detections_df, params["human_conf_threshold"], human_key
        ):
            logger.info(f"Skipping {file_id} as it contains humans")
            continue

        start_end_frames = get_animal_segments(
            video_detections_df[video_detections_df["category"] == animal_key],
            params["min_frames"],
            params["min_detection_threshold"],
        )

        if start_end_frames is not None:
            input_path = Path(input_video_folder) / video_rel_path

            output_path = Path(output_video_folder) / output_relative_path(
                video_rel_path, Path(video_rel_path).suffix
            )

            fps = video_detections_df.fps.values[0]
            if skip_trimming:
                link_video(
                    input_path,
                    output_path,
                    params["save_videos"],
                    params["skip_existing"],
                )
                trimmed_duration = video_detections_df.duration.values[0]
                start_frame = 0
                end_frame = int(round(trimmed_duration * fps))
            else:
                start_frame, end_frame = start_end_frames
                trimmed_duration = trim_video(
                    input_path,
                    start_frame,
                    end_frame,
                    output_path,
                    params["save_videos"],
                    params["skip_existing"],
                    params["frame_padding"],
                    fps,
                )
            md_outputs_df.loc[
                md_outputs_df["file_id"] == file_id, "trimmed_video_file_path"
            ] = str(output_path.relative_to(output_video_folder))
            md_outputs_df.loc[
                md_outputs_df["file_id"] == file_id, "trimmed_duration"
            ] = trimmed_duration
            md_outputs_df.loc[
                md_outputs_df["file_id"] == file_id, "trimmed_start_frame"
            ] = start_frame
            md_outputs_df.loc[
                md_outputs_df["file_id"] == file_id, "trimmed_end_frame"
            ] = end_frame

    output_df = (
        md_outputs_df.dropna(subset=["trimmed_video_file_path"])
        .drop_duplicates(subset=["file_id"])
        .reset_index(drop=True)
    )
    output_df = output_df[
        [
            "file_id",
            "trimmed_video_file_path",
            "trimmed_duration",
            "trimmed_start_frame",
            "trimmed_end_frame",
        ]
    ]

    upsert_video_data(conn, output_df, logger)

    upsert_stage(
        conn=conn,
        stage_name=str(Path(__file__).stem),
        parameters_dict=params,
        output_json_folder=None,
        output_video_folder=output_video_folder,
        examination_csv_path=None,
    )

    conn.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="YAML config file, e.g. configs/default.yaml",
    )

    opt = parser.parse_args()

    process_trimming(opt)
