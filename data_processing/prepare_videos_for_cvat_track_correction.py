"""
Convert Folder of annotations from MegaDetector to CVAT Video format.
Option to downscale the videos resolution by a given factor to reduce computational load on CVAT.
Outputs videos to a new folder if ratio < 1.0.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import cv2 as cv
import pandas as pd
from tqdm import tqdm

from data_processing.utils.db_utils import (
    get_previous_output,
    get_stage_parameters,
    load_database,
    parse_config,
    upsert_stage,
)
from data_processing.utils.megadetector_utils import from_json_to_video_xml

## Logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed output
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def resize_json_detections(input_json, resize_factor):

    input_json["info"]["width"] = int(input_json["info"]["width"] * resize_factor)
    input_json["info"]["height"] = int(input_json["info"]["height"] * resize_factor)

    return input_json


def resize_video(input_video_path, resize_factor, output_video_path):
    cap = cv.VideoCapture(str(input_video_path))
    fps = cap.get(cv.CAP_PROP_FPS)
    new_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) * resize_factor)
    new_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) * resize_factor)
    dim = (new_width, new_height)
    fourcc = cv.VideoWriter_fourcc(*"mp4v")  # type: ignore
    out = cv.VideoWriter(str(output_video_path), fourcc, fps, dim)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
        out.write(resized)

    cap.release()
    out.release()


def process_folder(opt):

    # Get parameters and setup
    config = parse_config(opt.config_file, logger=logger)
    params = get_stage_parameters(config, Path(__file__).stem)
    project_path = config["global"]["project_path"]
    output_xml_folder = Path(project_path) / params["output_xml_folder_name"]
    output_video_folder = Path(project_path) / params["output_video_folder_name"]
    output_csv_file = Path(project_path) / params["output_csv_filename"]

    conn = load_database(config["global"]["database_path"])

    input_csv = get_previous_output(
        conn, "prepare_videos_for_tracking_examination", "csv"
    )
    input_video_folder = get_previous_output(conn, "trimmed_from_detection", "video")
    input_json_folder = get_previous_output(conn, "process_videos_tracking", "json")

    annotations_df = pd.read_csv(input_csv)
    annotations_df["checked"] = annotations_df["checked"].astype(bool)

    not_checked = len(annotations_df) - len(annotations_df[annotations_df["checked"]])
    if not_checked > 0:
        logger.warning(
            "%d files to examine have not been reviewed (checked = False)", not_checked
        )

    # Get video filenames with a tracking problem to correct
    annotations_df["tracking_pbm"] = annotations_df.apply(
        lambda row: (
            0
            if row["false_positive_video"]
            else (
                1
                if row[
                    [
                        "false_positive_track",
                        "tracking_short",
                        "segmented_track",
                        "id_switch",
                    ]
                ].any()
                else 0
            )
        ),
        axis=1,
    )

    # Replace the whole VideoData table with the reviewed examination rows
    conn.execute("DELETE FROM VideoData")
    annotations_df.to_sql(
        "VideoData",
        conn,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )

    conn.commit()
    logger.info("VideoData table updated with %d rows.", len(annotations_df))

    # Export CSV with tracking problems that have been checked, adding corrected=False column
    tracking_correction_df = annotations_df[
        (annotations_df["tracking_pbm"] == 1) & (annotations_df["checked"] == True)
    ].copy()

    tracking_correction_df["corrected"] = False

    # Reorder columns to place new columns right after "checked"
    cols = list(tracking_correction_df.columns)
    checked_idx = cols.index("checked")
    cols.remove("corrected")
    cols.insert(checked_idx + 1, "corrected")
    tracking_correction_df = tracking_correction_df[cols]

    tracking_correction_df.to_csv(output_csv_file, index=False)
    logger.info(
        f"Exported {len(tracking_correction_df)} videos with tracking problems to {output_csv_file}"
    )

    # Continue with existing video processing
    video_rel_paths = tracking_correction_df["trimmed_video_file_path"].unique()
    video_rel_paths = [Path(f) for f in video_rel_paths]

    for video_rel_path in tqdm(video_rel_paths):
        video_filepath = Path(input_video_folder) / video_rel_path
        input_json_file = Path(input_json_folder) / video_rel_path.with_suffix(".json")
        output_xml = Path(output_xml_folder) / video_rel_path.with_suffix(".xml")
        output_video_filepath = Path(output_video_folder) / video_rel_path

        with open(input_json_file, "r") as f:
            input_json = json.load(f)

        if params.get("video_scale_factor", 1.0) < 1.0:
            assert (
                0.1 < params.get("video_scale_factor", 1.0) < 1.0
            ), "resizing factor must be between 0.1 and 1.0"
            input_json = resize_json_detections(
                input_json, params["video_scale_factor"]
            )
            if not (
                output_video_filepath.exists() and params.get("skip_existing", False)
            ):
                os.makedirs(output_video_filepath.parent, exist_ok=True)
                resize_video(
                    video_filepath, params["video_scale_factor"], output_video_filepath
                )

        if not (output_xml.exists() and params.get("skip_existing", False)):
            xml_out = from_json_to_video_xml(input_json)
            os.makedirs(output_xml.parent, exist_ok=True)
            with open(output_xml, "w") as f:
                f.write(xml_out)

    upsert_stage(
        conn=conn,
        stage_name=Path(__file__).stem,
        parameters_dict=params,
        output_video_folder=output_video_folder,
        output_xml_folder=output_xml_folder,
        examination_csv_path=output_csv_file,
    )
    conn.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the configuration file for video processing.",
    )

    opt = parser.parse_args()

    process_folder(opt)
