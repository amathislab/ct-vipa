"""
Merges the track corrections exported from CVAT with the tracking results. Videos
that were not corrected keep their tracking results, and videos marked as not
corrected in the correction CSV are left out.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data_processing.utils.db_utils import (
    get_previous_output,
    get_stage_parameters,
    load_database,
    parse_config,
    upsert_stage,
)
from data_processing.utils.megadetector_utils import from_video_xml_to_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def update_json_with_outside(input_json, output_json):
    with open(input_json, "r") as f_i:
        content = json.load(f_i)

    annotations = content["frames"]

    for f in annotations:
        for d in f["detections"]:
            d["outside"] = 0

    with open(output_json, "w") as f_o:
        json.dump(content, f_o, indent=1)


def main(opt):
    # Load config
    config = parse_config(opt.config_file, logger=logger)
    params = get_stage_parameters(config, Path(__file__).stem)
    project_path = config["global"]["project_path"]
    db_file = config["global"]["database_path"]

    # Load detections from database
    conn = load_database(db_file)
    input_xml_folder = Path(project_path) / params["input_xml_folder_name"]
    input_json_folder = get_previous_output(
        conn=conn,
        stage_name="process_videos_tracking",
        output_type="json",
    )
    cvat_corrections_csv = get_previous_output(
        conn=conn,
        stage_name="prepare_videos_for_cvat_track_correction",
        output_type="csv",
    )
    output_json_folder = Path(project_path) / params["output_json_folder_name"]

    # Get list of files to replace
    tracking_problem_df = pd.read_csv(cvat_corrections_csv)

    # Only keep files that have been corrected
    tracking_problem_df["corrected"] = tracking_problem_df["corrected"].astype(bool)
    not_corrected_ids = tracking_problem_df[~tracking_problem_df["corrected"]][
        "file_id"
    ]
    tracking_problem_df = tracking_problem_df[tracking_problem_df["corrected"]]

    metadata_df = pd.read_sql(
        "SELECT file_id, false_positive_video, trimmed_video_file_path FROM VideoData",
        conn,
    )
    animals_df = metadata_df[metadata_df["false_positive_video"] != 1]

    # We don't want to generate tracklets for tracks that have not been corrected
    animals_df = animals_df[~animals_df["file_id"].isin(not_corrected_ids.values)]
    logger.info("Found %d entries in VideoData.", len(animals_df))

    # Add outside property to all detections
    for _, row in animals_df.iterrows():
        rel_file_path = Path(row["trimmed_video_file_path"]).with_suffix(".json")
        input_json = Path(input_json_folder) / rel_file_path
        output_json = output_json_folder / rel_file_path
        os.makedirs(output_json.parent, exist_ok=True)
        update_json_with_outside(input_json, output_json)

    for _, row in tqdm(
        tracking_problem_df.iterrows(),
        total=tracking_problem_df.shape[0],
        desc="Processing CVAT corrections",
    ):
        rel_file_path = Path(row["trimmed_video_file_path"])
        json_file = Path(input_json_folder) / rel_file_path.with_suffix(".json")
        cvat_corrections_xml = Path(input_xml_folder) / rel_file_path.with_suffix(
            ".xml"
        )

        if not cvat_corrections_xml.exists():
            logger.warning(f"No CVAT correction file found for {cvat_corrections_xml}")
            continue
        else:

            # Save new json output
            rel_file_path = json_file.relative_to(Path(input_json_folder))
            output_file_path = output_json_folder / rel_file_path
            os.makedirs(output_file_path.parent, exist_ok=True)

            file_annotations = from_video_xml_to_json(cvat_corrections_xml)

            with open(json_file, "r") as f:
                new_annotations = json.load(f)

            mapping_video_track_global_track = {}
            track_id_counter = 1

            for frame_tracks in new_annotations["frames"]:
                frame_id = frame_tracks["frame_id"]
                new_annotations["frames"][frame_id]["detections"] = []
                new_annotations["frames"][frame_id]["max_detection_conf"] = 0

                if frame_id in file_annotations.keys():
                    for global_track_id in file_annotations[frame_id]:
                        if (
                            global_track_id
                            not in mapping_video_track_global_track.keys()
                        ):
                            mapping_video_track_global_track[global_track_id] = (
                                track_id_counter
                            )
                            track_id_counter += 1
                        video_track_id = mapping_video_track_global_track[
                            global_track_id
                        ]

                        new_annotations["frames"][frame_id]["max_detection_conf"] = 1.0
                        new_annotations["frames"][frame_id]["detections"].append(
                            {
                                "conf": 1.0,
                                "track_id": video_track_id,
                                "bbox": file_annotations[frame_id][global_track_id][
                                    "bbox"
                                ],
                                "category": 0,  # Should ideally be from lookup
                                "outside": file_annotations[frame_id][global_track_id][
                                    "outside"
                                ],
                            }
                        )
            with open(output_file_path, "w") as f:
                json.dump(new_annotations, f, indent=1, default=str)
    upsert_stage(
        conn=conn,
        stage_name=Path(__file__).stem,
        parameters_dict=params,
        output_json_folder=str(output_json_folder),
    )
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )

    opt = parser.parse_args()

    main(opt)
