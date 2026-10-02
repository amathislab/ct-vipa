import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from tqdm import tqdm

from data_processing.utils.db_utils import (
    get_previous_output,
    get_stage_parameters,
    load_database,
    output_relative_path,
    parse_config,
    upsert_stage,
)
from data_processing.utils.label_utils import (
    NO_SECONDARY_ACTION,
    find_unknown_labels,
    get_label_classes,
)
from data_processing.utils.megadetector_utils import from_track_xml_to_json

## Logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed output
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def correct_detection(
    detection: Dict,
    always_second_action: List[str],
    default_first_action: str,
) -> Dict:
    """
    Correct action annotations in a detection dict according to predefined rules.

    Args:
        detection (Dict): Dictionary containing detection attributes
        always_second_action (list): Actions that should always be set as secondary behavior
        default_first_action (str): Default primary action to use when moving actions to secondary

    Returns:
        Dict: Detection dictionary with corrected action attributes
    """
    # Move specified actions to Action2 field, setting default_first_action in Action
    for action2 in always_second_action:
        if detection["attributes"]["Action"] == action2:
            detection["attributes"]["Action"] = default_first_action
            detection["attributes"]["Action2"] = action2

    # Remove duplicate actions
    if detection["attributes"]["Action"] == detection["attributes"]["Action2"]:
        detection["attributes"]["Action2"] = NO_SECONDARY_ACTION

    return detection


def process_folder(opt):

    # Get parameters and setup
    config = parse_config(opt.config_file, logger=logger)
    params = get_stage_parameters(config, Path(__file__).stem)
    project_path = config["global"]["project_path"]
    input_xml_folder = Path(project_path) / params["input_xml_folder_name"]
    output_json_folder = Path(project_path) / params["output_json_folder_name"]
    always_second_action = params["always_second_action"]
    default_first_action = params["default_first_action"]
    label_classes = get_label_classes(config)

    for action in always_second_action + [default_first_action]:
        if action not in label_classes["actions"]:
            logger.warning(
                f"Action '{action}' is not among the configured action classes"
            )

    conn = load_database(config["global"]["database_path"])
    input_json_folder = get_previous_output(
        conn, "process_tracking_corrections_from_cvat", "json"
    )

    # Read tublet metadata file
    logger.info("Reading Tracklets table from database")
    metadata_df = pd.read_sql(
        """SELECT Tracklets.*, Videos.raw_file_path
        FROM Tracklets JOIN Videos USING (file_id)""",
        con=conn,
    )

    # Read XML annotations
    annotation_files = list(Path(input_xml_folder).rglob("*.xml"))
    logger.info(f"Parsing CVAT task annotations for {len(annotation_files)} tracklets")
    annotations_dict = {}
    for annotation_file in tqdm(annotation_files):
        try:
            tublet_id = annotation_file.stem
            annotations_dict[tublet_id] = from_track_xml_to_json(annotation_file)
        except Exception as e:
            logger.error(f"Could not parse annotation file for {annotation_file}")
            continue

    # Keep track of previously saved trimmed video json files as we open/close them multiple times for each tublet
    already_saved_jsons = set()

    # Annotated values that are not among the configured label classes, per attribute
    unknown_labels: Dict[str, Set[str]] = {}

    # For each tublet we load the trimmed video json file and insert annotations in to the corresponding track segment
    logger.info(f"Inserting CVAT task annotations in their corresponding JSON files")
    for tublet_id in tqdm(annotations_dict.keys()):

        # Get corresponding tublet informations
        track_metadata_info = next(
            metadata_df[metadata_df["tublet_file_id"] == tublet_id].iterrows(), None
        )
        if track_metadata_info is None:
            logger.error(f"No entries found in database for {tublet_id}")
            continue
        _, track_metadata_row = track_metadata_info
        file_id = track_metadata_row["file_id"]
        track_id = track_metadata_row["track_id"]
        tublet_start_frame = track_metadata_row["tublet_start_frame"]
        segment_id = track_metadata_row["segment_id"]
        json_relative_path = output_relative_path(
            track_metadata_row["raw_file_path"], ".json"
        )

        # Get input json file containing tracking information for the corresponding trimmed video
        json_folder = (
            output_json_folder if file_id in already_saved_jsons else input_json_folder
        )
        input_json_file = Path(json_folder) / json_relative_path
        if not input_json_file.exists():
            logger.error(f"No input JSON file matching {file_id} found for {tublet_id}")
            continue

        with open(input_json_file, "r") as f:
            trimmed_video_annotations_dict = json.load(f)

        # Remove default annotations if any
        if file_id not in already_saved_jsons:
            for i, frame_annotation in enumerate(
                trimmed_video_annotations_dict["frames"]
            ):
                for detection in frame_annotation["detections"]:
                    detection["is_annotated"] = False
                    if "attributes" in detection:
                        del detection["attributes"]

        # Add CVAT annotations to the corresponding track segment in the annotations from trimmed video
        for frame_annotations in annotations_dict[tublet_id]:
            trimmed_video_frame_id = (
                frame_annotations["frame_id"] + tublet_start_frame
            )  # offset WRT trimmed video start
            trimmed_video_frame_dict = trimmed_video_annotations_dict["frames"][
                trimmed_video_frame_id
            ]
            for detection in trimmed_video_frame_dict["detections"]:
                if detection["track_id"] == int(
                    track_id
                ):  # There can be multiple individuals in this frame
                    detection["occluded"] = frame_annotations["occluded"]
                    detection["is_annotated"] = True
                    detection["segment_id"] = segment_id
                    detection["attributes"] = frame_annotations[
                        "attributes"
                    ]  # Replace with annotations from CVAT
                    detection = correct_detection(
                        detection,
                        always_second_action,
                        default_first_action,
                    )
                    for attribute, value in find_unknown_labels(
                        detection["attributes"], label_classes
                    ).items():
                        unknown_labels.setdefault(attribute, set()).add(value)

        # Copy annotations from previous detection for outside detections
        for i, frame_annotation in enumerate(trimmed_video_annotations_dict["frames"]):
            for detection in frame_annotation.get("detections", []):
                if "outside" in detection and detection["outside"]:
                    track_id = detection["track_id"]
                    if i == 0:
                        logger.warning(
                            f"Trying to copy previous attributes from an outside frame, but this is the first frame. {file_id}"
                        )
                        continue
                    for prev_detection in trimmed_video_annotations_dict["frames"][
                        i - 1
                    ]["detections"]:
                        if prev_detection["track_id"] == track_id:
                            detection["attributes"] = prev_detection.get(
                                "attributes", {}
                            )

        # Save json file
        output_json = Path(output_json_folder) / json_relative_path
        os.makedirs(output_json.parent, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(trimmed_video_annotations_dict, f, indent=2)
        already_saved_jsons.add(file_id)

    for attribute, values in unknown_labels.items():
        logger.warning(
            f"{attribute} values not among the configured label classes: {sorted(values)}"
        )

    upsert_stage(
        conn=conn,
        stage_name=Path(__file__).stem,
        parameters_dict=params,
        output_json_folder=output_json_folder,
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

    process_folder(opt)
