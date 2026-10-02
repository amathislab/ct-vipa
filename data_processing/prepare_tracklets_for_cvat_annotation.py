"""
Creates a default CVAT annotation XML for every tracklet video, an examination
CSV listing all tracklets, and the CVAT label constructor built from the configured label classes.
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
    list_video_files,
    load_database,
    parse_config,
    upsert_stage,
)
from data_processing.utils.label_utils import (
    build_cvat_label_constructor,
    get_label_classes,
)
from data_processing.utils.megadetector_utils import create_default_xml_tracklet

## Logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed output
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def process_folder(opt):

    # Get parameters and setup
    config = parse_config(opt.config_file, logger=logger)
    params = get_stage_parameters(config, Path(__file__).stem)
    project_path = config["global"]["project_path"]
    output_xml_folder = Path(project_path) / params["output_xml_folder_name"]
    output_csv_file = Path(project_path) / params["output_csv_filename"]
    output_label_constructor_file = (
        Path(project_path) / params["output_label_constructor_filename"]
    )

    conn = load_database(config["global"]["database_path"])

    input_tracklets_folder = get_previous_output(
        conn, "generate_tublets_from_tracking", "video"
    )

    tracklets_df = pd.read_sql("SELECT * FROM Tracklets", con=conn)

    tracklets_fp = list_video_files(
        input_tracklets_folder, config.get("video_extensions")
    )

    for tracklet_file in tqdm(tracklets_fp):
        tracklet_rel_path = tracklet_file.relative_to(input_tracklets_folder)

        tracklet_row = tracklets_df[
            tracklets_df["tublet_file_path"] == str(tracklet_rel_path)
        ]
        tracklet_resolution = tracklet_row["tublet_resolution"].values[0]
        tracklet_size = [int(e) for e in tracklet_resolution.split("x")]

        output_xml = Path(output_xml_folder) / tracklet_rel_path.with_suffix(".xml")

        if not (output_xml.exists() and params.get("skip_existing", False)):
            xml_out = create_default_xml_tracklet(tracklet_size)
            os.makedirs(output_xml.parent, exist_ok=True)
            with open(output_xml, "w") as f:
                f.write(xml_out)

    # Examination file
    examination_df = (
        tracklets_df[
            [
                "tublet_file_id",
                "file_id",
                "track_id",
                "segment_id",
                "tublet_date_time",
                "tublet_duration",
            ]
        ]
        .sort_values("tublet_file_id")
        .copy()
    )

    examination_df["checked"] = False
    examination_df["needs_review"] = False
    examination_df["tracking_problem"] = False

    examination_df.to_csv(output_csv_file, index=False)

    # CVAT label constructor
    label_constructor = build_cvat_label_constructor(get_label_classes(config))
    with open(output_label_constructor_file, "w") as f:
        json.dump(label_constructor, f, indent=2)
    logger.info(f"CVAT label constructor saved to {output_label_constructor_file}")

    upsert_stage(
        conn=conn,
        stage_name=Path(__file__).stem,
        parameters_dict=params,
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
