import argparse
import logging
from pathlib import Path

import pandas as pd

from data_processing.utils.db_utils import (
    get_previous_output,
    get_stage_parameters,
    load_database,
    parse_config,
    upsert_stage,
)
from data_processing.utils.video_detection_processor import VideoDetectionProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def process_videos_sparse_detection(opt):
    """
    Process videos with sparse detection using MegaDetector V6.
    This script initializes the VideoDetectionProcessor with specific parameters
    and runs the detection on the provided video folder.
    """
    # Get config parameters
    config = parse_config(opt.config_file, logger=logger)
    params = get_stage_parameters(config, Path(__file__).stem)
    project_path = config["global"]["project_path"]

    # get input_video_folder from db in raw_video_parser stage in Stages Table
    conn = load_database(config["global"]["database_path"])
    input_video_folder = get_previous_output(
        conn=conn,
        stage_name="raw_video_parser",
        output_type="video",
    )
    if input_video_folder is None:
        raise ValueError("No output found from raw_video_parser stage, run it first.")
    raw_videos_df = pd.read_sql_query("SELECT * FROM Videos", conn)
    raw_videos_df["video_path"] = raw_videos_df["raw_file_path"]

    # Initialize the processor with the desired configuration
    processor = VideoDetectionProcessor(
        params, input_video_folder, project_path, raw_videos_df
    )
    video_extensions = config.get("video_extensions")

    # Run sparse detection
    processor.process_videos_folder(
        detection_categories=params["detector_label_map"],
        video_extensions=video_extensions,
    )

    logger.info("Detection results saved to the database.")
    upsert_stage(
        conn=conn,
        stage_name=str(Path(__file__).stem),
        parameters_dict=params,
        output_json_folder=processor.output_json_folder,
        output_video_folder=None,
        examination_csv_path=None,
    )

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process videos with sparse detection using MegaDetector V6."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the configuration file for video processing.",
    )

    opt = parser.parse_args()
    process_videos_sparse_detection(opt)