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


def process_videos_dense_detection(opt):
    """
    Process videos with dense detection using MegaDetector V6.
    This script initializes the VideoDetectionProcessor with specific parameters
    and runs the detection on the provided video folder.
    """
    config = parse_config(opt.config_file, logger=logger)
    params = get_stage_parameters(config, Path(__file__).stem)
    params["stride"] = 1 # Enforce stride of 1 for dense detection, as frame id is retrieved from list index in output dict.
    project_path = config["global"]["project_path"]

    # get input_video_folder from db in raw_video_parser stage in Stages Table
    conn = load_database(config["global"]["database_path"])
    # Use the trimmed videos if available, otherwise the raw videos
    input_video_folder = get_previous_output(
        conn=conn,
        stage_name="trimmed_from_detection",
        output_type="video",
    )
    if input_video_folder is not None:
        logger.info(f"Using videos from trimmed_from_detection stage: {input_video_folder}")
        raw_videos_df = pd.read_sql_query(
            """SELECT Videos.*, VideoData.trimmed_video_file_path
            FROM Videos JOIN VideoData USING (file_id)""",
            conn,
        )
        raw_videos_df["video_path"] = raw_videos_df["trimmed_video_file_path"]
    else:
        input_video_folder = get_previous_output(
            conn=conn,
            stage_name="raw_video_parser",
            output_type="video",
        )
        if input_video_folder is None:
            raise ValueError(
                "No output found from either trimmed_from_detection or raw_parser stages"
            )
        logger.info(f"Using videos from raw_parser stage: {input_video_folder}")
        raw_videos_df = pd.read_sql_query("SELECT * FROM Videos", conn)
        raw_videos_df["video_path"] = raw_videos_df["raw_file_path"]

    processor = VideoDetectionProcessor(params, input_video_folder, project_path, raw_videos_df=raw_videos_df)
    video_extensions = config.get("video_extensions")
    processor.process_videos_folder(
        detection_categories=params["detector_label_map"],
        video_extensions=video_extensions,
    )

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
        description="Process videos with dense detection using MegaDetector V6."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the configuration file for video processing.",
    )

    opt = parser.parse_args()
    process_videos_dense_detection(opt)
