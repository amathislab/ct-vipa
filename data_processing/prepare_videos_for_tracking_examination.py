"""
Renders the tracking results on the trimmed videos and exports the examination CSV
to fill manually.
"""

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
from visualization.generate_results_visualization import process_batch_video_folder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_excel_ready_csv(conn, output_csv_path: str) -> None:
    """
    Export the VideoData rows to a CSV for manual annotation.
    Rows of an existing CSV are preserved.

    Args:
        - conn: database connection
        - output_csv_path: output path for the csv file
    """
    df = pd.read_sql("SELECT * FROM VideoData", conn)
    # Output path
    df.sort_values(by=["file_id"], inplace=True)
    # if a csv already exists load it and merge with the new one to preserve existing rows
    if Path(output_csv_path).exists():
        existing_df = pd.read_csv(output_csv_path)
        df = pd.concat([existing_df, df]).drop_duplicates(subset=["file_id"])
        df.sort_values(by=["file_id"], inplace=True)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Excel-ready CSV saved to {output_csv_path}")


def generate_results_visualization(opt):
    config = parse_config(opt.config_file, logger=logger)
    params = get_stage_parameters(config, Path(__file__).stem)
    project_path = config["global"]["project_path"]
    output_video_folder = Path(project_path) / params["output_video_folder_name"]

    conn = load_database(config["global"]["database_path"])

    input_json_folder = get_previous_output(
        conn=conn,
        stage_name="process_videos_tracking",
        output_type="json",
    )

    # First try to get output from trimmed_from_detection stage
    input_video_folder = get_previous_output(
        conn=conn,
        stage_name="trimmed_from_detection",
        output_type="video",
    )
    if input_video_folder is None:
        # Fall back to raw_parser if trimmed_from_detection doesn't exist or has no output
        input_video_folder = get_previous_output(
            conn=conn,
            stage_name="raw_video_parser",
            output_type="video",
        )
        if input_video_folder is None:
            raise ValueError(
                "No output found from either trimmed_from_detection or raw_parser stages"
            )
        else:
            logger.info(f"Using videos from raw_parser stage: {input_video_folder}")
    else:
        logger.info(
            f"Using videos from trimmed_from_detection stage: {input_video_folder}"
        )

    process_batch_video_folder(
        input_video_folder=input_video_folder,
        input_json_folder=input_json_folder,
        output_video_folder=output_video_folder,
        num_workers=params["n_cores"],
        skip_existing=params["skip_existing"],
        visualization_type=params["visualization_type"],
        video_extensions=config.get("video_extensions"),
    )

    examination_csv_path = Path(project_path) / params["output_csv_filename"]
    generate_excel_ready_csv(conn, examination_csv_path)

    upsert_stage(
        conn,
        stage_name=Path(__file__).stem,
        parameters_dict=params,
        output_json_folder=None,
        output_video_folder=output_video_folder,
        examination_csv_path=examination_csv_path,
    )
    conn.commit()
    conn.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file", type=str, required=True, help="Path to the configuration file."
    )

    opt = parser.parse_args()
    generate_results_visualization(opt)


if __name__ == "__main__":
    main()
