"""
Flags false positive videos in the sparse detections (see utils/false_positive_filter.py),
then splits them into videos kept for trimming and false positive videos to review.
Only the first false positive video of each event is exported for review, along with
an examination CSV to fill manually.
"""

import argparse
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from data_processing.utils.db_utils import (
    get_previous_output,
    get_stage_parameters,
    load_database,
    output_relative_path,
    parse_config,
    upsert_stage,
)
from data_processing.utils.false_positive_filter import flag_false_positive_jsons
from data_processing.utils.megadetector_utils import load_all_detections
from visualization.generate_results_visualization import process_video, reencode_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def copy_jsons(output_folder: Path, df_to_copy: pd.DataFrame) -> None:
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    for src, dst in zip(
        df_to_copy["detection_file_path"], df_to_copy["output_json_path"]
    ):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def render_review_video(raw_path: str, json_path: str, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    process_video(raw_path, json_path, output_path, "detection")
    reencode_video(output_path)


def render_review_videos(
    df_to_render: pd.DataFrame,
    n_cores: int,
    skip_existing: bool,
) -> None:
    """Render the raw videos with their flagged detections to their output_video_path."""
    output_paths = df_to_render["output_video_path"]
    if skip_existing:
        df_to_render = df_to_render[~output_paths.map(lambda p: Path(p).exists())]
        output_paths = output_paths[df_to_render.index]
    with ProcessPoolExecutor(max_workers=max(1, n_cores)) as executor:
        list(
            executor.map(
                render_review_video,
                df_to_render["raw_file_path"],
                df_to_render["detection_file_path"],
                output_paths,
            )
        )


def generate_review_csv(df_to_review: pd.DataFrame, review_csv_path: Path) -> None:
    """Write the examination CSV, with 'checked' and 'true_positive' to fill manually."""
    review_df = df_to_review[["file_id", "event_id"]].sort_values("file_id")
    review_df["checked"] = False
    review_df["true_positive"] = False
    review_df.to_csv(review_csv_path, index=False)
    logger.info("Generated review CSV with %d entries", len(review_df))


def process_flagging(opt: argparse.Namespace) -> None:
    config = parse_config(opt.config_file, logger=logger)
    params = get_stage_parameters(config, Path(__file__).stem)
    project_path = Path(config["global"]["project_path"])
    output_json_folder = project_path / params["output_json_folder_name"]
    output_video_folder = project_path / params["output_video_folder_name"]
    review_csv_path = project_path / params["output_csv_filename"]

    conn = load_database(config["global"]["database_path"])
    input_json_folder = get_previous_output(
        conn=conn,
        stage_name="process_videos_sparse_detection",
        output_type="json",
    )
    input_video_folder = get_previous_output(
        conn=conn,
        stage_name="raw_video_parser",
        output_type="video",
    )
    raw_videos_df = pd.read_sql(
        "SELECT file_id, group_id, event_id, raw_file_path, is_IR, date_time FROM Videos",
        conn,
    )

    flag_false_positive_jsons(
        input_json_folder,
        raw_videos_df,
        iou_threshold=params["iou_threshold"],
        min_events=params["min_events"],
        n_jobs=params["n_cores"],
    )

    videos_df = (
        load_all_detections(input_json_folder)[
            ["file_id", "false_positive_video", "detection_file_path"]
        ]
        .drop_duplicates("file_id")
        .merge(
            raw_videos_df[["file_id", "event_id", "raw_file_path", "date_time"]],
            on="file_id",
            how="left",
        )
    )
    videos_df["output_json_path"] = videos_df["raw_file_path"].apply(
        lambda x: str(output_json_folder / output_relative_path(x, ".json"))
    )
    videos_df["output_video_path"] = videos_df["raw_file_path"].apply(
        lambda x: str(output_video_folder / output_relative_path(x, ".mp4"))
    )
    videos_df["raw_file_path"] = videos_df["raw_file_path"].apply(
        lambda x: str(Path(input_video_folder) / x)
    )

    is_false_positive = (
        videos_df["false_positive_video"].astype("boolean").fillna(False)
    )
    kept_df = videos_df[~is_false_positive]
    false_positive_df = videos_df[is_false_positive]

    # Select first video per event
    videos_to_review = false_positive_df.sort_values(
        ["event_id", "date_time"], ascending=True
    ).drop_duplicates("event_id")
    logger.info(
        "%d videos kept, %d false positive videos, %d to review (first per event)",
        len(kept_df),
        len(false_positive_df),
        len(videos_to_review),
    )

    if videos_to_review.empty:
        logger.info(
            "No false positive videos to review, trimming will use the sparse detections."
        )
        output_json_folder = output_video_folder = review_csv_path = None
    else:
        logger.info("Copying kept JSONs to: %s", output_json_folder)
        copy_jsons(output_folder=output_json_folder, df_to_copy=kept_df)

        logger.info("Rendering review videos to: %s", output_video_folder)
        render_review_videos(
            videos_to_review,
            n_cores=params["n_cores"],
            skip_existing=params["skip_existing"],
        )
        generate_review_csv(videos_to_review, review_csv_path)
        logger.info("Review CSV saved to: %s", review_csv_path)

    upsert_stage(
        conn=conn,
        stage_name=Path(__file__).stem,
        parameters_dict=params,
        output_json_folder=output_json_folder,
        output_video_folder=output_video_folder,
        examination_csv_path=review_csv_path,
    )
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flag false positive videos in the sparse detections and export them for review."
    )
    parser.add_argument(
        "--config_file", type=str, required=True, help="Path to the configuration file."
    )

    opt = parser.parse_args()
    process_flagging(opt)
