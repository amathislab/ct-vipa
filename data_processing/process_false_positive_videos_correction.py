"""
Puts back the videos wrongly flagged as false positives. Every event whose reviewed
video is marked 'checked' and 'true_positive' in the examination CSV has all its
videos copied from the sparse detections to the filtered detections, with the
false positive flags cleared.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from data_processing.utils.db_utils import (
    get_previous_output,
    load_database,
    output_relative_path,
    parse_config,
    upsert_stage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_true_positive_videos(
    review_csv_path: Path, conn, original_json_folder: Path
) -> pd.DataFrame:
    """Return file_id and detection_file_path of all videos from events reviewed as true positives."""
    review_df = pd.read_csv(review_csv_path, dtype={"event_id": str})
    for col in ["checked", "true_positive"]:
        if col not in review_df.columns:
            raise ValueError(f"CSV must include '{col}' column.")

    true_positive_events = review_df.loc[
        (review_df["checked"] == True) & (review_df["true_positive"] == True),
        "event_id",
    ].tolist()
    logger.info(
        "Found %d events reviewed as true positives.", len(true_positive_events)
    )

    event_videos_df = pd.read_sql(
        "SELECT file_id, raw_file_path FROM Videos WHERE event_id IN ({})".format(
            ",".join("?" for _ in true_positive_events)
        ),
        conn,
        params=true_positive_events,
    )
    event_videos_df["detection_file_path"] = event_videos_df["raw_file_path"].apply(
        lambda x: str(Path(original_json_folder) / output_relative_path(x, ".json"))
    )
    return event_videos_df[
        event_videos_df["detection_file_path"].map(lambda p: Path(p).exists())
    ]


def reintegrate_jsons(
    reviewed_df: pd.DataFrame, json_folder: Path, filtered_json_output: Path
) -> None:
    for detection_file_path in reviewed_df["detection_file_path"]:
        json_src = Path(detection_file_path)
        json_dst = Path(filtered_json_output) / json_src.relative_to(json_folder)
        json_dst.parent.mkdir(parents=True, exist_ok=True)

        data = json.loads(json_src.read_text())
        data.setdefault("info", {})["false_positive_video"] = False
        for frame in data.get("frames", []):
            for det in frame.get("detections", []):
                det["false_positive_detection"] = False
        json_dst.write_text(json.dumps(data, indent=2))

        logger.info("Reinserted JSON: %s", json_dst)


def process_reinsertion(opt: argparse.Namespace) -> None:
    config = parse_config(opt.config_file, logger=logger)

    conn = load_database(config["global"]["database_path"])
    original_json_folder = get_previous_output(
        conn, "process_videos_sparse_detection", "json"
    )
    filtered_json_folder = get_previous_output(
        conn, "flag_false_positive_videos", "json"
    )
    review_csv_path = get_previous_output(conn, "flag_false_positive_videos", "csv")

    if review_csv_path is not None and Path(review_csv_path).exists():
        reviewed_df = load_true_positive_videos(
            review_csv_path, conn, original_json_folder
        )
        logger.info("Reintegrating %d JSONs", len(reviewed_df))
        reintegrate_jsons(
            reviewed_df,
            json_folder=original_json_folder,
            filtered_json_output=filtered_json_folder,
        )
    else:
        logger.info("No examination CSV found, nothing to reintegrate.")

    upsert_stage(
        conn=conn,
        stage_name=Path(__file__).stem,
        parameters_dict={},
        output_json_folder=None,
        output_video_folder=None,
        examination_csv_path=None,
    )
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reintegrate the videos reviewed as true positives into the filtered detections."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the configuration file of the project.",
    )

    opt = parser.parse_args()
    process_reinsertion(opt)
