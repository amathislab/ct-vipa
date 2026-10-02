import argparse
import logging
import os
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import ffmpeg
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_processing.utils.db_utils import (
    ensure_project,
    file_id_from_path,
    get_stage_parameters,
    list_video_files,
    load_database,
    parse_config,
    upsert_stage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DAY_NIGHT_CLASSIFIER_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "day_night_clf_svm.pkl"
)
N_IR_SAMPLE_FRAMES = 4


def get_recording_time(video_path: Path) -> float:
    """
    Returns the recording start time of a video as a UNIX timestamp.
    Reads the container's creation_time tag (written by the camera, preserved
    when files are copied), falling back to the file modification time when
    the tag is missing or unreadable.
    Double check for timezone.
    """
    try:
        probe = ffmpeg.probe(str(video_path))
        tagged_sources = [probe.get("format", {})] + probe.get("streams", [])
        for source in tagged_sources:
            creation_time = source.get("tags", {}).get("creation_time")
            if creation_time:
                return pd.Timestamp(creation_time).timestamp()
    except (ffmpeg.Error, ValueError) as e:
        logger.warning(f"Could not read creation_time of {video_path}: {e}")
    logger.warning(f"No creation_time for {video_path}, using file modification time.")
    return os.path.getmtime(video_path)


class RawVideoParser:
    """
    Recursively scans a folder for video files (any layout, any naming),
    collects metadata for each one, and groups them into events.


    Parameters:
        event_time_period_s: time (in seconds) of inactivity required between two videos to constitute a novel event.
    """

    def __init__(self, event_time_period_s: int = 300) -> None:
        self.event_time_period_s = event_time_period_s
        self.results: Dict[str, Dict] = {}
        self.raw_df: Optional[pd.DataFrame] = None

    def parse_file(self, raw_video: Path, svm_clf) -> Dict:
        """
        Gets all relevant information from the raw_video file.
        Args:
            raw_video (Path): Path to the raw video file.
            svm_clf: Pre-trained SVM classifier for day/night classification.
        Returns:
            Dict: Dictionary containing video metadata.
        """

        date_time = get_recording_time(raw_video)

        vid = cv2.VideoCapture(raw_video.resolve().as_posix())
        if not vid.isOpened():
            logger.error(f"Error: Could not open video {raw_video}.")
            return {}

        # is_IR, from evenly spaced frames
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = np.linspace(
            0, total_frames, N_IR_SAMPLE_FRAMES, endpoint=False, dtype=int
        )
        frames = []
        for frame_idx in np.unique(sample_frames):
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = vid.read()
            if ret:
                # Downscale frame by factor 10
                frame = cv2.resize(frame, None, fx=0.1, fy=0.1)
                frames.append(frame)
        if not frames:
            logger.error(f"Error: Could not read frames of video {raw_video}.")
            vid.release()
            return {}
        img_array = np.array(frames)
        mean_colors = np.mean(img_array, axis=-1)
        rms_diff_mean = np.sum(
            np.sqrt((img_array - np.expand_dims(mean_colors, -1)) ** 2)
        ) / (img_array.shape[0] * img_array.shape[1] * img_array.shape[2])
        mean_R, mean_G, mean_B = (
            np.mean(img_array[:, :, :, 0]),
            np.mean(img_array[:, :, :, 1]),
            np.mean(img_array[:, :, :, 2]),
        )

        features = pd.DataFrame(
            [[rms_diff_mean, mean_R, mean_G, mean_B]],
            columns=["rms_diff_mean", "mean_R", "mean_G", "mean_B"],
        )
        is_IR = bool(1 - svm_clf.predict(features)[0])
        fps = vid.get(cv2.CAP_PROP_FPS)
        frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps != 0 else 0
        resolution = f"{width}x{height}"
        vid.release()
        return {
            "date_time": date_time,
            "duration": duration,
            "fps": fps,
            "resolution": resolution,
            "is_IR": is_IR,
        }

    def parse_folder(
        self,
        input_video_folder,
        video_extensions: Optional[List[str]] = None,
    ) -> Dict:
        """
        Recursively parses a folder for video files.

        Args:
            input_video_folder (Path): Path to the folder containing raw video files.
            video_extensions (List[str], optional): File extensions to treat as videos
                (case-insensitive). Defaults to DEFAULT_VIDEO_EXTENSIONS.
        Returns:
            Dict: Flat dictionary keyed by each video's path (relative to
                input_video_folder), containing parsed video metadata.
        """
        input_video_folder = Path(input_video_folder).resolve()
        assert (
            input_video_folder.is_dir()
        ), f"{input_video_folder} is not a directory or can't be read."

        self.results = {}
        raw_files = sorted(list_video_files(input_video_folder, video_extensions))
        if not raw_files:
            logger.warning(
                f"No video files found under {input_video_folder} "
                f"with extensions {video_extensions}"
            )

        logger.info("Detecting IR media")

        with open(DAY_NIGHT_CLASSIFIER_PATH, "rb") as f:
            svm_clf = pickle.load(f)

        for raw_file in tqdm(raw_files):
            relative_path = raw_file.relative_to(input_video_folder)
            key = relative_path.as_posix()

            meta = self.parse_file(raw_file, svm_clf)
            if not meta:
                logger.warning(f"Skipping unreadable video: {relative_path}")
                continue

            meta["raw_file_path"] = str(relative_path)
            meta["file_id"] = file_id_from_path(relative_path)
            # Videos in the same folder are assumed to come from the same camera
            parent = relative_path.parent
            meta["group_id"] = parent.as_posix() if str(parent) != "." else "root"

            self.results[key] = meta

        return self.results

    def get_event_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns an event_id to each video. Videos in the same group_id (i.e.
        the same containing folder) are split into separate events whenever
        the gap between one video ending and the next starting exceeds
        `event_time_period_s`.
        Args:
            df (pd.DataFrame): DataFrame containing video metadata, with
                "group_id", "date_time" and "end_time" columns.
        Returns:
            pd.DataFrame: DataFrame with an additional 'event_id' column.
        """
        df["event_id"] = None
        df_list = []
        event_id = 0

        for group in sorted(df["group_id"].unique()):
            event_id += 1  # New group --> must be a new event

            df_group = df[df["group_id"] == group].copy()
            df_group.sort_values(by="date_time", inplace=True)
            df_group.reset_index(drop=True, inplace=True)
            event_id_str = str(event_id)
            df_group.loc[0, "event_id"] = f"E{event_id_str.zfill(4)}"
            for index in range(1, len(df_group)):
                if (
                    df_group["date_time"][index] - df_group["end_time"][index - 1]
                    > self.event_time_period_s
                ):
                    event_id += 1
                event_id_str = str(event_id)
                df_group.loc[index, "event_id"] = f"E{event_id_str.zfill(4)}"

            df_list.append(df_group)

        final_df = pd.concat(df_list, ignore_index=True)
        return final_df

    def make_dataframe(self) -> pd.DataFrame:
        """
        Converts the flat results dictionary into a pandas DataFrame, and
        assigns event ids.
        Returns:
            pd.DataFrame: DataFrame containing parsed video metadata.
        """
        raw_df = pd.DataFrame(list(self.results.values()))
        if raw_df.empty:
            logger.warning("No videos were parsed; returning an empty DataFrame.")
            self.raw_df = raw_df
            return self.raw_df

        # Flattening can map two paths to the same file_id
        # (e.g. "a_b/c.mp4" and "a/b_c.mp4", or "c.mp4" and "c.avi")
        colliding_df = raw_df[raw_df["file_id"].duplicated(keep=False)]
        if not colliding_df.empty:
            raise ValueError(
                "Several videos map to the same file_id, rename them to make them unique:\n"
                + colliding_df.sort_values("file_id")[["file_id", "raw_file_path"]].to_string(index=False)
            )

        raw_df["end_time"] = raw_df["date_time"] + raw_df["duration"]
        raw_df = self.get_event_id(raw_df)

        raw_df["date_time"] = pd.to_datetime(
            raw_df["date_time"], unit="s", errors="coerce"
        )
        raw_df["end_time"] = pd.to_datetime(
            raw_df["end_time"], unit="s", errors="coerce"
        )

        cols = [
            "raw_file_path",
            "file_id",
            "group_id",
            "event_id",
            "duration",
            "fps",
            "resolution",
            "date_time",
            "is_IR",
        ]
        self.raw_df = (
            raw_df[cols]
            .dropna(subset=["raw_file_path"])
            .sort_values(["group_id", "date_time"])
            .reset_index(drop=True)
        )
        return self.raw_df

    def save_to_db(self, conn):
        """
        Saves the parsed video metadata to the database. Inserts new camera
        groups and videos, and updates existing videos with new metadata.
        Args:
            conn: SQLite database connection object.
        """
        if self.raw_df is None or self.raw_df.empty:
            logger.error("No raw data to save to database.")
            return
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()

        try:
            # Get existing videos
            existing_videos = pd.read_sql("SELECT file_id FROM Videos", conn)

            # Find new videos to insert
            new_videos = self.raw_df[
                ~self.raw_df["file_id"].isin(existing_videos["file_id"])
            ]
            video_cols = [
                "file_id",
                "group_id",
                "event_id",
                "raw_file_path",
                "date_time",
                "duration",
                "fps",
                "resolution",
                "is_IR",
            ]

            # Insert new videos
            if len(new_videos) > 0:
                new_videos[video_cols].to_sql(
                    "Videos", conn, if_exists="append", index=False
                )
                logger.info(f"Inserted {len(new_videos)} new videos in database.")

            # Update existing videos with new metadata
            existing_videos_to_update = self.raw_df[
                self.raw_df["file_id"].isin(existing_videos["file_id"])
            ]
            for _, row in existing_videos_to_update.iterrows():
                cursor.execute(
                    """UPDATE Videos
                    SET is_IR = ?, resolution = ?, fps = ?, duration = ?, event_id = ?, group_id = ?
                    WHERE file_id = ?""",
                    (
                        bool(row.get("is_IR", False)),
                        row["resolution"],
                        row["fps"],
                        row["duration"],
                        row["event_id"],
                        row["group_id"],
                        row["file_id"],
                    ),
                )

            logger.info(
                f"Updated {len(existing_videos_to_update)} existing videos with new data."
            )

        except sqlite3.IntegrityError as e:
            logger.error(f"Integrity error while inserting into database: {e}")
        except Exception:
            logger.exception("Error while inserting into database:")
        finally:
            conn.commit()
            cursor.close()


def process_folder(opt):
    # Load config file
    config = parse_config(opt.config_file, logger=logger)
    params = get_stage_parameters(config, Path(__file__).stem)
    input_dir = config["global"]["raw_path"]
    db_file = config["global"]["database_path"]
    video_extensions = config.get("video_extensions")

    # Parse videos
    parser = RawVideoParser(event_time_period_s=params["event_time_period_s"])
    logger.info("Parsing raw videos in %s", input_dir)
    _ = parser.parse_folder(input_dir, video_extensions=video_extensions)

    parser.make_dataframe()

    # Populate videos in database
    conn = load_database(db_file)
    parser.save_to_db(conn)

    # Register the project and the stage
    ensure_project(conn, config["global"], logger=logger)

    upsert_stage(
        conn,
        stage_name=Path(__file__).stem,
        parameters_dict=params,
        output_video_folder=str(input_dir),
    )  # if stage does not exist, add it or update it

    conn.close()
    logger.info("Parsing completed for %s", input_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="YAML config file",
    )

    opt = parser.parse_args()
    process_folder(opt)
