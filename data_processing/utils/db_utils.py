import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_VIDEO_EXTENSIONS = [".mp4"]


def get_config_path(config_path: str) -> Path:
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    return Path(config_path)


def parse_config(config_file: str, logger) -> Dict:

    config_path = get_config_path(config_file)

    # load yaml
    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {config_path}: {e}")

    # Check has global configuration section
    global_config = config.get("global", None)
    if not global_config:
        raise ValueError("Global config section is required in the configuration file.")
    global_config["config_path"] = str(config_path)

    # Default paths
    project_path = Path(global_config["project_path"])
    input_videos_folder = Path(global_config["raw_videos_folder"])
    parsed_videos_folder = project_path / "parsed_videos"
    database_path = project_path / "database.db"

    # Check project path is valid and contains a video folder
    if project_path.exists() == False:
        raise NotADirectoryError(f"{project_path} folder is not a valid path.")
    if input_videos_folder.exists() == False:
        raise NotADirectoryError(
            f"{input_videos_folder} raw videos folder is not a valid path."
        )

    global_config["raw_path"] = str(input_videos_folder)
    global_config["parsed_videos"] = str(parsed_videos_folder)

    # Check if a database already exists, creates a new one if not
    if database_path.exists() == False:
        logger.info(
            f"Database file {database_path} not found, creating a new database."
        )
        build_database(str(database_path))
    else:
        logger.info(f"Found existing database at {database_path}")
    global_config["database_path"] = database_path

    return config


def get_stage_parameters(config: Dict, script_name: str) -> Dict:

    # if script_name does not exist in the config, raise an error
    if script_name not in config:
        raise ValueError(f"Script name '{script_name}' not found in the configuration.")

    stage_params = config[script_name]
    return stage_params


def get_parameters_from_config(config: Dict, name):
    """
    Get parameters from the configuration file
    """
    if name not in config:
        raise ValueError(f"Key '{name}' not found in the configuration file.")
    return config[name]


def list_video_files(
    folder, video_extensions: Optional[List[str]] = None
) -> List[Path]:
    if video_extensions is None:
        video_extensions = DEFAULT_VIDEO_EXTENSIONS
    valid_extensions = {ext.lower() for ext in video_extensions}
    return [
        f
        for f in Path(folder).rglob("*")
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]


def file_id_from_path(relative_path: Path) -> str:
    """
    Flattens a video path relative to the raw video folder into a file_id,
    e.g. "siteA/cam1/IMG_0001.MP4" -> "siteA_cam1_IMG_0001".
    """
    return Path(relative_path).with_suffix("").as_posix().replace("/", "_")


def output_relative_path(raw_file_path: str, extension: str) -> Path:
    """
    Relative path of a stage output for a raw video: the raw folder structure is
    preserved and the file is named by the video's file_id, e.g.
    ("siteA/cam1/IMG_0001.MP4", ".json") -> "siteA/cam1/siteA_cam1_IMG_0001.json".
    """
    return Path(raw_file_path).parent / f"{file_id_from_path(raw_file_path)}{extension}"


def load_database(db_path: Path) -> sqlite3.Connection:
    """
    Load a SQLite database file, and update the stored paths if the project folder has moved.

    Args:
        db_path (Path): Path of the database file, at the root of the project folder.

    Returns:
        sqlite3.Connection: Database connection object.
    """

    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = sqlite3.connect(db_path)
    current_project_path = str(db_path.parent)
    cursor = conn.cursor()
    cursor.execute("SELECT project_path FROM Project")
    project_row = cursor.fetchone()
    if project_row is None:
        return conn

    # The project folder has moved: replace the old prefix of every stored path with the new one
    stored_project_path = project_row[0]
    if Path(stored_project_path) != db_path.parent:
        cursor.execute("UPDATE Project SET project_path = ?", (current_project_path,))
        cursor.execute(
            """
            UPDATE Stages SET output_json_folder = REPLACE(output_json_folder, ?, ?),
                              output_video_folder = REPLACE(output_video_folder, ?, ?),
                              output_xml_folder = REPLACE(output_xml_folder, ?, ?),
                              examination_csv_path = REPLACE(examination_csv_path, ?, ?)
            """,
            (stored_project_path, current_project_path) * 4,
        )
        conn.commit()

    return conn


def build_database(db_file: str):

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    ## TABLES

    # Single row describing the project the database belongs to
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS Project (
            project_path    TEXT,
            config_path     TEXT,
            description     TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS Videos (
            file_id          TEXT PRIMARY KEY,
            group_id         TEXT,
            event_id         TEXT,
            raw_file_path    TEXT,
            date_time        TEXT,
            duration         REAL,
            fps              REAL,
            resolution       TEXT,
            is_IR            BOOLEAN
        );
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS Stages (
            stage_name TEXT PRIMARY KEY,
            parameters TEXT,
            output_json_folder TEXT,
            output_video_folder TEXT,
            output_xml_folder TEXT,
            examination_csv_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

    """
    )
    # Per-video processing results and manual review state
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS VideoData (
            file_id                       TEXT PRIMARY KEY,
            trimmed_duration              REAL,
            trimmed_start_frame           INTEGER,
            trimmed_end_frame             INTEGER,
            trimmed_video_file_path       TEXT,
            checked                       BOOLEAN DEFAULT 0,
            individuals                   INTEGER DEFAULT 0,
            false_positive_video          BOOLEAN DEFAULT 0,
            false_positive_track          BOOLEAN DEFAULT 0,
            tracking_short                BOOLEAN DEFAULT 0,
            segmented_track               BOOLEAN DEFAULT 0,
            id_switch                     BOOLEAN DEFAULT 0,
            bbox_wrong                    BOOLEAN DEFAULT 0,
            tracking_pbm                  BOOLEAN DEFAULT 0,
            tracking_and_localization_correct BOOLEAN DEFAULT 0,
            tracking_correct              BOOLEAN DEFAULT 0,
            description                   TEXT,
            created_at                    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (file_id) REFERENCES Videos(file_id)
        );


    """
    )
    # Create tracklet table to store tracklets
    cursor.execute(
        """
        CREATE TABLE "Tracklets" (
            "tublet_file_id"	TEXT PRIMARY KEY,
            "file_id"	TEXT,
            "track_id"	INTEGER,
            "segment_id"	INTEGER,
            "tublet_date_time"	TEXT,
            "tublet_duration"	REAL,
            "tublet_resolution"	TEXT,
            "tublet_fps"	REAL,
            "tublet_start_frame"	INTEGER,
            "tublet_end_frame"	INTEGER,
            "tublet_file_path"	TEXT,
            "created_at"	TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY("file_id") REFERENCES "Videos"("file_id")
        );
    """
    )
    conn.commit()
    cursor.close()
    conn.close()


def ensure_project(conn, global_config, logger):
    """
    Ensure that the project is registered in the database.
    If it is not, insert it.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM Project")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            """
            INSERT INTO Project (project_path, config_path, description)
            VALUES (?, ?, ?)
        """,
            (
                str(Path(global_config["project_path"])),
                str(global_config["config_path"]),
                global_config["description"],
            ),
        )
        logger.info("Registered project in the database.")
    else:
        logger.info("Project already registered in the database.")
    cursor.close()
    conn.commit()


def upsert_stage(
    conn,
    stage_name,
    parameters_dict,
    output_json_folder=None,
    output_video_folder=None,
    output_xml_folder=None,
    examination_csv_path=None,
):
    parameters_str = json.dumps(parameters_dict)

    def to_str_or_none(val):
        return str(val) if val is not None else None

    cursor = conn.cursor()
    # update also the created_at timestamp on conflict
    cursor.execute(
        """
        INSERT INTO Stages (
            stage_name, parameters, output_json_folder, output_video_folder, output_xml_folder, examination_csv_path
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(stage_name) DO UPDATE SET
            parameters = excluded.parameters,
            output_json_folder = excluded.output_json_folder,
            output_video_folder = excluded.output_video_folder,
            output_xml_folder = excluded.output_xml_folder,
            examination_csv_path = excluded.examination_csv_path,
            created_at = CURRENT_TIMESTAMP
        """,
        (
            stage_name,
            parameters_str,
            to_str_or_none(output_json_folder),
            to_str_or_none(output_video_folder),
            to_str_or_none(output_xml_folder),
            to_str_or_none(examination_csv_path),
        ),
    )
    cursor.close()
    conn.commit()


def upsert_video_data(conn, output_df: pd.DataFrame, logger):
    """
    Upsert trimmed video data into VideoData, preserving annotation columns on conflict.
    Removes stale entries (in DB but no longer in output).
    """
    existing_videos = pd.read_sql("SELECT file_id FROM VideoData", conn)[
        "file_id"
    ].unique()

    # Remove stale entries (in DB but not in current output)
    entries_to_remove = set(existing_videos) - set(output_df["file_id"].unique())
    if entries_to_remove:
        cursor = conn.cursor()
        cursor.executemany(
            "DELETE FROM VideoData WHERE file_id = ?",
            [(file_id,) for file_id in entries_to_remove],
        )
        affected = cursor.rowcount
        conn.commit()
        cursor.close()
        logger.info(f"Removed {affected} stale entries from VideoData table.")

    if output_df.empty:
        logger.info("No trimmed videos to insert into the database.")
        return

    # Upsert: insert new rows or overwrite trimmed video columns on conflict, preserving annotation columns
    new_file_ids = set(output_df["file_id"].unique())
    conflicts = set(existing_videos) & new_file_ids
    if conflicts:
        logger.warning(
            f"{len(conflicts)} existing entries will be overwritten with updated trimmed video data: {conflicts}"
        )
    cursor = conn.cursor()
    cursor.executemany(
        """
        INSERT INTO VideoData (file_id, trimmed_duration, trimmed_start_frame, trimmed_end_frame, trimmed_video_file_path)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(file_id) DO UPDATE SET
            trimmed_duration = excluded.trimmed_duration,
            trimmed_start_frame = excluded.trimmed_start_frame,
            trimmed_end_frame = excluded.trimmed_end_frame,
            trimmed_video_file_path = excluded.trimmed_video_file_path
        """,
        [
            (
                row["file_id"],
                row["trimmed_duration"],
                row["trimmed_start_frame"],
                row["trimmed_end_frame"],
                row["trimmed_video_file_path"],
            )
            for _, row in output_df.iterrows()
        ],
    )
    conn.commit()
    cursor.close()


def upsert_tracklets(conn, output_df: pd.DataFrame):
    """
    Insert tracklets from output_df in Tracklets Table, updating the tracklets already present in the Table
    """
    columns = output_df.columns
    column_names = ", ".join(columns)
    num_cols = len(columns)
    update_stmt = ", ".join([f"{col} = excluded.{col}" for col in columns])

    cursor = conn.cursor()
    cursor.executemany(
        f"""
        INSERT INTO Tracklets (
            {column_names}
        )
        VALUES ({", ".join(["?" for _ in range(num_cols)])})
        ON CONFLICT(tublet_file_id) DO UPDATE SET
            {update_stmt}
        """,
        (output_df.to_records(index=False).tolist()),
    )

    affected = cursor.rowcount
    cursor.close()
    conn.commit()
    logger.info(f"Upserted {affected} tracklets")


def get_previous_output(
    conn: sqlite3.Connection, stage_name: str, output_type: str
) -> Optional[str]:
    """
    Get the previous output folder or file for a given stage and type of output.
    Returns None if the stage was not run or has no output of this type.
    """
    output_columns = {
        "json": "output_json_folder",
        "video": "output_video_folder",
        "csv": "examination_csv_path",
    }
    if output_type not in output_columns:
        raise ValueError(
            f"Unknown output_type: {output_type}. Expected 'json', 'video', or 'csv'."
        )

    cursor = conn.cursor()
    cursor.execute(
        f"SELECT {output_columns[output_type]} FROM Stages WHERE stage_name = ?",
        (stage_name,),
    )
    result = cursor.fetchone()
    cursor.close()

    if result is None or result[0] is None:
        logger.warning(
            f"No output found for stage '{stage_name}' with type '{output_type}'."
        )
        return None
    return result[0]
