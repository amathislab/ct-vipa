# Database Documentation

## Why the database exists

The pipeline is a multi-step, multi-day process with frequent manual interventions. Without a central registry, every script would need to know the file paths produced by every previous script, either hard-coded, or passed as arguments, or inferred from folder conventions. This becomes fragile as stages are re-run and paths change.

We use a SQLite database (`database.db`, stored alongside the data at `project_path/database.db`) to store this information as the processing progresses. 

- **Input retrieval:** Each script queries the database to obtain the output from the previous step.
- **Output registration:** Each script records its own outputs in the database.
- **Progress tracking:** The database captures which stage has been run last, with which parameters, at what time, providing a trace of the project.
- **Manual review state:** the tracking examination writes its results back into the database, making the outcome of manual review part of the same queryable record as the automated processing results.

The database is deliberately lightweight (SQLite, no server, no dependencies beyond Python's standard library) so it can live next to the data on any storage medium and be queried with standard tools.

One limitation of using the database is that it requires running the workflow from Step 1. Users who have already pre-processed part of their data would need to populate the database first.

---

## Schema

### `Project`

Describes the project the database belongs to. It holds a single row.

| Column | Type | Description |
|---|---|---|
| `project_path` | TEXT | Path to the project folder, where the database and all stage outputs are written |
| `config_path` | TEXT | Path to the YAML config file used to create this project |
| `description` | TEXT | Human-readable description from `global.description` in the config |
| `created_at` | TIMESTAMP | When the project was first registered |

**Role in the pipeline:** created by `raw_video_parser.py` if it does not already exist. When the database is loaded, the stored `project_path` is compared with the current location of the database: if the project folder has moved, the stored stage paths are rewritten to the new location.

---

### `Videos`

The core registry for the raw video files. One row per raw video file.

| Column | Type | Description |
|---|---|---|
| `file_id` | TEXT (PK) | Unique video identifier: the path relative to the raw videos folder, without extension, with `/` replaced by `_` (e.g. `siteA/cam1/IMG_0001.MP4` → `siteA_cam1_IMG_0001`). Every output file of this video is named `{file_id}.{extension}`, in the same subfolder as the raw video (e.g. `detections/siteA/cam1/siteA_cam1_IMG_0001.json`) |
| `group_id` | TEXT | Parent folder path of this video, relative to the raw videos folder |
| `event_id` | TEXT | Event this video belongs to (automatically assigned) |
| `raw_file_path` | TEXT | Path relative to the raw videos folder |
| `date_time` | TEXT | Recording timestamp extracted from ffprobe |
| `duration` | REAL | Video duration in seconds |
| `fps` | REAL | Frames per second |
| `resolution` | TEXT | Frame dimensions as `{width}x{height}` |
| `is_IR` | BOOLEAN | Whether the video is infrared or not, as determined by the SVM classifier |

**Role in the pipeline:** populated by `raw_video_parser.py`. All downstream tables reference `file_id` as their foreign key to the video. Raw videos are never renamed or copied: `file_id` only needs to be unique, and the parser stops with an error listing the videos whose paths flatten to the same `file_id` (e.g. `a_b/c.mp4` and `a/b_c.mp4`). `event_id` groups related videos for ecological analysis. `is_IR` excludes infrared videos from the false positive flagging in Step 2b.

---

### `Stages`

Records the execution of each pipeline stage. One row per stage.

| Column | Type | Description |
|---|---|---|
| `stage_name` | TEXT (PK) | Name of the script (e.g., `process_videos_sparse_detection`) |
| `parameters` | TEXT | JSON-serialised parameters used for this run |
| `output_json_folder` | TEXT | Path to the folder where this stage wrote its JSON outputs (if any) |
| `output_video_folder` | TEXT | Path to the folder where this stage wrote its video outputs (if any) |
| `output_xml_folder` | TEXT | Path to the folder where this stage wrote its XML outputs (if any) |
| `examination_csv_path` | TEXT | Path to the examination CSV produced by this stage (if any) |
| `created_at` | TIMESTAMP | When this stage was last run |

**Primary key:** `stage_name`, re-running a stage overwrites the record rather than appending to it.

**Role in the pipeline:** Tracks the I/O of each stage. For example, `process_videos_tracking.py` calls `get_previous_output(conn, "process_videos_dense_detection", "json")` which queries `Stages` to get the `output_json_folder` of the dense detection stage. No script hard-codes a path to the previous step's folder, it always reads it from this table. This means re-running a stage with a different `output_json_folder_name` in the config automatically updates the pointer for all downstream stages.

The `parameters` column stores the full parameter dict as JSON, providing an implicit record of which settings were used at every stage run.

---

### `VideoData`

This is the table that stores the annotation progress. It stores per-video processing results. Populated incrementally as the pipeline progresses and updated during the manual tracking examination (Step 6).

| Column | Type | Description |
|---|---|---|
| `file_id` | TEXT (PK, FK → Videos) | The unique video identifier |
| `trimmed_duration` | REAL | Duration of the trimmed video in seconds (set by Step 3) |
| `trimmed_start_frame` | INTEGER | First frame of the trimmed clip, relative to the parsed video (set by Step 3) |
| `trimmed_end_frame` | INTEGER | Last frame of the trimmed clip (set by Step 3) |
| `trimmed_video_file_path` | TEXT | Relative path to the trimmed MP4 file |
| `checked` | BOOLEAN | Manually set during tracking examination: this video has been reviewed |
| `individuals` | INTEGER | Number of individual animals observed (filled during manual review) |
| `false_positive_video` | INTEGER | There is no animal in the video | 
| `false_positive_track` | INTEGER | Some of the tracks are false positives (not all of them) |
| `tracking_short` | INTEGER | The tracker missed significant portions of the animal track |
| `segmented_track` | INTEGER | The tracker assigned multiple track IDs to the same animal across a video |
| `id_switch` | INTEGER | Two animals had their tracking IDs swapped |
| `bbox_wrong` | INTEGER | The bounding box localization is inaccurate (e.g., missing body part) |
| `tracking_pbm` | INTEGER | Any non-localisation tracking problem (derived in Step 7 from `false_positive_track`, `tracking_short`, `segmented_track` and `id_switch`) |
| `tracking_and_localization_correct` | INTEGER | Correct tracking, no correction needed (not computed yet) |
| `tracking_correct` | INTEGER | Correct tracking, but box localisations may be inaccurate, no correction needed (not computed yet) |
| `description` | TEXT | Free-text notes |
| `created_at` | TIMESTAMP | Row creation time |

With `skip_trimming: true`, the `trimmed_*` columns describe the full video: it starts at frame 0, ends at its last frame, and its duration is the raw video duration.

**Primary key:** `file_id`, one row per video.

**Role in the pipeline:** this table bridges the automated processing and the human review. Step 3 (trimming) writes the `trimmed_*` columns. Steps 6–7 write and read the examination columns. Step 7 (`prepare_videos_for_cvat_track_correction.py`) reads this table to decide which videos need track correction: it selects rows where `checked=True`, `false_positive_video=False`, and at least one tracking flag is set. Each time Step 7 runs, the rows of the table are replaced by the rows of the examination CSV, so reviewer edits made to the CSV are persisted.

---

### `Tracklets`

One row per tracklet (per-animal video crop) generated in Step 9.

| Column | Type | Description |
|---|---|---|
| `tublet_file_id` | TEXT (PK) | Unique tracklet identifier `{file_id}_ID_{track_id}_T{segment_id}` (used as filename stem) |
| `file_id` | TEXT (FK → Videos) | Source trimmed video this tracklet was extracted from |
| `track_id` | INTEGER | Track ID within the source video that this tracklet corresponds to |
| `segment_id` | INTEGER | Segment index within the track (a track may be split into multiple segments if it has a gap) |
| `tublet_date_time` | TEXT | Timestamp of the tracklet's first frame |
| `tublet_duration` | REAL | Tracklet duration in seconds |
| `tublet_resolution` | TEXT | Crop resolution as `{width}x{height}` |
| `tublet_fps` | REAL | Frames per second |
| `tublet_start_frame` | INTEGER | First frame of the tracklet relative to the trimmed video |
| `tublet_end_frame` | INTEGER | Last frame of the tracklet relative to the trimmed video |
| `tublet_file_path` | TEXT | Relative path to the tracklet MP4 file |
| `created_at` | TIMESTAMP | Row creation time |

**Primary key:** `tublet_file_id`.

**Role in the pipeline:** Step 11 (`process_tracklet_annotations_from_cvat.py`) is the primary consumer. When it reads a CVAT annotation XML file, the filename stem is the `tublet_file_id`. It queries this table to find the corresponding `file_id`, `track_id`, and, crucially, `tublet_start_frame`. This offset is added to each annotated frame's local index to reconstruct the correct absolute frame position within the trimmed video JSON, so annotations are inserted at the right place.

---

## Relationship diagram

```
Project      (single row describing the project)

Stages       (one row per stage)

Videos       (one row per raw video)
    │
    ├── VideoData   (one row per video)
    │
    └── Tracklets   (one row per tracklet)
```

---

## Practical notes

- The database file lives at `project_path/database.db`. It is created automatically by the first script run on the project.
- It can be inspected interactively with any SQLite client (DB Browser for SQLite, the `sqlite3` CLI, or directly from the notebook via `sqlite3.connect(DB_PATH)`).
- Each project folder holds a single pipeline run. To compare different parameter sets (or different subsets of data), run the pipeline in separate project folders by changing `project_path` in the config.
- Because `Stages` uses `stage_name` as a primary key, re-running any stage overwrites its record. Outputs from the previous run remain on disk but the database pointer is updated. Use `skip_existing: true` in the config file to avoid re-processing files unnecessarily.
- The `Videos` table is populated per project: each project folder parses its raw videos into its own database.
