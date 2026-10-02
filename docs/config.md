# Configuration File Reference

The pipeline is configured via a YAML file, passed to every script via `--config_file`. See `configs/default.yaml` for a working example.

---

## `global`

| Parameter | Example | Description |
|---|---|---|
| `description` | `"Demo run"` | Free-text description stored in the database for reference. |
| `project_path` | `"/path/to/project"` | Root folder where the database and all stage outputs are written. Use a different folder to run the pipeline with another parameter set. |
| `raw_videos_folder` | `"/path/to/raw_videos"` | Path to the original camera-trap videos to process and annotate. |


---

## `raw_video_parser`

| Parameter | Example | Description |
|---|---|---|
| `event_time_period_s` | `300` | Inactivity gap in seconds between two consecutive videos that would produce a new event ID. Lower values create more, shorter events; higher values merge distinct visits. |

---

## `process_videos_sparse_detection`

### Detector settings

| Parameter | Example | Description |
|---|---|---|
| `model` | `"MDV6-yolov10-e"` | MegaDetector model variant to use. |
| `stride` | `30` | Run the detector every `<stride>` frames. Higher values reduce computational cost but may miss fast-moving animals. |
| `json_confidence_threshold` | `0.2` | Minimum confidence to write a detection to the JSON. Intentionally permissive — harder thresholding happens in later steps. |
| `n_cores` | `2` | Number of parallel worker threads (frame extraction runs in parallel, inference on the GPU runs one video at a time). |
| `batch_size` | `8` | Frames per inference batch. Reduce if you encounter GPU out-of-memory errors. |
| `skip_existing` | `true` | Skip videos whose output JSON already exists. |
| `output_json_folder_name` | `"output_sparse_json"` | Folder name for the detection results. |
| `tmp_frame_folder_name` | `"tmp_detection_frames"` | Temporary folder for extracted frames during inference. |
| `force_frame_folder_deletion` | `true` | Delete the temporary frame folder once all videos are processed. The frames of each video are always deleted after its inference. |
| `detector_label_map` | `{"0": "animal", ...}` | Maps model class indices to label names. Only change if using a custom model with different class ordering than MegaDetector. |
| `debug_max_videos` | `5` | Maximum number of videos to process (debug) |

---

## `flag_false_positive_videos`

Flags false positive triggers in the sparse detections: daytime (non-IR) videos where every detection is an object of the scene, found at the same location in several events of the same camera. Then exports them for review.

| Parameter | Example | Description |
|---|---|---|
| `iou_threshold` | `0.8` | Minimum IoU for two bounding boxes of the same camera to be considered at the same location. |
| `min_events` | `2` | Minimum number of distinct events in which detections at the same location must appear to be flagged as false positive detections. A value of `1` flags every detection. |
| `n_cores` | `2` | Number of parallel worker processes for flagging and rendering. |
| `skip_existing` | `true` | Skip review videos already rendered. |
| `output_json_folder_name` | `"detections_filtered_json"` | Folder for detection JSONs of videos that are not false positives. |
| `output_video_folder_name` | `"false_positive_videos"` | Folder for review videos (first false positive video of each event), rendered with bounding-box overlays. |
| `output_csv_filename` | `"false_positive_examination.csv"` | Name of the CSV to be filled manually. |

## `process_false_positive_videos_correction`

No parameters. Corrections are driven entirely by the manually filled examination CSV.

---

## `trimmed_from_detection`

| Parameter | Example | Description |
|---|---|---|
| `skip_trimming` | `false` | Keep the full videos instead of trimming them. Videos are still filtered, and the kept ones are symlinked into `output_video_folder_name`. `frame_padding` is then ignored, and `save_videos` and `skip_existing` apply to the symlinks. |
| `min_detection_threshold` | `0.2` | Minimum animal confidence for a frame to be included in a detected segment. Separate from `json_confidence_threshold`, which controls what is written to the JSON. |
| `min_frames` | `15` | Discard videos whose last animal detection above `min_detection_threshold` is before this frame. Filters out marginal triggers unlikely to produce useful tracklets. |
| `human_conf_threshold` | `0.8` | Exclude videos where a person is detected above this confidence. |
| `frame_padding` | `15` | Frames to include after the last animal detection to avoid clipping the animal's exit. |
| `save_videos` | `true` | Write the trimmed videos (same container as the raw video). Set to `false` for a dry run that only updates the database. |
| `skip_existing` | `false` | Re-trim even if the output video already exists. |
| `debug_n_videos` | `null` | Process only this many videos. Useful for testing on a small subset. |
| `output_video_folder_name` | `"trimmed_videos"` | Folder under the project folder for the trimmed videos. |

---

## `process_videos_dense_detection`

Same structure as `process_videos_sparse_detection` on every frame (`stride = 1`). Only difference:

| Parameter | Value | Description |
|---|---|---|
| `json_confidence_threshold` | `0.15` | Also the lowest confidence seen by the tracker (tuned with the tracking parameters). |
| `output_json_folder_name` | `"trimmed_detections_json"` | Separate folder from sparse detection JSONs. |

---

## `process_videos_tracking`

| Parameter | Example | Description |
|---|---|---|
| `track_activation_threshold` | `0.64` | Minimum detection confidence to start a new track. Only has an effect when it is higher than `detection_threshold`. Higher values (e.g. `0.73`) give fewer swaps and false-positive tracks but start tracks later. |
| `detection_threshold` | `0.35` | Detections at or above this confidence are used in the first association step, the lower ones (down to the dense detection `json_confidence_threshold`) in the second one. |
| `minimum_matching_threshold_first_associations` | `0.94` | Maximum buffered IoU distance for the primary (high-confidence) detection-to-track assignment. Higher values are more permissive. Boxes that do not overlap once enlarged are at distance 1 and never match. |
| `minimum_matching_threshold_second_associations` | `0.66` | Maximum buffered IoU distance for the secondary assignment step, which matches the tracks left unassigned after the first step with low-confidence detections. |
| `lost_track_buffer` | `300` | Frames to keep a lost track alive before discarding it. Higher values help re-identify animals that disappear behind obstacles. |
| `buffer_scale` | `0.18` | The matching distance is 1 − IoU of the boxes enlarged on each side by `buffer_scale` × their width and height. For lost tracks the enlargement grows with sqrt(frames since the last match), so they can be recovered farther from their prediction. |
| `backward_extension_max_gap` | `30` | After tracking, extend every track backwards from its first frame with the untracked detections of any confidence, as long as a match is found within this number of frames. Tracks then start with the first detections of their animal instead of the first confident one. Uses `minimum_matching_threshold_second_associations` and `buffer_scale`. `null` disables it. |
| `interpolate_missing` | `true` | Linearly interpolate bounding boxes for frames where a track had no detection. |
| `smooth_tracks` | `true` | Apply a rolling-median filter to bounding box coordinates to reduce jitter. |
| `smoothing_period` | `9` | Window size in frames for the smoothing filter. |
| `skip_existing` | `true` | Skip videos whose tracking JSON already exists. |
| `output_json_folder_name` | `"tracking_trimmed_json"` | Folder where tracking JSONs are written. |

---

## `prepare_videos_for_tracking_examination`

| Parameter | Example | Description |
|---|---|---|
| `visualization_type` | `"tracking"` | Overlay style rendered on review videos. `"tracking"` draws bounding boxes and track IDs, `"detection"` draws bounding boxes and confidences. |
| `n_cores` | `2` | Number of parallel worker processes for rendering. |
| `skip_existing` | `true` | Skip videos whose visualisation already exists. |
| `output_video_folder_name` | `"trimmed_tracking_viz"` | Folder for visualised review videos. |
| `output_csv_filename` | `"tracking_examination.csv"` | Name of the examination CSV. One row per video with tracking quality flags. |

---

## `prepare_videos_for_cvat_track_correction`

| Parameter | Example | Description |
|---|---|---|
| `video_scale_factor` | `0.7` | Downscale factor applied to videos before CVAT upload. Reduces upload time and CVAT load. Detection coordinates in the XML are scaled to match. Must be between 0.1 and 1.0. |
| `skip_existing` | `true` | Skip videos already prepared for CVAT. |
| `output_xml_folder_name` | `"cvat_tracks_xml"` | Folder for CVAT-format XML files. |
| `output_video_folder_name` | `"cvat_tracks_videos"` | Folder for downscaled videos to upload to CVAT. |
| `output_csv_filename` | `"cvat_correction.csv"` | Summary CSV of videos prepared for correction. |

---

## `process_tracking_corrections_from_cvat`

| Parameter | Example | Description |
|---|---|---|
| `input_xml_folder_name` | `"corrected_cvat_tracks_xml"` | Folder containing XML files exported from CVAT after correction. |
| `output_json_folder_name` | `"corrected_tracks_json"` | Folder for converted and merged tracking JSONs. |

---

## `generate_tublets_from_tracking`

| Parameter | Example | Description |
|---|---|---|
| `tublet_height` | `380` | Height in pixels of each tracklet crop. Fixed for all tracklets regardless of bounding box size. |
| `tublet_width` | `380` | Width in pixels of each tracklet crop. |
| `background_padding` | `true` | Pad crops that extend beyond the frame edge with background, preserving spatial context. |
| `add_boundary_fraction` | `0.1` | Expand the bounding box by this fraction before cropping to increase spatial context. |
| `min_tublet_duration_s` | `0.5` | Discard tracklets shorter than this duration in seconds. |
| `skip_existing` | `true` | Skip tracklets whose output video already exists. |
| `output_video_folder_name` | `"tracklets"` | Folder for tracklet MP4 files. |

---

## `prepare_tracklets_for_cvat_annotation`

| Parameter | Example | Description |
|---|---|---|
| `skip_existing` | `true` | Skip tracklets already prepared for CVAT. |
| `output_xml_folder_name` | `"cvat_tracklets_xml"` | Folder for the CVAT task XML files. |
| `output_csv_filename` | `"cvat_tracklets_annotation.csv"` | Metadata CSV listing all tracklets with identifiers and source information for annotators. |
| `output_label_constructor_filename` | `"cvat_label_constructor.json"` | CVAT raw label constructor built from [`labels`](#labels), to paste in the CVAT tracklet annotation project. |

---

## `process_tracklet_annotations_from_cvat`

| Parameter | Example | Description |
|---|---|---|
| `input_xml_folder_name` | `"annotated_tracklets_xml"` | Folder containing the CVAT XML export of completed tracklet annotations. |
| `output_json_folder_name` | `"annotated_tracklets_json"` | Folder for the final annotated tracking JSONs (dataset output). |
| `always_second_action` | `["vocalizing", ...]` | Actions always treated as secondary — they will be moved to `Action2` automatically and `default_first_action` substituted into `Action`. |
| `default_first_action` | `"standing_head_up"` | Primary action assigned when an action is moved to `Action2`. |

Annotated values that are not among the configured [`labels`](#labels) are kept, and listed in a warning at the end of the stage.

---

## `video_extensions`

List of file extensions treated as video files when scanning folders. Matching is case-insensitive: `.mp4` also matches `.MP4`. Defaults to `[".mp4"]`.

```yaml
video_extensions:
  - ".mp4"
  - ".avi"
  - ".mpeg"
  - ".mpg"
```

---

## `labels`

Label classes of the three annotation categories. The pipeline does not assume any specific class: the CVAT label constructor (Step 10), the label checks (Step 11) and the colors of the dense annotation visualization are all derived from these lists.

| Category | CVAT attributes | Description |
|---|---|---|
| `species` | `Species` | Species of the individual, constant along a tracklet. |
| `activities` | `Activity` | Activity of the individual. |
| `actions` | `Action`, `Action2` | Primary and secondary actions of the individual. `Action2` also takes the value `"none"` when there is no secondary action; `"none"` must not be listed under `actions`. |

- The index of a class is its position in the list.
- The first class of each category is the default value in CVAT.

```yaml
labels:
  species:
    - "red_deer"
    - "fox"
    - "unknown"
  activities:
    - "foraging"
    - "vigilance"
    - "unknown"
  actions:
    - "walking"
    - "standing_head_up"
    - "unknown"
```
