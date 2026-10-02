# Pipeline Workflow

## Overview
We provide scripts to process raw camera-trap videos into per-animal annotated tracklets.  

We first parse raw videos and create a database for the project. 
Then we filter out false positive videos using MegaDetector and trim the raw videos to keep only segments containing animals. 
We perform animal tracking using an adapted ByteTrack model, and proceed to correcting these tracks in CVAT.
We create "tracklets", videos centered around each animals. This makes the annotation of individual behaviors much easier in CVAT.
Tracklets are uploaded and annotated in CVAT.
Finally, annotations are mapped back to the trimmed videos and can be visualized.
The database collects the progress of every step in the project, but user can easily collect intermediary outputs.

---

## Setup

```bash
git clone <repo_url>
uv sync && source .venv/bin/activate
```

All scripts read parameters from the same config file:
```bash
--config_file configs/<my_project>.yaml
```

---

## Step 1: Video parsing
1. Recursively find the videos of the raw videos folder, whatever its layout, and give each one a `file_id` by flattening its relative path (`siteA/cam1/IMG_0001.MP4` → `siteA_cam1_IMG_0001`). The outputs of the following steps keep the folder structure of the raw videos and are named `{file_id}.{extension}` (e.g. `siteA/cam1/siteA_cam1_IMG_0001.json`).
2. Detect infrared videos automatically using a simple SVM classifier on frame colour statistics.
3. Initialize the database and store the results

```bash
python data_processing/raw_video_parser.py --config_file <config>
```


## Step 2: Sparse animal detection

Run MegaDetector every N frames on each parsed video, writing one JSON of bounding boxes and confidence scores per video.

```bash
python data_processing/process_videos_sparse_detection.py --config_file <config>
```

> Steps 2 and 3 are required: Step 3 is the only step that fills the `VideoData` table, and Steps 6, 7 and 9 read its output videos.

---

## Steps 2b and 2c: False positive flagging and review *(optional)*

Flags false positive triggers: videos where the camera triggered but the only detections are objects of the scene.

1. `flag_false_positive_videos.py` processes the sparse detections of daytime (non-IR) videos. Detections that appear at the same location in multiple events of the same camera are flagged as `false_positive_detection`. Videos containing only such detections are flagged as `false_positive_video`. Videos without any detection are left to Step 3, which drops them.
2. It copies the detection JSONs of the videos that are not false positives to a filtered folder, which is used by Step 3.
3. It exports the first false positive video of each event, with bounding-box overlays, and produces an examination CSV.
4. Annotators complete the examination CSV (see below).
5. `process_false_positive_videos_correction.py` copies back to the filtered folder all the videos of the events reviewed as true positives. Confirmed false positives are not deleted from the disk.

```bash
python data_processing/flag_false_positive_videos.py --config_file <config>

python data_processing/process_false_positive_videos_correction.py --config_file <config>
```

**[Manual step]** Open `false_positive_examination.csv`, watch the videos in the review folder, and fill:

| Column | Value | Meaning |
|---|---|---|
| `checked` | `True` | video has been watched |
| `true_positive` | `True` | the video contains an animal: all videos of its event are put back |

---

## Step 3: Filter and trim videos from detections

1. Use the sparse detection JSONs to identify frames containing an animal above a given confidence threshold
2. Discard false positive videos (Step 2b), videos whose last animal detection is before a minimum number of frames, and videos containing humans with a high confidence.
3. Trim the kept videos from the start of the video to last frame containing an animal detection, padded by a given number of buffer frames. 

To keep the full videos, set `skip_trimming: true`. Steps 1 and 2 still run, and the kept videos are symlinked full length into the output folder instead of being trimmed.

```bash
python data_processing/trimmed_from_detection.py --config_file <config>
```

---

## Step 4: Dense detection (every frame)

1. Runs MegaDetector on every frame of the trimmed videos (`stride: 1`) using `pytorch_wildlife`
2. Save dense detection JSONs that serve as input to the tracker

```bash
python data_processing/process_videos_dense_detection.py --config_file <config>
```

---

## Step 5: Animal tracking (ByteTrack)

1. Run ByteTrack on the dense detection JSONs, assigning a consistent `track_id` to each animal across frames. Uses the adapted ByteTrack in `data_processing/utils/byte_tracker`.
2. (optionally) Interpolate gaps where the animal is temporarily undetected and smooth bounding-box jitter with a centered rolling median. 

```bash
python data_processing/process_videos_tracking.py --config_file <config>
```

---

## Step 6: Prepare videos for tracking examination

1. Render bounding boxes and track IDs on the trimmed videos and write them to a review folder 
2. Generate `tracking_examination.csv` for examination

```bash
python data_processing/prepare_videos_for_tracking_examination.py --config_file <config>
```

**[Manual step]** Watch the visualised videos and fill `tracking_examination.csv`. See the [Annotation Columns Reference](#annotation-columns-reference) below. 

---

## Step 7: Prepare videos for CVAT track correction

1. Select videos that are `checked`, not false positives, and have at least one tracking correction required. 
2. Convert their tracking JSONs to CVAT XML format and downscale the videos for upload.

```bash
python data_processing/prepare_videos_for_cvat_track_correction.py --config_file <config>
```

**[Manual step]** Upload videos and XML to CVAT, correct tracking errors, export corrected annotations as XML to `corrected_cvat_tracks_xml/`. Manually update corrected status in `cvat_correction.csv`. 
See [docs/cvat.md](cvat.md) for instructions on how to upload XML files to CVAT. 

---

## Step 8: Process CVAT track corrections

1. Reads the corrected XML files from CVAT
2. Converts them back to our JSON format
3. Merges corrections with the original tracking outputs from Step 5.

```bash
python data_processing/process_tracking_corrections_from_cvat.py --config_file <config>
```

See [docs/cvat.md](cvat.md) for instructions on how to download XML files from CVAT. 

---

## Step 9: Generate tracklets from tracking

1. Extracts a fixed-size square video crop centred on each animal for the full duration of this individual track. Pads with background context at frame edges. 
2. Discards tracks below a minimum duration. 
3. Stores tracklet metadata in the `Tracklets` database table.

```bash
python data_processing/generate_tublets_from_tracking.py --config_file <config>
```

---

## Step 10: Prepare tracklets for CVAT annotation

1. Generates a default CVAT XML annotation for each tracklet video
2. Generates a metadata CSV for annotators.
3. Generates the CVAT label constructor (`cvat_label_constructor.json`) from the `labels` classes of the config.

```bash
python data_processing/prepare_tracklets_for_cvat_annotation.py --config_file <config>
```

**[Manual step]** Create the CVAT tracklet annotation project from `cvat_label_constructor.json`. Upload to CVAT and annotate each tracklet with `Species`, `Activity`, `Action`, `Action2` and `occluded`. Export completed annotations as XML to `annotated_tracklets_xml/`.
See [docs/cvat.md](cvat.md) for instructions on how to upload XML files to CVAT. 

---

## Step 11: Process CVAT tracklet annotations

1. Reads the CVAT XML export
2. Maps each tracklet back to its track segment in the trimmed-video JSON (offset by `tublet_start_frame`) and inserts the annotations
3. Adds consistency for some labels (e.g. moving `vocalizing`/`defecating`/`urinating` to `Action2`). Any other CVAT attribute is kept as is.
4. Warns about annotated values that are not among the configured `labels` classes.
5. Save the final annotated JSON files.

```bash
python data_processing/process_tracklet_annotations_from_cvat.py --config_file <config>
```
  
See [docs/cvat.md](cvat.md) for instructions on how to download XML files from CVAT. 

---

## Annotation Columns Reference

`tracking_examination.csv` (Steps 6–7):

| Column | Description |
|---|---|
| `checked` | The video has been manually reviewed |
| `false_positive_video` | There is no animal in the video |
| `false_positive_track` | Some of the tracks are false positives (not all of them) |
| `tracking_short` | The tracker missed significant portions of the animal track |
| `segmented_track` | The tracker assigned multiple track IDs to the same animal across a video |
| `id_switch` | Two animals had their tracking IDs swapped |
| `bbox_wrong` | The bounding box localization is inaccurate (e.g., missing body part) |
| `tracking_pbm` | Any non-localisation tracking problem (derived in Step 7 from `false_positive_track`, `tracking_short`, `segmented_track` and `id_switch`) |
| `tracking_and_localization_correct` | Correct tracking, no correction needed (not computed yet) |
| `tracking_correct` | Correct tracking, but box localisations may be inaccurate, no correction needed (not computed yet) |
| `description` | Free-text notes |

---
