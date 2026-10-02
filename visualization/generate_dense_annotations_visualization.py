"""
Render the final annotated tracks (Species, Activity, Action) on the trimmed videos.
Tracks are colored according to the label classes defined in the config file.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import supervision as sv
import yaml
from tqdm.auto import tqdm

from data_processing.utils.db_utils import list_video_files
from data_processing.utils.label_utils import (
    LABEL_ATTRIBUTES,
    allowed_values,
    get_label_classes,
)

# --color_by option -> annotated attribute used to color the tracks
COLOR_BY_ATTRIBUTE: Dict[str, str] = {
    "species": "Species",
    "activity": "Activity",
    "action": "Action",
    "action2": "Action2",
}


def color_class_id(
    detection: Dict, attribute: str, label_classes: Dict[str, List[str]]
) -> int:
    """
    Index of the annotated value among the configured classes, or an extra index for missing or unknown values
    """
    values = allowed_values(label_classes, attribute)
    value = detection.get("attributes", {}).get(attribute)
    return values.index(value) if value in values else len(values)


def from_json(
    frame_detection_results,
    width,
    height,
    color_by: str,
    label_classes: Dict[str, List[str]],
) -> sv.Detections:
    if frame_detection_results["detections"]:
        detections_list = frame_detection_results.get("detections", [])
        # Coordinates
        xywh_prop = np.array([d["bbox"] for d in detections_list], dtype=float)
        xyxy_coord = np.zeros_like(xywh_prop)
        xyxy_coord[:, 0] = xywh_prop[:, 0] * width
        xyxy_coord[:, 1] = xywh_prop[:, 1] * height
        xyxy_coord[:, 2] = (xywh_prop[:, 0] + xywh_prop[:, 2]) * width
        xyxy_coord[:, 3] = (xywh_prop[:, 1] + xywh_prop[:, 3]) * height
        xyxy_coord = xyxy_coord.round().astype(int)

        # Core fields
        confidence = np.array([float(d["conf"]) for d in detections_list])
        class_id = np.array([int(d["category"]) for d in detections_list])

        # Optional field: track_id
        if "track_id" in detections_list[0]:
            tracker_id = np.array([int(d["track_id"]) for d in detections_list])
        else:
            tracker_id = None

        # Extra fields
        core_keys_present = {"bbox", "conf", "category"}
        if tracker_id is not None:
            core_keys_present.add("track_id")

        extra_data = {}

        # Label attributes first, followed by any other annotated attribute
        attribute_keys = list(LABEL_ATTRIBUTES)
        for detection in detections_list:
            for attribute_key in detection.get("attributes", {}):
                if attribute_key not in attribute_keys:
                    attribute_keys.append(attribute_key)

        for attribute_key in attribute_keys:
            values = [
                d.get("attributes", {}).get(attribute_key) for d in detections_list
            ]
            extra_data[attribute_key] = np.array(values)

        if color_by in COLOR_BY_ATTRIBUTE:
            class_id = np.array(
                [
                    color_class_id(
                        detection, COLOR_BY_ATTRIBUTE[color_by], label_classes
                    )
                    for detection in detections_list
                ]
            )

        detections = sv.Detections(
            xyxy=xyxy_coord,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
            data=extra_data,
        )
    else:
        detections = sv.Detections.empty()

    return detections


def process_video(
    source_video_path: str,
    detections_path: str,
    target_video_path: str,
    color_by: str,
    label_classes: Dict[str, List[str]],
) -> None:

    color_lookup = sv.ColorLookup("class")

    box_annotator = sv.BoxAnnotator(
        color_lookup=color_lookup, thickness=2
    )  # BoundingBox annotator instance
    box_fill_annotator = sv.ColorAnnotator(color_lookup=color_lookup, opacity=0.2)
    label_annotator = sv.LabelAnnotator(
        text_scale=1, color_lookup=color_lookup, text_thickness=2, smart_position=False
    )  # Label annotator instance
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=1
    )  # for generating frames from video
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    width, height = (video_info.width, video_info.height)

    with open(detections_path, "r") as f:
        detection_results = json.load(f)["frames"]

    frame_idx = 0
    with sv.VideoSink(
        target_path=target_video_path, video_info=video_info, codec="mp4v"
    ) as sink:
        for frame in tqdm(frame_generator):
            if len(detection_results[frame_idx]["detections"]) > 0:
                detections = from_json(
                    detection_results[frame_idx],
                    width,
                    height,
                    color_by=color_by,
                    label_classes=label_classes,
                )

                # Prepare labels
                labels = []
                for i in range(len(detections.tracker_id)):
                    # Annotating labels
                    label_parts = []

                    # Add any extra fields dynamically
                    for key, values in detections.data.items():
                        if values is not None and len(values) > i:
                            value = values[i]
                            if isinstance(value, dict):
                                dict_string = "\n".join(
                                    [
                                        f"{k}:{v}"
                                        for k, v in value.items()
                                        if (v is not None and v != "none")
                                    ]
                                )
                                label_parts.append(dict_string)
                            elif value is not None and value != "none":
                                label_parts.append(f"{key}: {value}")

                    labels.append("\n".join(label_parts))
                    # labels.append(" | ".join(str(detections.data[l][tracker_id]) for l in detections.data if detections.data[l][tracker_id] != "none").replace("_", " "))

                # Annotating detection boxes
                annotated_frame = box_annotator.annotate(
                    scene=frame.copy(), detections=detections
                )
                annotated_frame = box_fill_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )

                annotated_label_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=labels
                )
            else:
                annotated_label_frame = frame.copy()

            annotated_label_frame = cv2.putText(
                annotated_label_frame,
                str(frame_idx),
                org=(10, 20),
                color=(255, 255, 255),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            sink.write_frame(frame=annotated_label_frame)

            frame_idx += 1

    # Add audio


def process_batch_video_folder(options):
    """
    Process a folder of videos for both detection (MegaDetector) and tracking (adapted ByteTrack)
    Args:
        options: options
    """

    ## Check inputs
    assert os.path.isdir(options.input_video_folder), "{} is not a folder".format(
        options.input_video_folder
    )

    assert os.path.isdir(options.input_json_folder), "{} is not a folder".format(
        options.input_json_folder
    )

    assert (
        options.output_video_folder is not None
    ), "{} You must specify an output folder".format(options.output_video_folder)

    with open(options.config_file, "r") as f:
        config = yaml.safe_load(f)
    label_classes = get_label_classes(config)

    ## Get the paths of all videos in provided folder
    input_videos_full_paths = list_video_files(
        options.input_video_folder, config.get("video_extensions")
    )

    ## Process each video individually
    for input_video_file in tqdm(
        input_videos_full_paths,
        leave=True,
        position=0,
        total=len(input_videos_full_paths),
    ):
        rel_path = input_video_file.relative_to(options.input_video_folder)
        json_input_file = Path(options.input_json_folder) / rel_path.with_suffix(
            ".json"
        )
        if not json_input_file.exists():
            print(f"Skipping {input_video_file.stem} as no tracks annotation available")
            continue
        video_output_file = Path(options.output_video_folder) / rel_path
        os.makedirs(video_output_file.parent, exist_ok=True)
        if video_output_file.exists() and options.skip_existing:
            continue

        process_video(
            str(input_video_file),
            str(json_input_file),
            str(video_output_file),
            options.color_by,
            label_classes,
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("input_video_folder", type=str, help="video folder to process")

    parser.add_argument(
        "input_json_folder", type=str, default=None, help="output result folder"
    )

    parser.add_argument(
        "output_video_folder", type=str, default=None, help="output result folder"
    )

    parser.add_argument(
        "--color_by",
        choices=list(COLOR_BY_ATTRIBUTE) + ["individual"],
        type=str,
        default="action",
        help="Color of the animal tracks",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="YAML config file (e.g. configs/default.yaml) defining the label classes and video extensions",
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Don't save if the video already exists",
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    process_batch_video_folder(args)


if __name__ == "__main__":
    main()
