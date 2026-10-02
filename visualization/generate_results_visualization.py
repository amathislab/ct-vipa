"""
Visualize detection or tracking results on a batch of videos
"""

import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import ffmpeg
import numpy as np
from tqdm.auto import tqdm

import supervision as sv
from data_processing.utils.db_utils import list_video_files
from data_processing.utils.megadetector_utils import (
    from_megadetector_to_supervision,
)


def process_video(
    source_video_path: str,
    detections_path: str,
    target_video_path: str,
    visualization_type: str,
) -> None:

    if visualization_type == "detection":
        color_lookup = sv.ColorLookup("class")
    elif visualization_type == "tracking":
        color_lookup = sv.ColorLookup("track")
    else:
        raise NotImplementedError("Support only detection and tracking visualizations")

    box_annotator = sv.BoxAnnotator(
        color_lookup=color_lookup, thickness=5
    )  # BoundingBox annotator instance
    box_fill_annotator = sv.ColorAnnotator(color_lookup=color_lookup, opacity=0.2)
    label_annotator = sv.LabelAnnotator(
        text_scale=1, color_lookup=color_lookup, text_thickness=2, smart_position=True
    )  # Label annotator instance
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=1
    )  # for generating frames from video
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    width, height = (video_info.width, video_info.height)

    with open(detections_path, "r") as f:
        data = json.load(f)

    detection_results = data["frames"]
    false_positive_video = data.get("info", {}).get("false_positive_video", None)

    detection_results_dict = {d["frame_id"]: d for d in detection_results}

    frame_idx = 0
    with sv.VideoSink(
        target_path=target_video_path, video_info=video_info, codec="mp4v"
    ) as sink:
        for frame in frame_generator:
            if (frame_idx in detection_results_dict) and (
                len(detection_results_dict[frame_idx]["detections"]) > 0
            ):
                detections = from_megadetector_to_supervision(
                    detection_results_dict[frame_idx], width, height
                )

                # Prepare labels
                labels = []
                if visualization_type == "tracking":
                    for i in range(len(detections.tracker_id)):
                        tracker_id = detections.tracker_id[i]
                        conf = detections.confidence[i]

                        label_parts = [f"#{tracker_id} {conf:.2f}"]
                        # label_parts = []

                        # Add any extra fields dynamically
                        for key, values in detections.data.items():
                            if values is not None and len(values) > i:
                                value = values[i]
                                if isinstance(value, dict):
                                    dict_string = "\n".join(
                                        [f"{k}: {v}" for k, v in value.items()]
                                    )
                                    label_parts.append(dict_string)
                                else:
                                    label_parts.append(f"{key}: {value}")

                        labels.append("\n".join(label_parts))
                elif visualization_type == "detection":
                    for i in range(len(detections.class_id)):
                        cls = detections.class_id[i]
                        conf = detections.confidence[i]

                        label_parts = [f"#{cls} {conf:.2f}"]

                        # Add any extra fields dynamically
                        for key, values in detections.data.items():
                            if values is not None and len(values) > i:
                                value = values[i]
                                if isinstance(value, (np.bool_, bool)):
                                    value = "True" if value else "False"
                                elif hasattr(value, "item"):  # for numpy types
                                    value = value.item()
                                label_parts.append(f"{key}: {value}")

                        labels.append(" | ".join(label_parts))

                # Annotating detection boxes
                annotated_frame = box_annotator.annotate(
                    scene=frame.copy(), detections=detections
                )
                annotated_frame = box_fill_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )

                # Annotating labels
                annotated_label_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=labels
                )
            else:
                annotated_label_frame = frame.copy()

            annotated_label_frame = cv2.putText(
                annotated_label_frame,
                str(frame_idx),
                org=(10, 70),
                color=(255, 255, 255),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            if false_positive_video:
                text = f"false_positive_video"
                annotated_label_frame = cv2.putText(
                    annotated_label_frame,
                    text,
                    org=(10, 140),
                    color=(0, 0, 255),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
            sink.write_frame(frame=annotated_label_frame)

            frame_idx += 1


def reencode_video(target_video_path: str, codec="libx264"):
    coded_video_path = str(
        Path(target_video_path).parent
        / (Path(target_video_path).stem + "_" + codec + ".mp4")
    )
    (
        ffmpeg.input(target_video_path)
        .output(
            coded_video_path,
            vcodec="libx264",
            pix_fmt="yuv420p",
            video_bitrate="12048k",
            **{"profile:v": "high"},
        )
        .run(quiet=True)
    )

    os.remove(target_video_path)
    os.rename(coded_video_path, target_video_path)


def reencode_video_w_audio(
    target_video_path: str, orig_video_path: str, codec="libx264"
):
    coded_video_path = str(
        Path(target_video_path).parent
        / (Path(target_video_path).stem + "_" + codec + ".mp4")
    )
    (
        ffmpeg.input(target_video_path)
        .video.output(
            ffmpeg.input(orig_video_path).audio,
            coded_video_path,
            vcodec="libx264",
            pix_fmt="yuv420p",
            video_bitrate="12048k",
            acodec="copy",
            **{"profile:v": "high"},
        )
        .run(quiet=True)
    )

    os.remove(target_video_path)
    os.rename(coded_video_path, target_video_path)


def process_single_video(video_info, reencode=True, codec="libx264"):
    """Process a single video with provided information."""
    (
        input_video_file,
        input_video_folder,
        input_json_folder,
        output_video_folder,
        skip_existing,
        visualization_type,
    ) = video_info

    rel_path = input_video_file.relative_to(input_video_folder)
    name = rel_path.stem
    json_input_file = Path(input_json_folder) / rel_path.with_suffix(".json")

    if not json_input_file.exists():
        print(f"Skipping {name} as no annotation available")
        return False

    video_output_file = Path(output_video_folder) / rel_path.with_suffix(".mp4")
    if video_output_file.parent.exists() == False:
        os.makedirs(video_output_file.parent, exist_ok=True)
    if video_output_file.exists() and skip_existing:
        return False

    process_video(
        str(input_video_file),
        str(json_input_file),
        str(video_output_file),
        visualization_type,
    )

    if reencode and codec is not None:
        reencode_video(str(video_output_file), codec)

    return True


def process_batch_video_folder(
    input_video_folder: str,
    input_json_folder: str,
    output_video_folder: str,
    skip_existing: bool = True,
    visualization_type: str = "detection",
    num_workers: int = -1,
    video_extensions: list = None,
) -> None:
    """
    Process a folder of videos for both detection (MegaDetector) and tracking (adapted ByteTrack)
    Args:
        input_video_folder: Folder containing input videos
        input_json_folder: Folder containing JSON detection results
        output_video_folder: Folder for output annotated videos
        skip_existing: Skip videos that already exist in output folder
        visualization_type: Type of visualization to apply (detection or tracking)
        num_workers: Number of parallel workers (default: CPU count - 1)
    """
    assert os.path.isdir(input_video_folder), "{} is not a folder".format(
        input_video_folder
    )

    assert os.path.isdir(input_json_folder), "{} is not a folder".format(
        input_json_folder
    )

    assert (
        output_video_folder is not None
    ), "{} You must specify an output folder".format(output_video_folder)

    # Create output directories
    os.makedirs(output_video_folder, exist_ok=True)

    # Determine number of workers
    if num_workers == -1:
        # Use CPU count minus 1 as default, but at least 1
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    # Get the paths of all videos in provided folder
    input_videos_full_paths = list_video_files(input_video_folder, video_extensions)

    print(
        f"Found {len(input_videos_full_paths)} videos to process using {num_workers} workers"
    )

    # Prepare arguments for each video
    video_args = [
        (
            video_path,
            Path(input_video_folder),
            Path(input_json_folder),
            Path(output_video_folder),
            skip_existing,
            visualization_type,
        )
        for video_path in input_videos_full_paths
    ]

    # Process videos in parallel
    processed_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use tqdm to display progress bar
        results = list(
            tqdm(
                executor.map(process_single_video, video_args),
                total=len(input_videos_full_paths),
                desc="Videos processed",
            )
        )

        # Count number of videos actually processed
        processed_count = sum(1 for result in results if result)

    print(
        f"Completed processing {processed_count} videos out of {len(input_videos_full_paths)}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-IV", "--input_video_folder", type=str, help="video folder to process"
    )

    parser.add_argument(
        "-IJ",
        "--input_json_folder",
        type=str,
        default=None,
        help="input result folder",
    )

    parser.add_argument(
        "-OV",
        "--output_video_folder",
        type=str,
        default=None,
        help="output result folder",
    )

    parser.add_argument(
        "--video_extensions",
        type=str,
        nargs="+",
        default=None,
        help="Video file extensions to process (case-insensitive, default: .mp4)",
    )

    parser.add_argument(
        "--visualization_type", choices=["detection", "tracking"], default="detection"
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Don't save if the video already exists",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)",
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    process_batch_video_folder(
        args.input_video_folder,
        args.input_json_folder,
        args.output_video_folder,
        args.skip_existing,
        args.visualization_type,
        args.num_workers,
        args.video_extensions,
    )


if __name__ == "__main__":
    main()
