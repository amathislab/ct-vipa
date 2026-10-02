import json
import logging
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

import supervision as sv

from .md_output_parser import MDVideoOutputParser

logger = logging.getLogger(__name__)


def from_megadetector_to_supervision(
    frame_detection_results: Dict, width: int, height: int
) -> sv.Detections:
    """
    Converts a list of MegaDetector results dictionary (xywh) to a supervision Detections object (xyxy).

    Args:
        - frame_detection_results: List of dictionaries from MegaDetector results.
    Returns:
        - sv.Detections: A supervision Detections object with bounding boxes in xyxy format (absolute coordinates).
    """

    if frame_detection_results["detections"]:

        detections_list = frame_detection_results.get("detections", [])
        detections_list = sorted(detections_list, key=lambda x: x.get("track_id", 0))

        # Coordinates
        xywh_prop = np.array(
            [[val for val in d["bbox"]] for d in detections_list], dtype=float
        )
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
        all_keys = set().union(*(d.keys() for d in detections_list))
        extra_keys = all_keys - core_keys_present

        for key in extra_keys:
            values = [d.get(key, None) for d in detections_list]
            extra_data[key] = np.array(values)
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


def from_supervision_to_megadetector(
    detections: sv.Detections, width: int, height: int
) -> list:
    """
    Converts a supervision Detections object in xyxy format to MegaDetector-style format.
    Automatically includes all additional fields in detections.data.

    Returns:
        List[Dict]: List of detection dictionaries with keys:
          - bbox: [x_min_rel, y_min_rel, w_rel, h_rel]
          - conf: float
          - category: str
          - track_id: int or None
          - plus any extra fields from .data
    """
    detections_md = []

    for i in range(len(detections)):
        xyxy = detections.xyxy[i]
        conf = float(detections.confidence[i])
        class_id = str(int(detections.class_id[i]))

        if detections.tracker_id is not None and len(detections.tracker_id) > i:
            track_id = int(detections.tracker_id[i])
        else:
            track_id = None

        # Convert to relative MegaDetector format
        x_min = float(xyxy[0] / width)
        y_min = float(xyxy[1] / height)
        w = float((xyxy[2] - xyxy[0]) / width)
        h = float((xyxy[3] - xyxy[1]) / height)

        det_dict = {
            "bbox": [x_min, y_min, w, h],
            "conf": conf,
            "category": class_id,
            "track_id": track_id,
        }

        # Add all extra fields from .data if present
        if detections.data:
            for key, values in detections.data.items():
                if values is not None and len(values) > i:
                    det_dict[key] = (
                        values[i].item() if hasattr(values[i], "item") else values[i]
                    )

        detections_md.append(det_dict)

    return detections_md


def from_json_to_video_xml(input_json):

    root = ET.Element("annotations")
    width = input_json["info"]["width"]
    height = input_json["info"]["height"]

    max_frame = max([f["frame_id"] for f in input_json["frames"]])

    # Group detections by track_id
    track_detections = {}
    for frame in input_json["frames"]:
        for detection in frame["detections"]:
            track_id = detection["track_id"]
            if track_id not in track_detections:
                track_detections[track_id] = []
            track_detections[track_id].append((frame["frame_id"], detection["bbox"]))

    # Sort track detections by frame id
    for track_id, detections in sorted(track_detections.items()):
        track_elem = ET.SubElement(
            root, "track", id=str(track_id), label="Animal", source="semi-auto"
        )
        for idx, (frame_id, bbox) in enumerate(detections):
            box_elem = ET.SubElement(
                track_elem,
                "box",
                frame=str(frame_id),
                keyframe="1",
                outside="0",
                occluded="0",
                xtl=str(np.round(bbox[0] * width, 2)),
                ytl=str(np.round(bbox[1] * height, 2)),
                xbr=str(np.round((bbox[0] + bbox[2]) * width, 2)),
                ybr=str(np.round((bbox[1] + bbox[3]) * height, 2)),
                z_order="0",
            )

        # Add outside property for the next frame if not video end
        if frame_id + 1 <= max_frame:
            box_elem = ET.SubElement(
                track_elem,
                "box",
                frame=str(frame_id + 1),
                keyframe="1",
                outside="1",
                occluded="0",
                xtl=str(np.round(bbox[0] * width, 2)),
                ytl=str(np.round(bbox[1] * height, 2)),
                xbr=str(np.round((bbox[0] + bbox[2]) * width, 2)),
                ybr=str(np.round((bbox[1] + bbox[3]) * height, 2)),
                z_order="0",
            )

    xml_str = ET.tostring(root, encoding="utf-8")
    xml_dom = xml.dom.minidom.parseString(xml_str)
    pretty_xml_str = xml_dom.toprettyxml(indent="  ")

    return pretty_xml_str


def from_video_xml_to_json(annotations_file):
    tree = ET.parse(annotations_file)
    root = tree.getroot()
    task = root.find("meta").find("task")

    # Holder for track corrections
    annotations_dict = {}
    for track in root.findall("track"):
        track_id = int(track.attrib["id"])
        width, height = int(task.find("original_size").find("width").text), int(
            task.find("original_size").find("height").text
        )

        label = track.attrib["label"]

        for box in track.findall("box"):
            video_frame_id = int(box.attrib["frame"])
            box_coords = [
                float(box.attrib["xtl"]) / width,
                float(box.attrib["ytl"]) / height,
                (float(box.attrib["xbr"]) - float(box.attrib["xtl"])) / width,
                (float(box.attrib["ybr"]) - float(box.attrib["ytl"])) / height,
            ]
            outside = bool(int(box.attrib["outside"]))
            occluded = bool(int(box.attrib["occluded"]))

            attributes = {}
            for attribute in box.findall("attribute"):
                attribute_name = attribute.attrib["name"]
                if attribute_name == "description":
                    continue
                attribute_value = attribute.text
                attributes[attribute_name] = attribute_value

            if video_frame_id not in annotations_dict:
                annotations_dict[video_frame_id] = {}

            annotations_dict[video_frame_id][track_id] = {
                "bbox": box_coords,
                "attributes": attributes,
                "label": label,
                "outside": outside,
                "occluded": occluded,
            }

    return annotations_dict


def from_track_xml_to_json(annotation_file) -> List[Dict]:
    """
    Parses a task annotations from CVAT (XML files)
    and returns the corresponding annotations in JSON
    """
    tublet_annotations = []
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    track = root.find("track")
    for frame_annot in track.findall("box"):
        frame_id = int(frame_annot.attrib["frame"])
        occluded = bool(int(frame_annot.attrib["occluded"]))
        frame_attributes = {}
        for attribute in frame_annot.findall("attribute"):
            frame_attributes[attribute.attrib["name"]] = attribute.text

        tublet_annotations.append(
            {
                "frame_id": frame_id,
                "attributes": frame_attributes,
                "occluded": occluded,
            }
        )

    return tublet_annotations


def create_default_xml_tracklet(tublet_size):
    root = ET.Element("annotations")
    track_elem = ET.SubElement(root, "track", id="0", label="Animal", source="manual")

    # A single box is enough for the whole video as long
    # In CVAT, a box element will extend until there is one with a
    # keyframe and outside property.
    box_elem = ET.SubElement(
        track_elem,
        "box",
        frame="0",
        keyframe="1",
        outside="0",
        occluded="0",
        xtl=str(0),
        ytl=str(0),
        xbr=str(tublet_size[0]),
        ybr=str(tublet_size[1]),
        z_order="0",
    )

    xml_str = ET.tostring(root, encoding="utf-8")
    xml_dom = xml.dom.minidom.parseString(xml_str)
    pretty_xml_str = xml_dom.toprettyxml(indent="  ")

    return pretty_xml_str


def get_video_dimensions(video_file: str) -> Tuple[int, int]:
    """
    Extracts the width and height from the video file.

    Args:
        video_file (str): Path to the video file.

    Returns:
        (width, height): Tuple of video dimensions.

    Raises:
        ValueError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError("Cannot open video file: " + video_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def save_frames_to_disk(video_path: str, output_folder: str, stride: int) -> None:
    """
    Extract every <stride> frames from a video and save them to disk as JPEGs,
    named by their index among the extracted frames.
    """
    os.makedirs(output_folder, exist_ok=True)

    for index, frame in enumerate(
        sv.get_video_frames_generator(source_path=video_path, stride=stride)
    ):
        frame_path = os.path.join(output_folder, f"frame_{index:05d}.jpg")
        cv2.imwrite(frame_path, frame)


def delete_frames_from_disk(frame_folder: str) -> None:
    """
    Deletes all .jpg files in the folder.
    """
    if not os.path.exists(frame_folder):
        return

    files = os.listdir(frame_folder)

    # Filter to get only .jpg files
    jpg_files = [
        f
        for f in files
        if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(frame_folder, f))
    ]

    # Delete all .jpg files
    for f in jpg_files:
        try:
            os.remove(os.path.join(frame_folder, f))
        except Exception as e:
            logger.warning("Failed to delete file %s: %s", f, str(e))


def write_results_to_file(
    results,
    input_video_file,
    output_file,
    file_id,
    group_id,
    event_id,
    width,
    height,
    detector,
    detection_categories,
):
    """
    Writes list of detection results to JSON output file. Format matches:

    https://github.com/agentmorris/MegaDetector/tree/master/api/batch_processing#batch-processing-api-output-format

    Args:
    - results: list of dict, each dict represents detections on one image
    - input_video_file: input video file name
    - output_file: str, path to JSON output file, should end in '.json'
    - file_id: unique identifier for this video (its path relative to the
        raw video folder, flattened with "_"). This is the DB primary key and
        the stem of every output file of this video.
    - group_id: identifier for the camera/location this video was grouped
        under (derived from the folder it was found in).
    - event_id: the event this video was assigned to during raw video
        parsing, if known.
    - width, height: video dimensions in pixels
    - detector: name of the detector model used to generate these results
    - detection_categories: mapping of detector category ids to label names

    Returns:
    - Complete output dictionary that was written to the output file.
    """

    # The info struct
    info = {
        "detection_completion_time": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "format_version": "1.3",
        "video_file": input_video_file,
    }

    info["detector"] = detector
    info["detector_metadata"] = f"Pytorch-Wildlife Version {version('PytorchWildlife')}"
    info["file_id"] = file_id
    info["group_id"] = group_id
    info["event_id"] = event_id
    info["width"] = width
    info["height"] = height

    final_output = {
        "frames": results,
        "detection_categories": detection_categories,
        "info": info,
    }

    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=2)
    logger.debug("Output file saved at %s", output_file)
    return final_output


def outputs_df(md_input_folder: str) -> pd.DataFrame:
    """
    Parses the json files of md_input_folder into a dataframe with one row per detection.

    Args:
        md_input_folder (str): The folder containing the json files from MegaDetector.

    Returns:
        detection_df (pd.DataFrame)
    """
    parser = MDVideoOutputParser()
    _ = parser.parse_folder(md_input_folder)
    detection_df = parser.to_dataframe()

    return detection_df


def load_all_detections(json_folder):
    """
    Loads all detections from JSON files, with the file_id/group_id of each
    file's "info" block.
    """
    all_detections = []
    for fname in Path(json_folder).rglob("*.json"):
        with open(fname) as f:
            data = json.load(f)

        info = data.get("info", {})
        false_positive_video = info.get("false_positive_video", None)
        file_id = info.get("file_id")
        group_id = info.get("group_id")

        for frame in data.get("frames", []):
            for det in frame.get("detections", []):
                all_detections.append(
                    {
                        "file_id": file_id,
                        "group_id": group_id,
                        "frame_id": frame["frame_id"],
                        "tracking_id": det.get("track_id", None),
                        "detection_file_path": str(fname),
                        "detections": det,
                        "false_positive_video": false_positive_video,
                        "bbox": det["bbox"],
                        "conf": det["conf"],
                        "category": det["category"],
                        "false_positive_detection": det.get("false_positive_detection", None),
                    }
                )
    return pd.DataFrame(all_detections)
