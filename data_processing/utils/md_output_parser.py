"""
Parses a folder of MegaDetector outputs to form a dataframe.
"""

import json
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm


class MDVideoOutputParser:
    def __init__(self) -> None:
        self.results = {}

    def parse_file(self, json_output) -> Dict:
        """
        Parses a single JSON file and returns its content as a dictionary.
        """
        with open(json_output, "r") as f:
            content = json.load(f)

        return content

    def parse_folder(self, folder) -> Dict:
        """
        Parses all JSON files in a given folder recursively.

        Args:
        folder (str): The folder containing JSON files to parse.

        Returns:
        Dict: A flat dictionary keyed by each file's path relative to
            `folder`, containing the parsed JSON content.
        """
        folder_path = Path(folder)
        assert folder_path.is_dir(), f"{folder} is not a directory or can't be read."
        json_files = list(folder_path.rglob("*.json"))

        self.results = {}
        for result_file in tqdm(json_files):
            rel_path = result_file.relative_to(folder_path)
            key = rel_path.as_posix()

            content = self.parse_file(result_file)
            content["detection_file_path"] = rel_path
            self.results[key] = content

        return self.results

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the results into a pandas DataFrame.

        file_id/group_id/event_id are read directly from each JSON's "info"
        block (written by megadetector_utils.write_results_to_file) rather
        than being inferred from the folder structure or filename.

        Returns:
        pd.DataFrame: A DataFrame with all parsed data.
        """
        detection_df = pd.DataFrame(index=None)

        for file_key, file_results in self.results.items():
            info = file_results.get("info", {})
            file_id = info.get("file_id") or Path(file_key).stem

            results_df = pd.DataFrame.from_dict(file_results["frames"])
            results_df["file_id"] = file_id
            results_df["group_id"] = info.get("group_id")
            results_df["event_id"] = info.get("event_id")
            results_df["detection_file_path"] = file_results["detection_file_path"]

            # Propagate false_positive_video flag from the "info" section
            false_positive_video_flag = info.get("false_positive_video", None)
            if false_positive_video_flag is not None:
                results_df["false_positive_video"] = false_positive_video_flag

            if len(detection_df):
                detection_df = pd.concat(
                    [detection_df, results_df], ignore_index=True
                )
            else:
                detection_df = results_df

        # Separates fields of the json detection output into columns of df
        detection_exploded_df = detection_df.explode("detections")
        detections_expanded = pd.json_normalize(detection_exploded_df["detections"])

        detection_exploded_df = detection_exploded_df.reset_index(drop=True)
        detections_expanded = detections_expanded.reset_index(drop=True)

        df = pd.concat([detection_exploded_df, detections_expanded], axis=1)

        if "conf" in df.columns:
            df["conf"] = df["conf"].astype(float)

        return df