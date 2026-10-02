import concurrent.futures
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import humanfriendly
import pandas as pd
import torch
from PytorchWildlife.models import detection as pw_detection
from tqdm.auto import tqdm

from data_processing.utils import megadetector_utils as md_utils
from data_processing.utils.db_utils import list_video_files, output_relative_path

logger = logging.getLogger(__name__)


class VideoDetectionProcessor:
    def __init__(
        self,
        params: Dict[str, Any],
        input_video_folder: str,
        project_path: str,
        raw_videos_df: pd.DataFrame,
    ):
        # Required parameters
        self.model = params["model"]
        self.json_confidence_threshold = params["json_confidence_threshold"]
        self.input_video_folder = input_video_folder

        # Detection parameters
        self.output_json_folder = Path(project_path) / params["output_json_folder_name"]
        self.frame_folder = Path(project_path) / params.get(
            "tmp_frame_folder_name", "tmp_detection_frames"
        )
        self.skip_existing = params["skip_existing"]
        self.force_frame_folder_deletion = params["force_frame_folder_deletion"]
        self.n_cores = params.get("n_cores", 1)
        self.batch_size = params.get("batch_size", 16)
        self.stride = params.get("stride", 1)
        self.debug_max_videos = params.get("debug_max_videos", -1)

        # raw_videos_df["video_path"] is the path of each video relative to input_video_folder
        self._video_meta = raw_videos_df.set_index("video_path")[
            ["file_id", "group_id", "event_id", "raw_file_path"]
        ].to_dict("index")

        os.makedirs(self.output_json_folder, exist_ok=True)
        os.makedirs(self.frame_folder, exist_ok=True)

    def _get_model_and_runner(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "MDV6" in self.model:
            model = pw_detection.MegaDetectorV6(
                device=device,
                pretrained=True,
                version=self.model,
            )
            model.predictor.args.verbose = False

            def run_megadetector(frames_path, model, batch_size, conf_threshold):
                return model.batch_image_detection(
                    data_path=frames_path,
                    batch_size=batch_size,
                    det_conf_thres=conf_threshold,
                )

        elif self.model in ("a", "b"):  # MDv5
            model = pw_detection.MegaDetectorV5(
                device=device,
                pretrained=True,
                version=self.model,
            )

            def run_megadetector(frames_path, model, batch_size, conf_threshold):
                return model.batch_image_detection(
                    data_path=frames_path,
                    batch_size=batch_size,
                    det_conf_thres=conf_threshold,
                )

        else:
            raise ValueError(f"Unsupported model version in name: {self.model}")

        # Inference model
        model.eval()

        return model, run_megadetector

    def _video_key(self, video_path: Path) -> str:
        return video_path.relative_to(self.input_video_folder).as_posix()

    def _lookup_video_meta(self, video_path: Path) -> Dict[str, str]:
        """
        Get file_id, group_id, event_id and raw_file_path of a video from raw_videos_df
        """
        return self._video_meta[self._video_key(video_path)]

    def process_videos_folder(
        self,
        detection_categories: Dict[str, str],
        video_extensions: Optional[List[str]] = None,
    ):
        folder_start_time = time.time()

        # Get input videos
        input_files = list_video_files(self.input_video_folder, video_extensions)
        unknown_files = [
            f for f in input_files if self._video_key(f) not in self._video_meta
        ]
        if unknown_files:
            logger.warning(
                "Skipping %d videos not registered in the database: %s",
                len(unknown_files),
                [str(f) for f in unknown_files],
            )
            input_files = [f for f in input_files if f not in unknown_files]
        if self.debug_max_videos > 0:
            input_files = input_files[: self.debug_max_videos]

        if not input_files:
            logger.warning("No videos found in folder %s", self.input_video_folder)
            return

        model, run_megadetector = self._get_model_and_runner()

        # Disable autograd globally for the current thread
        torch.set_grad_enabled(False)

        # Allow only 1 thread to access GPU at a time
        infer_gate = threading.Semaphore(1)

        # Define process_single_video function for parallel execution
        def process_single_video(video_path: Path):

            video_meta = self._lookup_video_meta(video_path)
            json_output_path = Path(self.output_json_folder) / output_relative_path(
                video_meta["raw_file_path"], ".json"
            )

            if self.skip_existing and json_output_path.exists():
                logger.info("Skipping existing result: %s", json_output_path)
                return None
            os.makedirs(json_output_path.parent, exist_ok=True)

            # Create thread-specific frame folder using video name or thread ID
            thread_id = str(threading.get_ident())
            thread_frame_folder = os.path.join(self.frame_folder, f"thread_{thread_id}")

            # Ensure a clean folder
            if os.path.exists(thread_frame_folder):
                shutil.rmtree(thread_frame_folder, ignore_errors=True)
            os.makedirs(thread_frame_folder, exist_ok=True)

            try:
                # I/O-bound work
                md_utils.save_frames_to_disk(
                    str(video_path), thread_frame_folder, self.stride
                )

                # GPU-bound work
                with infer_gate:
                    results = run_megadetector(
                        frames_path=thread_frame_folder,
                        model=model,
                        batch_size=self.batch_size,
                        conf_threshold=self.json_confidence_threshold,
                    )

                width, height = md_utils.get_video_dimensions(str(video_path))

                # Get total frames from video
                cap = cv2.VideoCapture(str(video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                # Calculate the last frame to process based on stride
                max_output_frame = (total_frames - 1) // self.stride * self.stride

                # Build a dictionary mapping frame_id to detections
                results_by_frame = {}
                for r in results:
                    frame_number = (
                        int(r["img_id"].split("_")[-1].split(".")[0]) * self.stride
                    )
                    detection_list = md_utils.from_supervision_to_megadetector(
                        r.get("detections", []), width, height
                    )
                    # Remove tracker IDs if present
                    for d in detection_list:
                        d.pop("track_id", None)
                    results_by_frame[frame_number] = detection_list

                # Calculate expected number of frames
                expected_frame_count = (max_output_frame // self.stride) + 1

                # Check if all frames are already present (fast path)
                if len(results_by_frame) == expected_frame_count:
                    # All frames present, just convert dict to sorted list
                    frames_out = [
                        {"frame_id": frame_idx, "detections": detections}
                        for frame_idx, detections in sorted(results_by_frame.items())
                    ]
                else:
                    # Some frames missing, fill with empty detections
                    frames_out = [
                        {
                            "frame_id": frame_idx,
                            "detections": results_by_frame.get(frame_idx, []),
                        }
                        for frame_idx in range(0, max_output_frame + 1, self.stride)
                    ]

                md_utils.write_results_to_file(
                    frames_out,
                    str(video_path),
                    str(json_output_path),
                    file_id=video_meta["file_id"],
                    group_id=video_meta["group_id"],
                    event_id=video_meta["event_id"],
                    width=width,
                    height=height,
                    detector=self.model,
                    detection_categories=detection_categories,
                )

                return str(video_path)

            finally:
                md_utils.delete_frames_from_disk(thread_frame_folder)
                try:
                    os.rmdir(thread_frame_folder)  # Remove the empty directory
                except OSError:
                    pass  # Folder not empty or in use

        # ThreadPool for I/O + serialized inference
        n_cores = max(1, self.n_cores)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_cores) as executor:
            list(
                tqdm(
                    executor.map(process_single_video, input_files),
                    total=len(input_files),
                    desc="Videos processed",
                    leave=True,
                )
            )

        if self.force_frame_folder_deletion:
            shutil.rmtree(self.frame_folder, ignore_errors=True)
            logger.info("Deleted frame folder: %s", self.frame_folder)

        # Log total processing time
        folder_processing_time = time.time() - folder_start_time
        logger.info(
            "Processed %d videos in %s. Output saved to %s",
            len(input_files),
            humanfriendly.format_timespan(folder_processing_time),
            self.output_json_folder,
        )
