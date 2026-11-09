from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from PersonReid.PipelineConfig import PipelineConfig

PERSON_CLASS_ID = 0


@dataclass
class TrackedDetections:
    boxes_xyxy: np.ndarray  # (N, 4) float
    confidences: np.ndarray  # (N,)
    local_track_ids: np.ndarray  # (N,) int


class CameraStream:
    """One video source with its own YOLO + ByteTrack instance (tracker state is per model)."""

    def __init__(self, camera_id: int, video_path: str, config: PipelineConfig, device: str):
        self.camera_id = camera_id
        self.video_path = video_path
        self.capture = cv2.VideoCapture(video_path)
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.config = config
        self.device = device
        self.detector = YOLO(config.detector_weights)
        self.finished = False

    def read_frame(self) -> np.ndarray | None:
        if self.finished:
            return None
        success, frame = self.capture.read()
        if not success:
            self.finished = True
            return None
        return frame

    def detect_and_track(self, frame: np.ndarray) -> TrackedDetections:
        results = self.detector.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS_ID],
            tracker=self.config.tracker_config,
            conf=self.config.detection_confidence,
            imgsz=self.config.inference_image_size,
            device=self.device,
            verbose=False,
        )[0]
        boxes = results.boxes
        # ByteTrack returns no IDs for detections not yet confirmed as tracks
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return TrackedDetections(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
        return TrackedDetections(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.id.cpu().numpy().astype(np.int64),
        )

    def release(self) -> None:
        self.capture.release()
