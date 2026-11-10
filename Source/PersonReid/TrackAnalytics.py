import csv
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Observation:
    camera_id: int
    frame_index: int
    local_track_id: int
    global_id: int | None
    box_xyxy: tuple[float, float, float, float]
    confidence: float


class TrackAnalytics:
    """Collects per-frame observations and produces tracks, dwell-time and handoff reports."""

    def __init__(self, camera_fps: dict[int, float], frame_stride: int, segment_gap_seconds: float):
        self.camera_fps = camera_fps
        self.frame_stride = frame_stride
        self.segment_gap_seconds = segment_gap_seconds
        self.observations: list[Observation] = []
        # global_id -> (quality score, crop image) of the clearest view seen so far
        self.best_crops: dict[int, tuple[float, np.ndarray]] = {}

    def record(
        self,
        camera_id: int,
        frame_index: int,
        frame: np.ndarray,
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray,
        local_track_ids: np.ndarray,
        global_ids: dict[int, int | None],
        embeddable_mask: np.ndarray,
    ) -> None:
        for box, confidence, local_id, is_clean_view in zip(
            boxes_xyxy, confidences, local_track_ids.tolist(), embeddable_mask
        ):
            global_id = global_ids.get(local_id)
            self.observations.append(Observation(
                camera_id, frame_index, local_id, global_id,
                tuple(round(float(value), 1) for value in box), round(float(confidence), 3),
            ))
            if global_id is None or not is_clean_view:
                continue
            x1, y1, x2, y2 = box.astype(int)
            quality = float((x2 - x1) * (y2 - y1) * confidence)
            if quality > self.best_crops.get(global_id, (0.0, None))[0]:
                self.best_crops[global_id] = (quality, frame[max(0, y1):y2, max(0, x1):x2].copy())

    def _seconds(self, camera_id: int, frame_index: int) -> float:
        return frame_index / self.camera_fps[camera_id]

    def _segments(self, camera_id: int, frames: list[int]) -> list[tuple[int, int]]:
        gap_frames = self.segment_gap_seconds * self.camera_fps[camera_id]
        segments = []
        start = previous = frames[0]
        for frame_index in frames[1:]:
            if frame_index - previous > gap_frames:
                segments.append((start, previous))
                start = frame_index
            previous = frame_index
        segments.append((start, previous))
        return segments

    def build_summary(self, resolve: Callable[[int | None], int | None], frames_processed: int) -> dict:
        frames_by_identity_camera: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
        for observation in self.observations:
            global_id = resolve(observation.global_id)
            if global_id is not None:
                frames_by_identity_camera[global_id][observation.camera_id].add(observation.frame_index)

        identities = {}
        total_handoffs = 0
        people_per_camera: dict[int, int] = defaultdict(int)
        for global_id in sorted(frames_by_identity_camera):
            camera_reports = {}
            all_segments = []
            for camera_id, frame_set in sorted(frames_by_identity_camera[global_id].items()):
                people_per_camera[camera_id] += 1
                segments = self._segments(camera_id, sorted(frame_set))
                all_segments.extend((camera_id, start, end) for start, end in segments)
                camera_reports[camera_id] = {
                    "visible_seconds": round(len(frame_set) * self.frame_stride / self.camera_fps[camera_id], 2),
                    "segments_seconds": [
                        [round(self._seconds(camera_id, start), 2), round(self._seconds(camera_id, end), 2)]
                        for start, end in segments
                    ],
                }
            all_segments.sort(key=lambda segment: self._seconds(segment[0], segment[1]))
            handoffs = [
                {
                    "from_camera": previous[0],
                    "to_camera": current[0],
                    "time_seconds": round(self._seconds(current[0], current[1]), 2),
                }
                for previous, current in zip(all_segments, all_segments[1:])
                if previous[0] != current[0]
            ]
            total_handoffs += len(handoffs)
            identities[global_id] = {
                "first_seen_seconds": round(min(self._seconds(c, s) for c, s, _ in all_segments), 2),
                "last_seen_seconds": round(max(self._seconds(c, e) for c, _, e in all_segments), 2),
                "cameras": camera_reports,
                "handoffs": handoffs,
            }

        return {
            "frames_processed": frames_processed,
            "unique_people": len(identities),
            "people_per_camera": dict(people_per_camera),
            "total_handoffs": total_handoffs,
            "identities": identities,
        }

    def write_reports(
        self, output_dir: Path, resolve: Callable[[int | None], int | None], frames_processed: int
    ) -> dict:
        with open(output_dir / "Tracks.csv", "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["camera_id", "frame", "local_track_id", "global_id", "x1", "y1", "x2", "y2", "confidence"])
            for observation in self.observations:
                writer.writerow([
                    observation.camera_id, observation.frame_index, observation.local_track_id,
                    resolve(observation.global_id), *observation.box_xyxy, observation.confidence,
                ])

        summary = self.build_summary(resolve, frames_processed)
        (output_dir / "Summary.json").write_text(json.dumps(summary, indent=2))

        # Merged identities keep whichever crop is clearest
        resolved_crops: dict[int, tuple[float, np.ndarray]] = {}
        for global_id, (quality, crop) in self.best_crops.items():
            resolved_id = resolve(global_id)
            if quality > resolved_crops.get(resolved_id, (0.0, None))[0]:
                resolved_crops[resolved_id] = (quality, crop)
        gallery_dir = output_dir / "Gallery"
        gallery_dir.mkdir(exist_ok=True)
        for global_id, (_, crop) in resolved_crops.items():
            cv2.imwrite(str(gallery_dir / f"Person{global_id:03d}.jpg"), crop)
        return summary
