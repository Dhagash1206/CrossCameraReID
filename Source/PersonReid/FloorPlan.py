import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from PersonReid.Visualization import color_for_global_id


class FloorPlan:
    """Projects detections onto a shared ground plane (metres) and renders a bird's-eye map.

    Calibration JSON:
        {"world_bounds_m": [x_min, y_min, x_max, y_max], "pixels_per_meter": 40,
         "cameras": {"0": {"homography": [[...], [...], [...]]}, ...}}
    Each homography maps image pixels to floor coordinates in metres.
    """

    def __init__(self, calibration_path: str | Path, trail_length: int = 50):
        calibration = json.loads(Path(calibration_path).read_text())
        self.homographies = {
            int(camera_id): np.asarray(camera["homography"], dtype=np.float64)
            for camera_id, camera in calibration["cameras"].items()
        }
        self.world_bounds = calibration.get("world_bounds_m", [0.0, 0.0, 10.0, 10.0])
        self.pixels_per_meter = calibration.get("pixels_per_meter", 40)
        self.trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_length))
        self.current_positions: dict[int, np.ndarray] = {}

    def project(self, camera_id: int, boxes_xyxy: np.ndarray) -> np.ndarray | None:
        homography = self.homographies.get(camera_id)
        if homography is None:
            return None
        if len(boxes_xyxy) == 0:
            return np.empty((0, 2), dtype=np.float64)
        # The bottom-centre of a person box approximates where their feet touch the floor
        foot_points = np.stack([(boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2, boxes_xyxy[:, 3]], axis=1)
        return cv2.perspectiveTransform(foot_points.reshape(-1, 1, 2).astype(np.float64), homography).reshape(-1, 2)

    def update(self, positions_by_global_id: dict[int, list[np.ndarray]]) -> None:
        # A person seen by several cameras gets the average of their projected positions
        self.current_positions = {
            global_id: np.mean(np.stack(positions), axis=0)
            for global_id, positions in positions_by_global_id.items()
        }
        for global_id, position in self.current_positions.items():
            self.trails[global_id].append(position)

    def _to_canvas(self, world_xy: np.ndarray) -> tuple[int, int]:
        x_min, y_min = self.world_bounds[:2]
        return (
            int((world_xy[0] - x_min) * self.pixels_per_meter),
            int((world_xy[1] - y_min) * self.pixels_per_meter),
        )

    def render(self) -> np.ndarray:
        x_min, y_min, x_max, y_max = self.world_bounds
        width = int((x_max - x_min) * self.pixels_per_meter)
        height = int((y_max - y_min) * self.pixels_per_meter)
        canvas = np.full((height, width, 3), 30, dtype=np.uint8)
        for meter in range(int(np.ceil(x_max - x_min)) + 1):
            x = meter * self.pixels_per_meter
            cv2.line(canvas, (x, 0), (x, height), (60, 60, 60), 1)
        for meter in range(int(np.ceil(y_max - y_min)) + 1):
            y = meter * self.pixels_per_meter
            cv2.line(canvas, (0, y), (width, y), (60, 60, 60), 1)

        for global_id, trail in self.trails.items():
            if len(trail) > 1:
                points = np.array([self._to_canvas(point) for point in trail], dtype=np.int32)
                cv2.polylines(canvas, [points], False, color_for_global_id(global_id), 2)
        for global_id, position in self.current_positions.items():
            center = self._to_canvas(position)
            cv2.circle(canvas, center, 7, color_for_global_id(global_id), -1)
            cv2.putText(canvas, str(global_id), (center[0] + 9, center[1] - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, "FLOOR PLAN", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return canvas
