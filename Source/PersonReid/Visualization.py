import colorsys

import cv2
import numpy as np

PENDING_COLOR = (160, 160, 160)


def color_for_global_id(global_id: int) -> tuple[int, int, int]:
    # Golden-ratio hue stepping keeps consecutive IDs visually distinct
    hue = (global_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(blue * 255), int(green * 255), int(red * 255)


def draw_tracks(
    frame: np.ndarray,
    camera_id: int,
    boxes_xyxy: np.ndarray,
    local_track_ids: np.ndarray,
    global_ids: dict[int, int | None],
) -> np.ndarray:
    canvas = frame.copy()
    for box, local_id in zip(boxes_xyxy.astype(int), local_track_ids.tolist()):
        global_id = global_ids.get(local_id)
        color = PENDING_COLOR if global_id is None else color_for_global_id(global_id)
        label = f"ID {global_id}" if global_id is not None else f"? (t{local_id})"
        x1, y1, x2, y2 = box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(canvas, (x1, y1 - text_height - 8), (x1 + text_width + 6, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(canvas, f"CAM {camera_id}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return canvas


def build_mosaic(frames: list[np.ndarray], tile_height: int = 360) -> np.ndarray:
    resized = [
        cv2.resize(frame, (int(frame.shape[1] * tile_height / frame.shape[0]), tile_height))
        for frame in frames
    ]
    return np.hstack(resized)
