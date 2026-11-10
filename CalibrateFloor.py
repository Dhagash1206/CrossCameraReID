import argparse
import json
from pathlib import Path

import cv2
import numpy as np

WINDOW_NAME = "Click >= 4 floor points, then press Enter (Esc to cancel, u to undo)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a camera-to-floor homography by clicking floor points with known positions (metres)"
    )
    parser.add_argument("source", help="Video file or image from the camera")
    parser.add_argument("--camera-id", type=int, required=True, help="Camera index as passed to RunReid.py")
    parser.add_argument("--output", default="Configs/FloorCalibration.json")
    parser.add_argument("--pixels-per-meter", type=int, default=40, help="Floor-plan render scale")
    return parser.parse_args()


def read_first_frame(source: str) -> np.ndarray:
    capture = cv2.VideoCapture(source)
    success, frame = capture.read()
    capture.release()
    if not success:
        raise SystemExit(f"Cannot read a frame from {source}")
    return frame


def collect_image_points(frame: np.ndarray) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    while True:
        canvas = frame.copy()
        for number, point in enumerate(points, start=1):
            cv2.circle(canvas, point, 5, (0, 0, 255), -1)
            cv2.putText(canvas, str(number), (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(points) >= 4:
            break
        if key == 27:
            raise SystemExit("Calibration cancelled")
        if key == ord("u") and points:
            points.pop()
    cv2.destroyAllWindows()
    return points


def main() -> None:
    args = parse_args()
    image_points = collect_image_points(read_first_frame(args.source))

    world_points = []
    for number, point in enumerate(image_points, start=1):
        x_meters, y_meters = map(float, input(f"Point {number} at pixel {point}: floor X Y in metres: ").split())
        world_points.append((x_meters, y_meters))

    homography, _ = cv2.findHomography(np.float64(image_points), np.float64(world_points))
    if homography is None:
        raise SystemExit("Homography failed: points may be collinear")
    projected = cv2.perspectiveTransform(np.float64(image_points).reshape(-1, 1, 2), homography).reshape(-1, 2)
    error = np.linalg.norm(projected - np.float64(world_points), axis=1).mean()
    print(f"Mean reprojection error: {error:.3f} m")

    output_path = Path(args.output)
    calibration = json.loads(output_path.read_text()) if output_path.exists() else {"cameras": {}}
    calibration["cameras"][str(args.camera_id)] = {"homography": homography.tolist()}
    calibration["pixels_per_meter"] = args.pixels_per_meter
    # Grow the rendered floor area to cover every camera's calibration points
    world = np.float64(world_points)
    margin = 1.0
    new_bounds = [
        world[:, 0].min() - margin, world[:, 1].min() - margin,
        world[:, 0].max() + margin, world[:, 1].max() + margin,
    ]
    old_bounds = calibration.get("world_bounds_m", new_bounds)
    calibration["world_bounds_m"] = [
        float(min(old_bounds[0], new_bounds[0])), float(min(old_bounds[1], new_bounds[1])),
        float(max(old_bounds[2], new_bounds[2])), float(max(old_bounds[3], new_bounds[3])),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(calibration, indent=2))
    print(f"Saved camera {args.camera_id} calibration to {output_path}")


if __name__ == "__main__":
    main()
