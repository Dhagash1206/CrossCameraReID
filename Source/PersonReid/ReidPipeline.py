import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from PersonReid.CameraStream import CameraStream
from PersonReid.PipelineConfig import PipelineConfig
from PersonReid.OsnetEmbedder import OSNetEmbedder
from PersonReid.IdentityManager import GlobalIdentityManager
from PersonReid.Visualization import build_mosaic, draw_tracks


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_embeddable(boxes_xyxy: np.ndarray, confidences: np.ndarray, config: PipelineConfig) -> np.ndarray:
    heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    return (heights >= config.min_crop_height) & (confidences >= config.min_embedding_confidence)


def run_pipeline(
    video_paths: list[str],
    output_dir: str,
    config: PipelineConfig,
    show: bool = False,
    max_frames: int | None = None,
) -> None:
    device = resolve_device(config.device)
    print(f"Using device: {device}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cameras = [CameraStream(index, path, config, device) for index, path in enumerate(video_paths)]
    embedder = OSNetEmbedder(config.reid_weights, device)
    identity_manager = GlobalIdentityManager(config)

    writers = [
        cv2.VideoWriter(
            str(output_path / f"cam{camera.camera_id}_reid.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            camera.fps,
            (camera.frame_width, camera.frame_height),
        )
        for camera in cameras
    ]
    csv_file = open(output_path / "tracks.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["camera_id", "frame", "local_track_id", "global_id", "x1", "y1", "x2", "y2", "confidence"])

    frame_index = 0
    try:
        # Lock-step over cameras assumes the videos are time-synchronised
        while not all(camera.finished for camera in cameras):
            if max_frames is not None and frame_index >= max_frames:
                break
            annotated_frames = []
            for camera, writer in zip(cameras, writers):
                frame = camera.read_frame()
                if frame is None:
                    continue
                detections = camera.detect_and_track(frame)

                embeddable_mask = select_embeddable(detections.boxes_xyxy, detections.confidences, config)
                embeddings: list[np.ndarray | None] = [None] * len(detections.local_track_ids)
                if embeddable_mask.any():
                    features = embedder.embed(frame, detections.boxes_xyxy[embeddable_mask])
                    for feature, detection_index in zip(features, np.flatnonzero(embeddable_mask)):
                        embeddings[detection_index] = feature

                global_ids = identity_manager.update(
                    camera.camera_id, frame_index, detections.local_track_ids, embeddings
                )

                for box, confidence, local_id in zip(
                    detections.boxes_xyxy, detections.confidences, detections.local_track_ids.tolist()
                ):
                    csv_writer.writerow([
                        camera.camera_id, frame_index, local_id, global_ids[local_id],
                        *np.round(box, 1).tolist(), round(float(confidence), 3),
                    ])

                annotated = draw_tracks(
                    frame, camera.camera_id, detections.boxes_xyxy, detections.local_track_ids, global_ids
                )
                writer.write(annotated)
                annotated_frames.append(annotated)

            if show and annotated_frames:
                cv2.imshow("Multi-camera ReID", build_mosaic(annotated_frames))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if frame_index % 50 == 0:
                print(f"frame {frame_index}: {len(identity_manager.identities)} global identities")
            frame_index += 1
    finally:
        for camera, writer in zip(cameras, writers):
            camera.release()
            writer.release()
        csv_file.close()
        if show:
            cv2.destroyAllWindows()

    print(f"Done. {frame_index} frames, {len(identity_manager.identities)} global identities.")
    print(f"Outputs written to {output_path.resolve()}")
