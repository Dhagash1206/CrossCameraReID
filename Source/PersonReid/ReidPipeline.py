import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from PersonReid.CameraStream import CameraStream
from PersonReid.FloorPlan import FloorPlan
from PersonReid.IdentityManager import GlobalIdentityManager
from PersonReid.OsnetEmbedder import OSNetEmbedder
from PersonReid.PipelineConfig import PipelineConfig
from PersonReid.TrackAnalytics import TrackAnalytics
from PersonReid.Visualization import build_mosaic, draw_tracks


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def occlusion_ratios(boxes_xyxy: np.ndarray) -> np.ndarray:
    """For each box, the largest fraction of its area covered by any other box."""
    if len(boxes_xyxy) < 2:
        return np.zeros(len(boxes_xyxy))
    top_left = np.maximum(boxes_xyxy[:, None, :2], boxes_xyxy[None, :, :2])
    bottom_right = np.minimum(boxes_xyxy[:, None, 2:], boxes_xyxy[None, :, 2:])
    intersection = np.prod(np.clip(bottom_right - top_left, 0, None), axis=2)
    np.fill_diagonal(intersection, 0)
    areas = np.prod(boxes_xyxy[:, 2:] - boxes_xyxy[:, :2], axis=1)
    return (intersection / np.clip(areas[:, None], 1e-6, None)).max(axis=1)


def select_embeddable(boxes_xyxy: np.ndarray, confidences: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Only large, confident, unoccluded crops feed the galleries; the rest would pollute them."""
    heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    return (
        (heights >= config.min_crop_height)
        & (confidences >= config.min_embedding_confidence)
        & (occlusion_ratios(boxes_xyxy) <= config.max_occlusion_ratio)
    )


def run_pipeline(
    sources: list[str],
    output_dir: str,
    config: PipelineConfig,
    show: bool = False,
    max_frames: int | None = None,
) -> dict:
    device = resolve_device(config.device)
    print(f"Using device: {device}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    Path(config.detector_weights).parent.mkdir(parents=True, exist_ok=True)

    cameras = [CameraStream(index, source, config, device) for index, source in enumerate(sources)]
    embedder = OSNetEmbedder(config.reid_weights, device, config.flip_augmentation)
    identity_manager = GlobalIdentityManager(config)
    floor_plan = FloorPlan(config.floor_calibration, config.trail_length) if config.floor_calibration else None
    analytics = TrackAnalytics(
        {camera.camera_id: camera.fps for camera in cameras}, config.frame_stride, config.segment_gap_seconds
    )

    video_codec = cv2.VideoWriter_fourcc(*"mp4v")
    writers = [
        cv2.VideoWriter(
            str(output_path / f"Camera{camera.camera_id}Reid.mp4"), video_codec,
            camera.fps / config.frame_stride, (camera.frame_width, camera.frame_height),
        )
        for camera in cameras
    ]
    floor_writer = None

    frame_index = 0
    processed_frames = 0
    start_time = time.perf_counter()
    try:
        # Lock-step over cameras assumes the sources are time-synchronised
        while not all(camera.finished for camera in cameras):
            if max_frames is not None and frame_index >= max_frames:
                break
            frames = [camera.read_frame() for camera in cameras]
            if all(frame is None for frame in frames):
                break
            if frame_index % config.frame_stride != 0:
                frame_index += 1
                continue

            annotated_frames = []
            floor_positions: dict[int, list[np.ndarray]] = defaultdict(list)
            for camera, writer, frame in zip(cameras, writers, frames):
                if frame is None:
                    continue
                detections = camera.detect_and_track(frame)

                embeddable_mask = select_embeddable(detections.boxes_xyxy, detections.confidences, config)
                embeddings: list[np.ndarray | None] = [None] * len(detections.local_track_ids)
                if embeddable_mask.any():
                    features = embedder.embed(frame, detections.boxes_xyxy[embeddable_mask])
                    for feature, detection_index in zip(features, np.flatnonzero(embeddable_mask)):
                        embeddings[detection_index] = feature

                world_positions = floor_plan.project(camera.camera_id, detections.boxes_xyxy) if floor_plan else None
                global_ids = identity_manager.update(
                    camera.camera_id, frame_index, detections.local_track_ids, embeddings, world_positions
                )
                analytics.record(
                    camera.camera_id, frame_index, frame, detections.boxes_xyxy, detections.confidences,
                    detections.local_track_ids, global_ids, embeddable_mask,
                )
                if world_positions is not None:
                    for local_id, position in zip(detections.local_track_ids.tolist(), world_positions):
                        if global_ids[local_id] is not None:
                            floor_positions[global_ids[local_id]].append(position)

                annotated = draw_tracks(
                    frame, camera.camera_id, detections.boxes_xyxy, detections.local_track_ids, global_ids
                )
                writer.write(annotated)
                annotated_frames.append(annotated)

            processed_frames += 1
            if processed_frames % config.merge_interval_frames == 0:
                for merged_id, kept_id in identity_manager.consolidate():
                    print(f"  merged identity {merged_id} -> {kept_id}")

            if floor_plan is not None:
                floor_plan.update(floor_positions)
                floor_panel = floor_plan.render()
                if floor_writer is None:
                    floor_writer = cv2.VideoWriter(
                        str(output_path / "FloorPlan.mp4"), video_codec,
                        cameras[0].fps / config.frame_stride, (floor_panel.shape[1], floor_panel.shape[0]),
                    )
                floor_writer.write(floor_panel)
                annotated_frames.append(floor_panel)

            if show and annotated_frames:
                cv2.imshow("Multi-camera ReID", build_mosaic(annotated_frames))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if processed_frames % 50 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"frame {frame_index}: {len(identity_manager.identities)} identities, "
                      f"{processed_frames / elapsed:.1f} FPS")
            frame_index += 1
    finally:
        for camera, writer in zip(cameras, writers):
            camera.release()
            writer.release()
        if floor_writer is not None:
            floor_writer.release()
        if show:
            cv2.destroyAllWindows()

    # Final consolidation catches fragments created since the last periodic pass
    identity_manager.consolidate()
    summary = analytics.write_reports(output_path, identity_manager.resolve, processed_frames)
    identity_ids, identity_embeddings = identity_manager.identity_index()
    np.savez(output_path / "IdentityIndex.npz", global_ids=identity_ids, embeddings=identity_embeddings)

    elapsed = time.perf_counter() - start_time
    print(f"Done. {processed_frames} frames in {elapsed:.1f}s ({processed_frames / max(elapsed, 1e-6):.1f} FPS)")
    print(f"Unique people: {summary['unique_people']} | handoffs: {summary['total_handoffs']} "
          f"| per camera: {summary['people_per_camera']}")
    print(f"Outputs written to {output_path.resolve()}")
    return summary
