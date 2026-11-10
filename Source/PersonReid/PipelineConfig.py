from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class PipelineConfig:
    # Detection / per-camera tracking
    detector_weights: str = "Weights/yolo11n.pt"
    tracker_config: str = "bytetrack.yaml"
    detection_confidence: float = 0.4
    inference_image_size: int = 640
    frame_stride: int = 1

    # Appearance embeddings
    reid_weights: str = "Weights/osnet_x0_25_msmt17.pt"
    flip_augmentation: bool = True
    min_crop_height: int = 64  # tiny crops produce unreliable embeddings
    min_embedding_confidence: float = 0.5
    max_occlusion_ratio: float = 0.3

    # Cross-camera identity association
    min_track_hits_before_match: int = 8
    match_similarity_threshold: float = 0.62
    top_k_similarities: int = 3
    local_track_embedding_buffer: int = 30
    global_gallery_size: int = 100
    gallery_update_interval: int = 5
    local_track_ttl_frames: int = 60

    # Identity consolidation (merging fragmented IDs)
    merge_interval_frames: int = 100
    merge_similarity_threshold: float = 0.75

    # Floor plan / spatial gating (enabled when a calibration file is given)
    floor_calibration: str | None = None
    spatial_gate_meters: float = 1.5
    spatial_time_window_frames: int = 5
    spatial_bonus: float = 0.1
    trail_length: int = 50

    # Analytics
    segment_gap_seconds: float = 1.0

    device: str = "auto"
    cache_clear_interval_frames: int = 100

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        import yaml

        with open(path) as config_file:
            values = yaml.safe_load(config_file) or {}
        unknown_keys = set(values) - {field.name for field in fields(cls)}
        if unknown_keys:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown_keys)}")
        return cls(**values)
