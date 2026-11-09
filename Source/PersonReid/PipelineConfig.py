from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # Detection / per-camera tracking
    detector_weights: str = "yolo11n.pt"
    tracker_config: str = "bytetrack.yaml"
    detection_confidence: float = 0.4
    inference_image_size: int = 640

    # Appearance embeddings
    reid_weights: str = "osnet_x0_25_msmt17.pt"
    min_crop_height: int = 64  # tiny crops produce unreliable embeddings
    min_embedding_confidence: float = 0.5

    # Cross-camera identity association
    min_track_hits_before_match: int = 8
    match_similarity_threshold: float = 0.62
    top_k_similarities: int = 3
    local_track_embedding_buffer: int = 30
    global_gallery_size: int = 100
    gallery_update_interval: int = 5
    local_track_ttl_frames: int = 60

    device: str = "auto"
