from pathlib import Path

import numpy as np


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class OSNetEmbedder:
    """OSNet appearance encoder (via BoxMOT) producing L2-normalised person embeddings."""

    def __init__(self, weights_path: str, device: str, flip_augmentation: bool = True):
        from boxmot.reid.core.runtime import ReID

        weights = Path(weights_path).resolve()
        weights.parent.mkdir(parents=True, exist_ok=True)
        # BoxMOT downloads known checkpoints (e.g. osnet_x0_25_msmt17.pt) on first use
        self.model = ReID(weights=weights, device=device, half=False)
        self.flip_augmentation = flip_augmentation

    @staticmethod
    def crop_boxes(frame: np.ndarray, boxes_xyxy: np.ndarray) -> list[np.ndarray]:
        frame_height, frame_width = frame.shape[:2]
        crops = []
        for x1, y1, x2, y2 in boxes_xyxy.astype(int):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_width, max(x2, x1 + 1)), min(frame_height, max(y2, y1 + 1))
            crops.append(frame[y1:y2, x1:x2])
        return crops

    def embed_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, 0), dtype=np.float32)
        if not self.flip_augmentation:
            return self.model(crops)
        # Test-time augmentation: average original and mirrored views in a single batch
        mirrored = [np.ascontiguousarray(crop[:, ::-1]) for crop in crops]
        features = self.model(crops + mirrored)
        return l2_normalize(features[: len(crops)] + features[len(crops):])

    def embed(self, frame: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        return self.embed_crops(self.crop_boxes(frame, boxes_xyxy))
