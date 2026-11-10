"""Multi-camera person re-identification with YOLO + OSNet."""

import os

# Must be set before torch initialises MPS: caps the GPU allocator at half of unified memory
# so a cache growing from variable batch shapes cannot consume all system RAM.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
# Torch rejects a low watermark above the high one, and its default low ratio is 1.4
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
# Ops not yet implemented on MPS run on CPU instead of raising NotImplementedError
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from PersonReid.PipelineConfig import PipelineConfig

__all__ = ["PipelineConfig"]
