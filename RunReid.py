import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "Source"))

from PersonReid.PipelineConfig import PipelineConfig  # noqa: E402
from PersonReid.ReidPipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-camera person ReID (YOLO + ByteTrack + OSNet)")
    parser.add_argument("videos", nargs="+", help="One synchronised video per camera")
    parser.add_argument("--output", default="outputs", help="Directory for annotated videos and tracks.csv")
    parser.add_argument("--detector", default=PipelineConfig.detector_weights)
    parser.add_argument("--reid-weights", default=PipelineConfig.reid_weights)
    parser.add_argument("--match-threshold", type=float, default=PipelineConfig.match_similarity_threshold)
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda:0")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--show", action="store_true", help="Show live mosaic window (press q to quit)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        detector_weights=args.detector,
        reid_weights=args.reid_weights,
        match_similarity_threshold=args.match_threshold,
        device=args.device,
    )
    run_pipeline(args.videos, args.output, config, show=args.show, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
