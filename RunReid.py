import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "Source"))

from PersonReid.PipelineConfig import PipelineConfig  # noqa: E402
from PersonReid.ReidPipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-camera person ReID (YOLO + ByteTrack + OSNet)")
    parser.add_argument("sources", nargs="+", help="One synchronised source per camera: video file, webcam index or RTSP URL")
    parser.add_argument("--config", default="Configs/Default.yaml", help="YAML config; CLI flags override it")
    parser.add_argument("--output", default="Outputs", help="Directory for videos, Tracks.csv, Summary.json, Gallery/")
    parser.add_argument("--floor-calibration", help="Floor-plan calibration JSON (from CalibrateFloor.py)")
    parser.add_argument("--match-threshold", type=float, help="Cosine similarity needed to link a track to an identity")
    parser.add_argument("--frame-stride", type=int, help="Process every Nth frame for speed")
    parser.add_argument("--device", help="auto | cpu | mps | cuda:0")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--show", action="store_true", help="Show live mosaic window (press q to quit)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig.from_yaml(args.config) if Path(args.config).exists() else PipelineConfig()
    overrides = {
        "floor_calibration": args.floor_calibration,
        "match_similarity_threshold": args.match_threshold,
        "frame_stride": args.frame_stride,
        "device": args.device,
    }
    config = replace(config, **{key: value for key, value in overrides.items() if value is not None})
    run_pipeline(args.sources, args.output, config, show=args.show, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
