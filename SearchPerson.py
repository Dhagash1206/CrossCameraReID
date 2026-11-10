import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "Source"))

from PersonReid.OsnetEmbedder import OSNetEmbedder  # noqa: E402
from PersonReid.PipelineConfig import PipelineConfig  # noqa: E402
from PersonReid.ReidPipeline import resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find where a person (query photo) appeared across all cameras")
    parser.add_argument("query_image", help="Cropped photo of the person to search for")
    parser.add_argument("--results", default="Outputs", help="Output directory of a RunReid.py run")
    parser.add_argument("--reid-weights", default=PipelineConfig.reid_weights)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results)
    index = np.load(results_dir / "IdentityIndex.npz")
    summary = json.loads((results_dir / "Summary.json").read_text())
    if len(index["global_ids"]) == 0:
        sys.exit("Identity index is empty; run RunReid.py first.")

    query = cv2.imread(args.query_image)
    if query is None:
        sys.exit(f"Cannot read image: {args.query_image}")
    embedder = OSNetEmbedder(args.reid_weights, resolve_device(args.device))
    query_embedding = embedder.embed_crops([query])[0]

    similarities = index["embeddings"] @ query_embedding
    ranking = np.argsort(-similarities)[: args.top_k]
    print(f"Top {len(ranking)} matches for {args.query_image}:")
    for rank, row in enumerate(ranking, start=1):
        global_id = int(index["global_ids"][row])
        report = summary["identities"].get(str(global_id), {})
        print(f"\n{rank}. Person {global_id}  similarity={similarities[row]:.3f}  "
              f"gallery={results_dir / 'Gallery' / f'Person{global_id:03d}.jpg'}")
        for camera_id, camera_report in report.get("cameras", {}).items():
            spans = ", ".join(f"{start}s-{end}s" for start, end in camera_report["segments_seconds"])
            print(f"   camera {camera_id}: {camera_report['visible_seconds']}s visible ({spans})")


if __name__ == "__main__":
    main()
