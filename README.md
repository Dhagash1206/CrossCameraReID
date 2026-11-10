# CrossCameraReID

Multi-camera person re-identification: a person keeps **one global ID** while moving between cameras.

```
 Camera 0 ─┐   YOLO11 ─► ByteTrack ─► local tracks ─┐
 Camera 1 ─┼─►                                      ├─► OSNet embeddings ─► Global Identity Manager ─► global IDs
 Camera N ─┘   (one detector + tracker per camera)  │        (flip TTA,         • Hungarian matching on gallery similarity
                                                    │   occlusion filtering)    • same-camera exclusivity
                         floor homography ──────────┘                           • floor-plan spatial gating
                                                                                • cannot-link aware ID merging
```

## Features

| Feature | What it does |
|---|---|
| Per-camera tracking | YOLO11 person detection + ByteTrack, one independent tracker per camera |
| Appearance ReID | OSNet (MSMT17) embeddings with horizontal-flip test-time augmentation |
| Quality filtering | Small, low-confidence or occluded crops are kept out of identity galleries |
| Global association | Top-k gallery similarity + Hungarian assignment; one person can't hold two tracks in one camera |
| Floor plan (optional) | Homography projects feet to metres; matches that are physically impossible are blocked, nearby ones boosted; live bird's-eye map with trails |
| ID consolidation | Periodically merges fragmented IDs, never merging two people seen together in one frame |
| Analytics | Dwell time per camera, visibility segments, camera handoffs, unique-people count |
| Person search | Query with a photo; returns matching identities and when/where they appeared |
| Sources | Video files, webcam index (`0`) or RTSP URLs |

## Setup

Python 3.10–3.13.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r Requirements.txt
```

YOLO and OSNet weights download automatically into `Weights/` on first run. Apple Silicon (MPS) and CUDA are detected automatically.

## Data

Any set of time-synchronised videos works. For a quick test, download two views of the
[EPFL multi-camera pedestrian dataset](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/) "4p" sequence:

```bash
mkdir -p Dataset/EpflFourPeople
curl -L -A "Mozilla/5.0" -o Dataset/EpflFourPeople/Camera0.avi https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/4p-c0.avi
curl -L -A "Mozilla/5.0" -o Dataset/EpflFourPeople/Camera1.avi https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/4p-c1.avi
```

## Usage

```bash
# Run ReID across cameras (add --show for a live mosaic window)
python RunReid.py Dataset/EpflFourPeople/Camera0.avi Dataset/EpflFourPeople/Camera1.avi

# Optional: calibrate each camera to the floor, then run with spatial gating + floor map
python CalibrateFloor.py Dataset/EpflFourPeople/Camera0.avi --camera-id 0
python CalibrateFloor.py Dataset/EpflFourPeople/Camera1.avi --camera-id 1
python RunReid.py Dataset/EpflFourPeople/Camera0.avi Dataset/EpflFourPeople/Camera1.avi \
    --floor-calibration Configs/FloorCalibration.json

# Find a person from a photo
python SearchPerson.py query.jpg --results Outputs
```

All parameters live in [Configs/Default.yaml](Configs/Default.yaml); CLI flags override them.

## Outputs (`Outputs/`)

| File | Contents |
|---|---|
| `Camera{N}Reid.mp4` | Annotated video per camera, colour-coded by global ID |
| `FloorPlan.mp4` | Bird's-eye map with trails (when calibrated) |
| `Tracks.csv` | Every detection: camera, frame, local track ID, global ID, box, confidence |
| `Summary.json` | Unique people, per-camera dwell time, visibility segments, handoffs |
| `Gallery/PersonNNN.jpg` | Clearest crop of each identity |
| `IdentityIndex.npz` | Mean embedding per identity (used by `SearchPerson.py`) |

## Project structure

```
RunReid.py              CLI: run the pipeline
SearchPerson.py         CLI: search a person by photo
CalibrateFloor.py       CLI: click floor points to build a camera homography
Configs/Default.yaml    Pipeline settings
Source/PersonReid/
  CameraStream.py       Video source + YOLO/ByteTrack per camera
  OsnetEmbedder.py      OSNet embeddings (flip TTA)
  IdentityManager.py    Cross-camera matching, spatial gating, ID merging
  FloorPlan.py          Homography projection + bird's-eye rendering
  TrackAnalytics.py     Tracks CSV, dwell/handoff summary, gallery crops
  Visualization.py      Box drawing and mosaic
  ReidPipeline.py       Main loop wiring it all together
  PipelineConfig.py     Config dataclass + YAML loader
Tests/                  Unit tests for identity association
```

## Tests

```bash
pytest
```

## Tuning tips

- **Too many IDs for one person:** lower `match_similarity_threshold` or `merge_similarity_threshold`.
- **Different people sharing an ID:** raise `match_similarity_threshold`, or add floor calibration.
- **Slow on CPU:** set `frame_stride: 2` or a smaller `inference_image_size`.
