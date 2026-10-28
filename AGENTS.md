# birdwatcher
Play with bird detection and recognition

# Software Design

Data pipeline runs continuously as a three-tier cascade, cheapest stage first:
1. Pull the latest camera frame (capture thread always holds the newest frame; the processing
   loop pulls from it, so slow stages drop frames instead of building lag).
2. Motion gate: OpenCV background subtraction on a downscaled frame. Drops empty-scene frames.
3. Detector: small neural object detector confirms "bird" and returns bounding boxes.
4. Classifier: local species model runs on each crop. No cloud calls.
5. Store: SQLite metadata plus saved crops on disk.

Decisions:
- Think in visits, not frames. A detection within `debounce_seconds` of the previous one
  extends the open visit; otherwise the visit closes and a new one opens. One DB row per visit
  plus per-frame observations, not one row per frame.
- Persist images. Every detection saves its crop under `data/crops/YYYY/MM/DD/`. This is the
  future labeled dataset and lets a better classifier be re-run over history.
- Record provenance. Rows carry detector and classifier identifiers so results stay comparable
  as models change.
- Each stage is a Protocol with a trivial stub implementation, so the whole loop runs on the
  Mac with a directory of images and no model. `bird replay <dir>` does this.

Package layout (`src/bird/`):
- `frames.py` - `FrameSource` protocol; `CameraSource` (OpenCV), `DirectorySource` (tests/replay)
- `motion.py` - `MotionGate`
- `detect.py` - `Detector` protocol, `Detection`; `StubDetector`
- `classify.py` - `Classifier` protocol, `Prediction`; `StubClassifier`
- `store.py` - SQLite `Store` with `visit` and `observation` tables, crop file writer
- `pipeline.py` - orchestrator, capture thread, visit debounce
- `config.py` - `pydantic-settings` `Settings`
- `cli.py` - `click` commands: `run`, `replay`, `snapshot`, `focus`

`scripts/` is kept for one-off camera experiments. Tests live in `tests/`.

# Tech Stack

Python:
- Python 3.13, pinned in `.python-version`
- uv - package manager; keep `uv.lock` committed
- click - CLI
- pydantic-settings - typed, validated configuration (env vars / .env)
- hatchling - build backend
- ruff - linting and formatting
- mypy (strict) - type checking
- pytest - testing
- pre-commit - runs ruff, mypy, pytest on commit
- just - task runner (`just setup`, `just test`, `just lint`, `just fmt`, `just check`)
- Start with the most recent stable versions of all libraries.

Computer vision:
- OpenCV 4
- ML runtime: ONNX Runtime preferred; PyTorch only if a chosen model requires it. Inference must
  run CPU-only on the deployment laptop.

ELP High Speed USB Zoom Camera:
- Sensor: Sony IMX577 (12MP)
- Resolutions & Frame Rates:
- 4K @ 60fps
- 1080p @ 230fps
- 12MP @ 30fps
- Interface: USB 3.0 (USB 2.0 compatible)
- Lens: 5–50mm manual zoom (10×), manual focus
- Platform support: Windows, Linux, macOS, Raspberry Pi, Jetson Nano
- Manual: https://manuals.plus/asin/B0CDGX5HYZ

# Development and Deployment

Develop on MacBook (M4 Pro, 24 GB RAM). OpenCV uses the AVFoundation backend.
Deploy on Ubuntu Linux (System76 Galago Pro laptop, vintage 2017, CPU only). OpenCV uses V4L2.
Treat the Mac as the test rig; model choices are constrained by the Linux box.

# Focusing the camera

This camera has manual focus.
Tools:
- On the Mac, use Camera Controller (brew install camera-controller) to set manual focus.
- On Linux, use fswebcam or guvcview — live preview with basic controls; guvcview shows a live histogram which helps
- `bird focus` shows a live Laplacian-variance focus score overlay

# Open Questions

- Which small detector (YOLO-nano "bird" class vs purpose-trained) and which species classifier
  (NABirds / iNaturalist-trained). Benchmark on the Linux box before committing.
- Capture settings on Linux V4L2 (MJPEG fourcc, 1080p) vs Mac AVFoundation.
- Image retention policy for `data/crops`.
- Ground-truth labeling for evaluation (a small hand-labeled crop set).
- Unattended operation: systemd unit, restart on camera disconnect, log rotation.
- Daylight gating of the loop.
- A results viewer (CLI report first).
