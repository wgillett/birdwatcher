# 0001: Bird detector — pretrained YOLO11-nano via ONNX Runtime

## Status

Accepted. Implemented in commits `01f14f6` and `1ea8d2a`.

## Context

`src/bird/detect.py` only had `StubDetector`, which ignores the image
entirely and always reports "one bird covering the whole frame." It let the
rest of the pipeline (`frames.py`, `motion.py`, `pipeline.py`, `store.py`,
`cli.py`) run end-to-end before a real model existed, but it didn't actually
detect anything.

AGENTS.md's "Open Questions" section left the detector choice open: YOLO-nano
"bird" class vs. a purpose-trained detector, to be benchmarked on the Linux
deployment box. The tech stack calls for CPU-only inference with ONNX Runtime
preferred over PyTorch. No training data or labeling pipeline existed yet, so
a purpose-trained model wasn't an option for a first slice — the goal was a
real detector, proven against real photos, to unblock further pipeline work.

## Decision

Use a pretrained YOLO11-nano model (COCO's 80 classes), exported once locally
to ONNX and committed to `models/yolo11n.onnx` (~10 MB), run via
`onnxruntime.InferenceSession` with `CPUExecutionProvider`, filtered to COCO
class id 14 (`"bird"`). Implemented as `YoloOnnxDetector` in
`src/bird/detect.py`, alongside — not replacing — `StubDetector` (which
stays in use for replay/pipeline tests that don't want a real model).

No training, and no new heavy runtime dependency: `ultralytics`/`torch` are
only needed once, locally, to do the export; the committed `.onnx` file plus
`onnxruntime` is all production inference needs, on both the Mac dev box and
the Ubuntu deployment laptop.

### Implementation notes

- **Preprocessing**: letterbox-resize to 640x640 (preserve aspect ratio,
  pad), BGR→RGB, normalize to `[0, 1]`, HWC→CHW — matching how YOLO models
  are trained/exported, so the model sees what it expects.
- **Postprocessing**: decode the `(1, 84, 8400)` output (4 box coords + 80
  class scores, no separate objectness term), keep only candidates whose top
  class is `COCO_BIRD_CLASS_ID` above `conf_threshold` (default 0.4), run NMS
  via `cv2.dnn.NMSBoxes` (OpenCV is already a dependency — reused instead of
  hand-rolling NMS), then invert the letterbox transform to map boxes back to
  original-image pixel coordinates.
- **Multiple birds**: `detect()` returns one `Detection` per surviving box,
  not just the best one — verified with a two-blue-jays test fixture
  (`tests/test_detect.py::test_detects_two_birds_in_same_frame`).

### Test fixtures (`tests/fixtures/real_photos/`)

Real iPhone photos drive `tests/test_detect.py`, not synthetic images, per
naming convention:
- `bird_*.jpg` — one file per bird photo (any number; free-form after the
  prefix, e.g. `bird_house_sparrow.jpg`)
- `no_bird.jpg` — deliberately feeder-agnostic name (not `empty_feeder.jpg`),
  since the project aims to detect/classify birds generically, not just at a
  feeder

HEIC is explicitly unsupported — `cv2.imread` has no HEIC codec on most
platforms, and adding `pillow-heif` support was ruled out of scope for this
slice. Fixtures must be JPG/PNG.

**Privacy**: this repo (`wgillett/birdwatcher`) is public on GitHub. iPhone
photos embed EXIF metadata including GPS coordinates. All committed fixture
photos had EXIF stripped (Pillow re-encode) before committing, to avoid
publishing the feeder's exact location. Stripping EXIF without first baking
in the EXIF orientation transform turned two photos sideways (the raw sensor
data was landscape; the EXIF `orientation` tag was the only thing telling
viewers to rotate it to portrait) — caught by a before/after visual check of
detection crops, fixed by rotating the affected photos 90° before re-saving.
Lesson: when stripping EXIF from photos that may need orientation
correction, bake the rotation into the pixels first (e.g.
`PIL.ImageOps.exif_transpose`) rather than dropping the tag outright.

## Why this over a purpose-trained model

No labeled dataset or training pipeline exists yet. A pretrained COCO model
unblocks a real, working detector and an end-to-end real-image test today.
Benchmarking a purpose-trained bird detector against this baseline remains
open (see AGENTS.md).

## Consequences / follow-ups

- `YoloOnnxDetector` is not yet wired into `cli.py`/`pipeline.py`'s default
  build — the CLI still defaults to `StubDetector`. Purely additive so far.
- The classifier stage (species ID) and the purpose-trained-vs-YOLO
  benchmarking question are unaffected and remain open.
- `conf_threshold`/`nms_threshold`/`input_size` defaults (0.4 / 0.45 / 640)
  are untuned — chosen as reasonable YOLO defaults, not benchmarked.
