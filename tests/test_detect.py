from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from bird.detect import YoloOnnxDetector
from bird.frames import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "yolo11n.onnx"
PHOTOS_DIR = REPO_ROOT / "tests" / "fixtures" / "real_photos"

bird_photos = sorted(PHOTOS_DIR.glob("bird_*.*")) if PHOTOS_DIR.is_dir() else []
NUM_BIRD_PHOTOS = 3
assert len(bird_photos) >= NUM_BIRD_PHOTOS, (
    f"Fewer than {NUM_BIRD_PHOTOS} bird photos found in {PHOTOS_DIR}"
)
no_bird_photos = sorted(PHOTOS_DIR.glob("no_bird.*")) if PHOTOS_DIR.is_dir() else []

if not MODEL_PATH.is_file():
    pytest.skip(f"model not found at {MODEL_PATH}", allow_module_level=True)
if not bird_photos or not no_bird_photos:
    pytest.skip(f"sample photos not found in {PHOTOS_DIR}", allow_module_level=True)


@pytest.fixture(scope="module")
def detector() -> YoloOnnxDetector:
    return YoloOnnxDetector(MODEL_PATH)


def _load(path: Path) -> Image:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert image is not None, f"could not read {path}"
    return np.asarray(image, dtype=np.uint8)


@pytest.mark.parametrize("path", bird_photos, ids=lambda p: p.stem)
def test_detects_bird_in_photo(detector: YoloOnnxDetector, path: Path) -> None:
    detections = detector.detect(_load(path))
    assert any(d.label == "bird" and d.score >= detector.conf_threshold for d in detections)


@pytest.mark.parametrize("path", no_bird_photos, ids=lambda p: p.stem)
def test_no_bird_detected_in_empty_photo(detector: YoloOnnxDetector, path: Path) -> None:
    detections = detector.detect(_load(path))
    assert detections == []
