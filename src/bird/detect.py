"""Detector stage: is there a bird, and where?"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import onnxruntime as ort

from bird.frames import Image

# COCO class index for "bird", as used by stock YOLO models.
COCO_BIRD_CLASS_ID = 14


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    w: int
    h: int

    def crop(self, image: Image) -> Image:
        return image[self.y : self.y + self.h, self.x : self.x + self.w]


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    score: float
    label: str = "bird"


class Detector(Protocol):
    id: str

    def detect(self, image: Image) -> list[Detection]: ...


class StubDetector:
    """Reports the whole frame as one bird. Lets the pipeline run before a model is chosen."""

    id = "stub-detector"

    def detect(self, image: Image) -> list[Detection]:
        h, w = image.shape[:2]
        return [Detection(bbox=BBox(0, 0, w, h), score=1.0)]


def _letterbox(
    image: Image, size: int, color: tuple[int, int, int] = (114, 114, 114)
) -> tuple[Image, float, float, float]:
    """Resize preserving aspect ratio and pad to a size x size square.

    Matches the preprocessing YOLO models are trained/exported with, so the
    model sees the same kind of input it expects. Returns the padded image
    plus the scale and (left, top) pixel offsets needed to map detections
    back to the original image.
    """
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return np.asarray(padded, dtype=np.uint8), scale, float(left), float(top)


class YoloOnnxDetector:
    """Pretrained YOLO (COCO classes), run via ONNX Runtime and filtered to "bird".

    No training, no new heavy runtime dependency (CPU-only ONNX Runtime).
    Whether this or a purpose-trained model wins is still open (see AGENTS.md).
    """

    id = "yolo11n-onnx-coco-bird"

    def __init__(
        self,
        model_path: Path,
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.45,
        input_size: int = 640,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, image: Image) -> list[Detection]:
        letterboxed, scale, pad_x, pad_y = _letterbox(image, self.input_size)
        blob = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        (output,) = self._session.run(None, {self._input_name: blob})
        predictions = output[0].T  # (num_candidates, 4 box coords + 80 class scores)

        boxes_cxcywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        best_class = np.argmax(class_scores, axis=1)
        best_score = class_scores[np.arange(len(class_scores)), best_class]

        keep = (best_class == COCO_BIRD_CLASS_ID) & (best_score >= self.conf_threshold)
        if not np.any(keep):
            return []

        cx, cy, w, h = boxes_cxcywh[keep].T
        scores = best_score[keep]
        xywh = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)

        indices = cv2.dnn.NMSBoxes(
            xywh.tolist(), scores.tolist(), self.conf_threshold, self.nms_threshold
        )

        img_h, img_w = image.shape[:2]
        detections: list[Detection] = []
        for i in np.asarray(indices).flatten():
            x, y, box_w, box_h = xywh[i]
            x = (x - pad_x) / scale
            y = (y - pad_y) / scale
            box_w /= scale
            box_h /= scale
            x0 = max(0, int(round(x)))
            y0 = max(0, int(round(y)))
            x1 = min(img_w, int(round(x + box_w)))
            y1 = min(img_h, int(round(y + box_h)))
            detections.append(
                Detection(
                    bbox=BBox(x0, y0, max(0, x1 - x0), max(0, y1 - y0)),
                    score=float(scores[i]),
                    label="bird",
                )
            )
        return detections
