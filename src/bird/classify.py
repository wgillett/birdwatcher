"""Classifier stage: what kind of bird is in this crop?"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import onnxruntime as ort

from bird.frames import Image

# NABirds class labels carry sex/age/plumage/catalog-id suffixes after the
# species name (e.g. "Northern_Cardinal_Adult_Male", "Blue_Jay_ID_950").
# Stop collapsing a label into a plain species name at the first such token.
_LABEL_STOP_WORDS = {
    "ID",
    "Adult",
    "Male",
    "Female",
    "Juvenile",
    "Immature",
    "Breeding",
    "Nonbreeding",
    "Winter",
    "Summer",
    "Plumage",
    "Morph",
    "Eclipse",
}


@dataclass(frozen=True)
class Prediction:
    species: str
    probability: float


class Classifier(Protocol):
    id: str

    def classify(self, crop: Image) -> list[Prediction]:
        """Return predictions sorted by descending probability."""
        ...


class StubClassifier:
    """Always answers "unknown". Lets the pipeline run before a model is chosen."""

    id = "stub-classifier"

    def classify(self, crop: Image) -> list[Prediction]:
        return [Prediction(species="unknown", probability=1.0)]


def _species_name(raw_label: str) -> str:
    """Collapse a fine-grained NABirds class label to a plain species name.

    "Blue_Jay_ID_950" -> "Blue Jay"; "Northern_Cardinal_Adult_Male" -> "Northern Cardinal".
    """
    words: list[str] = []
    for word in raw_label.replace("_", " ").split():
        if word in _LABEL_STOP_WORDS or word.isdigit():
            break
        words.append(word)
    return " ".join(words) if words else raw_label


class NABirdsOnnxClassifier:
    """Pretrained NABirds species classifier (EfficientNetV2-S), run via ONNX Runtime.

    Model: Ornimetrics/ornimetrics-edge on Hugging Face, trained on NABirds (555
    classes, collapsed here to species-level names). Non-commercial research use
    per NABirds dataset terms (see docs/decisions/0002-species-classifier-nabirds.md).
    """

    id = "nabirds-onnx-efficientnetv2s"

    def __init__(self, model_path: Path, labels_path: Path | None = None, top_k: int = 5) -> None:
        labels_path = labels_path or model_path.with_suffix(".json")
        config = json.loads(labels_path.read_text())
        self.input_size: int = config["input_size"]
        self.top_k = top_k
        self._mean = np.array(config["rgb_mean"], dtype=np.float32)
        self._std = np.array(config["rgb_std"], dtype=np.float32)
        self._species_names = [_species_name(label) for label in config["classes"]]
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def classify(self, crop: Image) -> list[Prediction]:
        size = (self.input_size, self.input_size)
        resized = cv2.resize(crop, size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self._mean) / self._std
        blob = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        (logits,) = self._session.run(None, {self._input_name: blob})
        logits = logits[0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()

        merged: dict[str, float] = {}
        for species, prob in zip(self._species_names, probs, strict=True):
            merged[species] = merged.get(species, 0.0) + float(prob)

        ranked = sorted(merged.items(), key=lambda item: item[1], reverse=True)
        return [Prediction(species=s, probability=p) for s, p in ranked[: self.top_k]]
