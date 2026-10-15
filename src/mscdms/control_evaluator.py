"""Shared components for exp4 ICA/SCA/BCA/ACSR evaluation.

The evaluator deliberately uses only real images. A locally stored CLIP vision
encoder extracts frozen image embeddings and a small multitask probe predicts:

* identity: one global head;
* state: one source-specific head per semantic taxonomy;
* background: one global acquisition-domain head.

This module contains no dataset split policy. The protocol builder and training
script are responsible for enforcing train/val/test isolation.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_image_path(path_value: str, root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (root / path).resolve()


def classification_metrics(
    targets: Iterable[int],
    predictions: Iterable[int],
    *,
    num_classes: int,
) -> dict[str, Any]:
    """Return accuracy and support-aware macro metrics without sklearn."""

    target = np.asarray(list(targets), dtype=np.int64)
    prediction = np.asarray(list(predictions), dtype=np.int64)
    if target.shape != prediction.shape:
        raise ValueError(
            f"target/prediction shape mismatch: {target.shape} vs {prediction.shape}"
        )
    valid = (target >= 0) & (target < num_classes)
    target = target[valid]
    prediction = prediction[valid]
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    if len(target):
        np.add.at(confusion, (target, prediction), 1)
    support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros(num_classes, dtype=np.float64),
        where=predicted_support > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(num_classes, dtype=np.float64),
        where=support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    observed = support > 0
    total = int(confusion.sum())
    return {
        "n": total,
        "num_classes": int(num_classes),
        "observed_classes": int(observed.sum()),
        "accuracy": float(true_positive.sum() / total) if total else None,
        "macro_precision": float(precision[observed].mean())
        if observed.any()
        else None,
        "macro_recall": float(recall[observed].mean()) if observed.any() else None,
        "macro_f1": float(f1[observed].mean()) if observed.any() else None,
        "support": support.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


class ControlProbe(nn.Module):
    """Small multitask probe on top of normalized frozen image embeddings."""

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int,
        identity_classes: int,
        state_classes: dict[str, int],
        background_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.identity_classes = int(identity_classes)
        self.state_classes = dict(state_classes)
        self.background_classes = int(background_classes)
        self.trunk = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.identity_head = nn.Linear(hidden_dim, identity_classes)
        self.background_head = nn.Linear(hidden_dim, background_classes)
        self.state_heads = nn.ModuleDict(
            {
                group: nn.Linear(hidden_dim, count)
                for group, count in state_classes.items()
            }
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.trunk(F.normalize(features.float(), dim=-1))

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.encode(features)
        return (
            self.identity_head(hidden),
            self.background_head(hidden),
            {group: head(hidden) for group, head in self.state_heads.items()},
        )

    def config(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "identity_classes": self.identity_classes,
            "state_classes": self.state_classes,
            "background_classes": self.background_classes,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ControlProbe":
        return cls(**config)


def load_clip_image_encoder(
    model_dir: Path,
    device: torch.device,
    parameter_delta: dict[str, torch.Tensor] | None = None,
) -> tuple[Any, Any]:
    """Load the existing local CLIP model without network access."""

    from transformers import CLIPImageProcessor, CLIPModel

    processor = CLIPImageProcessor.from_pretrained(
        str(model_dir), local_files_only=True
    )
    model = CLIPModel.from_pretrained(str(model_dir), local_files_only=True)
    if parameter_delta:
        parameters = dict(model.named_parameters())
        unknown = sorted(set(parameter_delta) - set(parameters))
        if unknown:
            raise ValueError(
                "fine-tuned CLIP parameter delta contains unknown keys: "
                + ", ".join(unknown[:5])
            )
        with torch.no_grad():
            for name, value in parameter_delta.items():
                parameters[name].copy_(
                    value.to(
                        device=parameters[name].device,
                        dtype=parameters[name].dtype,
                    )
                )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return processor, model


@torch.inference_mode()
def extract_clip_features(
    image_paths: list[Path],
    *,
    processor: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Extract normalized CLIP image embeddings as a CPU float32 tensor."""

    all_features: list[torch.Tensor] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images: list[Image.Image] = []
        try:
            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            vision_output = model.vision_model(pixel_values=pixel_values)
            projected = model.visual_projection(vision_output.pooler_output)
            projected = F.normalize(projected.float(), dim=-1)
            all_features.append(projected.cpu())
        finally:
            for image in images:
                image.close()
    if not all_features:
        return torch.empty((0, int(model.projection_dim)), dtype=torch.float32)
    return torch.cat(all_features, dim=0)


def majority_vote(values: Iterable[str]) -> str:
    """Deterministic majority vote with lexical tie breaking."""

    counts = Counter(values)
    if not counts:
        raise ValueError("cannot vote over an empty sequence")
    best_count = max(counts.values())
    return sorted(value for value, count in counts.items() if count == best_count)[0]
