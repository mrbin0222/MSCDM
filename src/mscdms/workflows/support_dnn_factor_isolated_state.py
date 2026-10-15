#!/usr/bin/env python3
"""Support dnn factor isolated state."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mscdms.workflows.support_decoded_dinov2_surrogate_capacity import (
    DINO_MEAN,
    DINO_STD,
    dino_descriptor,
)
from mscdms.workflows.support_dnn_3t3_hek293_factor_boundary import construct_view
from mscdms.workflows.support_latent_factor_probe_capacity import FeatureProbe
from mscdms.control_evaluator import classification_metrics, read_jsonl, sha256_file
from mscdms.workflows.prepare_mask_geometry import largest_component


STATES = ("3t3", "hek293")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=Path, default=Path("models/dinov2-large"))
    parser.add_argument(
        "--foreground_probe",
        type=Path,
        default=Path(
            "outputs/v4_dnn_3t3_hek293_factor_boundary_stage37a_v1/"
            "probes/3t3_vs_hek293_foreground_only/checkpoint_best.pt"
        ),
    )
    parser.add_argument(
        "--background_probe",
        type=Path,
        default=Path(
            "outputs/v4_dnn_3t3_hek293_factor_boundary_stage37a_v1/"
            "probes/3t3_vs_hek293_background_only/checkpoint_best.pt"
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--mask_manifest",
        type=Path,
        default=Path(
            "data/processed/"
            "mask_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--control_edits",
        type=Path,
        default=Path(
            "data/processed/"
            "control_disentangled_v1/val_control_edits.jsonl"
        ),
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=run directory containing generated_manifest.jsonl and diagnostics.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/v4_dnn_factor_isolated_state_stage37b_v1"),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def normalize_state(value: Any) -> str:
    label = str(value or "").strip().lower()
    suffix = " cell-line state"
    return label[: -len(suffix)].strip() if label.endswith(suffix) else label


def parse_runs(values: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for value in values:
        label, separator, raw = value.partition("=")
        if not separator or not label or not raw or label in runs:
            raise ValueError(f"invalid or duplicate --run value: {value!r}")
        runs[label] = resolve(Path(raw))
    return runs


def reconstruct_target_mask(
    template_mask: np.ndarray,
    placement: dict[str, Any],
) -> np.ndarray:
    # The Stage-33a generator calls generate_v4_foreground_renderer_stage26
    # load_mask(), which removes disconnected satellite components before
    # translating the template.  Reconstruction must be bitwise equivalent.
    template = largest_component(np.asarray(template_mask, dtype=bool))
    y, x = np.nonzero(template)
    if not len(y):
        raise ValueError("template mask is empty")
    patch = template[int(y.min()) : int(y.max() + 1), int(x.min()) : int(x.max() + 1)]
    y0, y1 = int(placement["y0"]), int(placement["y1"])
    x0, x1 = int(placement["x0"]), int(placement["x1"])
    if patch.shape != (y1 - y0, x1 - x0):
        raise ValueError("recorded placement does not match template-mask bounds")
    output = np.zeros((512, 512), dtype=bool)
    output[y0:y1, x0:x1] = patch
    if int(output.sum()) != int(template.sum()):
        raise ValueError("reconstructed mask changed template area")
    return output


class FrozenProbe(torch.nn.Module):
    def __init__(self, path: Path) -> None:
        super().__init__()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "exp4.latent_factor_probe.checkpoint.v1":
            raise ValueError(f"unsupported Stage-37a probe checkpoint: {path}")
        self.network = FeatureProbe(**payload["model_config"])
        self.network.load_state_dict(payload["model"])
        self.register_buffer("feature_mean", payload["feature_mean"].float())
        self.register_buffer("feature_std", payload["feature_std"].float())
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features.float() - self.feature_mean) / self.feature_std
        return self.network(normalized)


class GeneratedFactorDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, int, int, str, str]]
):
    def __init__(
        self,
        run_dir: Path,
        *,
        data_dir: Path,
        training_masks: dict[str, str],
        edits: dict[str, dict[str, Any]],
        states: tuple[str, ...] = STATES,
    ) -> None:
        if len(states) < 2 or len(set(states)) != len(states):
            raise ValueError("states must contain at least two unique labels")
        self.states = tuple(states)
        diagnostics = {
            str(row["generated_id"]): row
            for row in read_jsonl(run_dir / "diagnostics.jsonl")
        }
        self.rows: list[dict[str, Any]] = []
        for row in read_jsonl(run_dir / "generated_manifest.jsonl"):
            target = normalize_state(row.get("state"))
            if target not in self.states:
                continue
            generated_id = str(row["generated_id"])
            diagnostic = diagnostics.get(generated_id)
            edit = edits.get(str(row["sample_id"]))
            if diagnostic is None or edit is None:
                raise ValueError(f"missing diagnostic/edit: {generated_id}")
            template_id = str(diagnostic["template_sample_id"])
            mask_path = training_masks.get(template_id)
            if mask_path is None:
                raise ValueError(f"template is not a DNN train mask: {template_id}")
            value = dict(row)
            value["diagnostic"] = diagnostic
            value["template_mask_path"] = mask_path
            value["target_state_id"] = self.states.index(target)
            original = normalize_state(edit["original_labels"]["state"])
            value["original_state_id"] = (
                self.states.index(original) if original in self.states else -1
            )
            self.rows.append(value)
        self.rows.sort(key=lambda row: str(row["generated_id"]))
        if not self.rows or len(self.rows) % 16:
            raise ValueError("binary generated rows must form complete 16-image conditions")
        counts: dict[str, int] = defaultdict(int)
        for row in self.rows:
            counts[str(row["sample_id"])] += 1
        if set(counts.values()) != {16}:
            raise ValueError("every selected condition must contain exactly 16 images")
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    256,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(DINO_MEAN, DINO_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, str, str]:
        row = self.rows[index]
        with Image.open(self.data_dir / str(row["template_mask_path"])) as image:
            template = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        mask = reconstruct_target_mask(template, row["diagnostic"]["placement"])
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        generated_path = Path(str(row["generated_image"]))
        if not generated_path.is_file():
            raise FileNotFoundError(generated_path)
        with Image.open(generated_path) as generated:
            foreground = self.transform(construct_view(generated, mask_image, "foreground_only"))
            background = self.transform(construct_view(generated, mask_image, "background_only"))
        return (
            foreground,
            background,
            int(row["target_state_id"]),
            int(row["original_state_id"]),
            str(row["sample_id"]),
            str(row["generated_id"]),
        )


def classification(
    logits: torch.Tensor,
    labels: torch.Tensor,
    states: tuple[str, ...] = STATES,
) -> dict[str, Any]:
    predictions = logits.argmax(dim=1)
    num_classes = len(states)
    if logits.ndim != 2 or logits.shape[1] != num_classes:
        raise ValueError("probe output dimension does not match state vocabulary")
    metrics = classification_metrics(
        labels.tolist(), predictions.tolist(), num_classes=num_classes
    )
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    support = confusion.sum(axis=1)
    recall = np.divide(
        np.diag(confusion),
        support,
        out=np.zeros(num_classes, dtype=np.float64),
        where=support > 0,
    )
    metrics["per_class_recall"] = {
        label: float(recall[index]) for index, label in enumerate(states)
    }
    return metrics


def target_margin(logits: torch.Tensor, target: int) -> float:
    if logits.ndim != 1 or not (0 <= target < logits.numel()):
        raise ValueError("invalid logits or target")
    competitors = torch.cat([logits[:target], logits[target + 1 :]])
    return float(logits[target] - competitors.max())


@torch.inference_mode()
def evaluate_run(
    *,
    label: str,
    dataset: GeneratedFactorDataset,
    backbone: torch.nn.Module,
    foreground_probe: FrozenProbe,
    background_probe: FrozenProbe,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    precision: str,
    states: tuple[str, ...] = STATES,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    foreground_logits: list[torch.Tensor] = []
    background_logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    originals: list[torch.Tensor] = []
    edit_ids: list[str] = []
    generated_ids: list[str] = []
    amp_enabled = device.type == "cuda" and precision != "fp32"
    for batch_index, batch in enumerate(loader, start=1):
        foreground, background, target, original, batch_edit_ids, batch_generated_ids = batch
        pixels = torch.cat([foreground, background], dim=0).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.bfloat16):
            descriptor = dino_descriptor(backbone(pixel_values=pixels).last_hidden_state)
            foreground_feature, background_feature = descriptor.chunk(2, dim=0)
            foreground_prediction = foreground_probe(foreground_feature)
            background_prediction = background_probe(background_feature)
        foreground_logits.append(foreground_prediction.float().cpu())
        background_logits.append(background_prediction.float().cpu())
        targets.append(target)
        originals.append(original)
        edit_ids.extend(str(value) for value in batch_edit_ids)
        generated_ids.extend(str(value) for value in batch_generated_ids)
        if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(loader):
            print(
                f"{label}: evaluated {min(batch_index * batch_size, len(dataset))}/"
                f"{len(dataset)}",
                flush=True,
            )

    foreground_all = torch.cat(foreground_logits)
    background_all = torch.cat(background_logits)
    target_all = torch.cat(targets).long()
    original_all = torch.cat(originals).long()
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, edit_id in enumerate(edit_ids):
        grouped[edit_id].append(index)
    group_foreground: list[torch.Tensor] = []
    group_background: list[torch.Tensor] = []
    group_target: list[int] = []
    group_original: list[int] = []
    condition_rows: list[dict[str, Any]] = []
    for edit_id in sorted(grouped):
        indices = grouped[edit_id]
        if len(indices) != 16:
            raise ValueError(f"{edit_id}: expected 16 repeats")
        target_values = set(target_all[indices].tolist())
        original_values = set(original_all[indices].tolist())
        if len(target_values) != 1 or len(original_values) != 1:
            raise ValueError(f"{edit_id}: inconsistent state labels")
        foreground_mean = foreground_all[indices].mean(dim=0)
        background_mean = background_all[indices].mean(dim=0)
        target = next(iter(target_values))
        original = next(iter(original_values))
        group_foreground.append(foreground_mean)
        group_background.append(background_mean)
        group_target.append(target)
        group_original.append(original)
        condition_rows.append(
            {
                "edit_id": edit_id,
                "target_state": states[target],
                "original_state": states[original] if original >= 0 else None,
                "foreground_prediction": states[int(foreground_mean.argmax())],
                "background_prediction": states[int(background_mean.argmax())],
                "foreground_target_margin": target_margin(foreground_mean, target),
                "background_target_margin": target_margin(background_mean, target),
            }
        )
    grouped_foreground = torch.stack(group_foreground)
    grouped_background = torch.stack(group_background)
    grouped_target = torch.tensor(group_target, dtype=torch.long)
    grouped_original = torch.tensor(group_original, dtype=torch.long)
    anchor_selected = grouped_original >= 0
    anchor_agreement = None
    if bool(anchor_selected.any()):
        anchor_agreement = float(
            (
                grouped_background[anchor_selected].argmax(dim=1)
                == grouped_original[anchor_selected]
            ).float().mean()
        )
    return {
        "label": label,
        "n_generated": len(dataset),
        "n_conditions": len(grouped),
        "foreground_target": classification(
            grouped_foreground, grouped_target, states=states
        ),
        "background_target": classification(
            grouped_background, grouped_target, states=states
        ),
        "background_anchor": {
            "eligible_conditions": int(anchor_selected.sum()),
            "agreement": anchor_agreement,
        },
        "conditions": condition_rows,
    }


def main() -> None:
    args = parse_args()
    model_dir = resolve(args.model_dir)
    foreground_path = resolve(args.foreground_probe)
    background_path = resolve(args.background_probe)
    data_dir = resolve(args.data_dir)
    mask_path = resolve(args.mask_manifest)
    edits_path = resolve(args.control_edits)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = parse_runs(args.run)
    for path in (
        model_dir / "model.safetensors",
        foreground_path,
        background_path,
        mask_path,
        edits_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    training_masks = {
        str(row["sample_id"]): str(row["target_mask_path"])
        for row in read_jsonl(mask_path)
        if row.get("source_name") == "dynamicnuclearnet" and row.get("split") == "train"
    }
    edits = {str(row["edit_id"]): row for row in read_jsonl(edits_path)}
    device = torch.device(args.device)
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    foreground_probe = FrozenProbe(foreground_path).to(device).eval()
    background_probe = FrozenProbe(background_path).to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    results: dict[str, Any] = {}
    for label, run_dir in runs.items():
        dataset = GeneratedFactorDataset(
            run_dir,
            data_dir=data_dir,
            training_masks=training_masks,
            edits=edits,
        )
        results[label] = evaluate_run(
            label=label,
            dataset=dataset,
            backbone=backbone,
            foreground_probe=foreground_probe,
            background_probe=background_probe,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            precision=args.precision,
        )
        with (output_dir / f"{label}_conditions.jsonl").open("w", encoding="utf-8") as handle:
            for row in results[label]["conditions"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    candidate = results.get("stage36c")
    if candidate is None:
        raise ValueError("Stage-37b requires a run labelled stage36c")
    recalls = candidate["foreground_target"]["per_class_recall"]
    if recalls["3t3"] >= 0.75 and recalls["hek293"] >= 0.75:
        decision = "freeze_stage36c_and_use_foreground_isolated_dnn_sca"
    else:
        decision = "build_sparse_3t3_joint_geometry_appearance_teacher_with_foreground_judge"
    summary = {
        "schema": "exp4.dnn.factor_isolated_state.stage37b.v1",
        "status": "pass",
        "decision": decision,
        "results": results,
        "evaluator": {
            "backbone": "frozen DINOv2-L/14",
            "foreground_probe_sha256": sha256_file(foreground_path),
            "background_probe_sha256": sha256_file(background_path),
            "real_validation_population16_f1": {
                "foreground": 0.9949998749968749,
                "background": 0.9919994879672299,
            },
        },
        "generator_trained": False,
        "images_regenerated": False,
        "test_accessed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Stage-37b factor-isolated DNN state audit",
        "",
        f"- Decision: `{decision}`",
        "- Existing images only; generator trained: no",
        "- Test accessed: no",
        "",
        "| Run | Foreground target Acc/F1 | 3T3 recall | HEK293 recall | Background target Acc/F1 | Background anchor agreement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, result in results.items():
        fg = result["foreground_target"]
        bg = result["background_target"]
        anchor = result["background_anchor"]["agreement"]
        lines.append(
            f"| {label} | {fg['accuracy']:.4f}/{fg['macro_f1']:.4f} | "
            f"{fg['per_class_recall']['3t3']:.4f} | "
            f"{fg['per_class_recall']['hek293']:.4f} | "
            f"{bg['accuracy']:.4f}/{bg['macro_f1']:.4f} | "
            f"{anchor:.4f} |" if anchor is not None else
            f"| {label} | {fg['accuracy']:.4f}/{fg['macro_f1']:.4f} | "
            f"{fg['per_class_recall']['3t3']:.4f} | "
            f"{fg['per_class_recall']['hek293']:.4f} | "
            f"{bg['accuracy']:.4f}/{bg['macro_f1']:.4f} | N/A |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved", output_dir / "summary.json")
    print("saved", output_dir / "report.md")
    print("STAGE-37B DECISION:", decision)
    print("TEST DATA ACCESSED: NO")


