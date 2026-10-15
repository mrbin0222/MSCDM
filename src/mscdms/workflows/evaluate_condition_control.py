#!/usr/bin/env python3
"""Evaluate condition control."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mscdms.workflows.evaluate_control_metrics import attach_targets
from mscdms.control_evaluator import (
    ControlProbe,
    classification_metrics,
    extract_clip_features,
    load_clip_image_encoder,
    read_jsonl,
    resolve_image_path,
    sha256_file,
    write_json,
    write_jsonl,
)


IDENTITY_PATTERN = re.compile(
    r"^(?P<entity>.+?) with "
    r"(?P<morphology>elongated|oval|round) morphology and "
    r"(?P<area>large|medium|small) area$",
    re.IGNORECASE,
)
FORMAL_STATE_SOURCES = ("bbbc021", "dynamicnuclearnet")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp4_root", type=Path, default=root)
    parser.add_argument("--generated_manifest", type=Path, required=True)
    parser.add_argument(
        "--protocol_dir",
        type=Path,
        default=Path(
            "data/processed/control_metrics_v2"
        ),
    )
    parser.add_argument(
        "--final_protocol",
        type=Path,
        default=Path("outputs/v4_control_metric_protocol_final_v1/protocol.json"),
    )
    parser.add_argument(
        "--shared_evaluator_dir",
        type=Path,
        default=Path("outputs/v4_control_evaluators_finetuned_v2"),
    )
    parser.add_argument(
        "--bbbc_evaluator_dir",
        type=Path,
        default=Path(
            "outputs/v4_control_source_expert_suite_v3/experts/bbbc021"
        ),
    )
    parser.add_argument(
        "--clip_model_dir",
        type=Path,
        default=Path("models/clip-vit-large-patch14"),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["auto", "paired", "edits"], default="auto")
    parser.add_argument("--control_edits", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sca_bootstrap_reps", type=int, default=500)
    parser.add_argument("--bootstrap_seed", type=int, default=20260731)
    parser.add_argument(
        "--allow_incomplete_edit_groups",
        action="store_true",
        help="Diagnostic only: skip edit conditions that do not contain 16 images.",
    )
    return parser.parse_args()


def from_root(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def observable_identity(label: str) -> str:
    match = IDENTITY_PATTERN.fullmatch(label.strip())
    if match is None:
        raise ValueError(f"unsupported identity label: {label}")
    return (
        f"{match.group('entity').lower()}|"
        f"{match.group('morphology').lower()}"
    )


def observable_identity_groups(
    full_labels: list[str],
) -> tuple[list[str], list[list[int]], dict[str, int]]:
    labels: list[str] = []
    label_to_id: dict[str, int] = {}
    groups: list[list[int]] = []
    for full_id, full_label in enumerate(full_labels):
        label = observable_identity(full_label)
        if label not in label_to_id:
            label_to_id[label] = len(labels)
            labels.append(label)
            groups.append([])
        groups[label_to_id[label]].append(full_id)
    return labels, groups, label_to_id


def project_identity_predictions(
    logits: torch.Tensor,
    groups: list[list[int]],
) -> torch.Tensor:
    full_to_observable = torch.empty(
        logits.shape[1],
        dtype=torch.long,
        device=logits.device,
    )
    for observable_id, group in enumerate(groups):
        full_to_observable[
            torch.as_tensor(group, dtype=torch.long, device=logits.device)
        ] = observable_id
    return full_to_observable[logits.argmax(dim=1)]


def load_probe(
    evaluator_dir: Path,
    *,
    vocab_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], ControlProbe]:
    checkpoint_path = evaluator_dir / "checkpoint_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint["vocab_sha256"] != sha256_file(vocab_path):
        raise ValueError(f"{checkpoint_path}: vocabulary hash mismatch")
    model = ControlProbe.from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model.to(device).eval()


@torch.inference_mode()
def shared_logits(
    features: torch.Tensor,
    *,
    model: ControlProbe,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    identities: list[torch.Tensor] = []
    backgrounds: list[torch.Tensor] = []
    dynamic_states: list[torch.Tensor] = []
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size].to(device)
        identity, background, states = model(batch)
        identities.append(identity.cpu())
        backgrounds.append(background.cpu())
        dynamic_states.append(states["dynamicnuclearnet"].cpu())
    return (
        torch.cat(identities),
        torch.cat(backgrounds),
        torch.cat(dynamic_states),
    )


@torch.inference_mode()
def source_state_logits(
    features: torch.Tensor,
    *,
    model: ControlProbe,
    source: str,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    logits: list[torch.Tensor] = []
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size].to(device)
        _, _, states = model(batch)
        logits.append(states[source].cpu())
    return torch.cat(logits)


def target_ids(
    rows: list[dict[str, Any]],
    *,
    identity_to_id: dict[str, int],
    vocab: dict[str, Any],
) -> tuple[list[int], list[int], list[int]]:
    identities: list[int] = []
    states: list[int] = []
    backgrounds: list[int] = []
    for row in rows:
        target = row["_target"]
        identities.append(identity_to_id[observable_identity(target["identity"])])
        state_group = target["state_group"]
        state_label = target["state"]
        state_label_to_id = (
            vocab["state"].get(state_group, {}).get("label_to_id", {})
        )
        # Some samples from a formal SCA source deliberately have no formal
        # state target.  For example, BBBC021 DMSO controls have no MoA class
        # in the frozen vocabulary.  Keep them available to ICA/BCA while
        # excluding them from formal SCA instead of inventing a state class.
        if (
            state_group not in FORMAL_STATE_SOURCES
            or state_label is None
            or state_label not in state_label_to_id
        ):
            states.append(-1)
        else:
            states.append(int(state_label_to_id[state_label]))
        backgrounds.append(
            int(vocab["background"]["label_to_id"][target["background"]])
        )
    return identities, states, backgrounds


def nonoverlapping_population_bags(
    rows: list[dict[str, Any]],
    *,
    source: str,
    state_targets: list[int],
    bag_size: int,
) -> tuple[list[list[int]], int]:
    by_state: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if (
            row["_target"]["state_group"] == source
            and state_targets[index] >= 0
        ):
            by_state[state_targets[index]].append(index)
    bags: list[list[int]] = []
    dropped = 0
    for state_id in sorted(by_state):
        indices = sorted(by_state[state_id], key=lambda index: rows[index]["generated_id"])
        usable = len(indices) // bag_size * bag_size
        bags.extend(
            indices[start : start + bag_size]
            for start in range(0, usable, bag_size)
        )
        dropped += len(indices) - usable
    return bags, dropped


def score_population_bags(
    bags: list[list[int]],
    *,
    logits_by_global_index: dict[int, torch.Tensor],
    state_targets: list[int],
    num_classes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    targets: list[int] = []
    predictions: list[int] = []
    records: list[dict[str, Any]] = []
    for bag_index, indices in enumerate(bags):
        logits = torch.stack([logits_by_global_index[index] for index in indices])
        mean_logits = logits.mean(dim=0)
        target = state_targets[indices[0]]
        if any(state_targets[index] != target for index in indices):
            raise ValueError("population bag mixes target states")
        prediction = int(mean_logits.argmax())
        targets.append(target)
        predictions.append(prediction)
        records.append(
            {
                "bag_index": bag_index,
                "member_indices": indices,
                "target_id": target,
                "prediction_id": prediction,
                "correct": prediction == target,
            }
        )
    return (
        classification_metrics(targets, predictions, num_classes=num_classes),
        records,
    )


def summarize_distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
    }


def bootstrap_population_score(
    rows: list[dict[str, Any]],
    *,
    source: str,
    state_targets: list[int],
    logits_by_global_index: dict[int, torch.Tensor],
    num_classes: int,
    bag_size: int,
    bootstrap_reps: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_state: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if (
            row["_target"]["state_group"] == source
            and state_targets[index] >= 0
        ):
            by_state[state_targets[index]].append(index)
    rng = np.random.default_rng(seed)
    accuracies: list[float] = []
    macro_f1s: list[float] = []
    records: list[dict[str, Any]] = []
    for bootstrap_index in range(bootstrap_reps):
        targets: list[int] = []
        predictions: list[int] = []
        for state_id in sorted(by_state):
            candidates = np.asarray(by_state[state_id], dtype=np.int64)
            selected = rng.choice(
                candidates,
                size=bag_size,
                replace=len(candidates) < bag_size,
            )
            mean_logits = torch.stack(
                [
                    logits_by_global_index[int(index)]
                    for index in selected.tolist()
                ]
            ).mean(dim=0)
            prediction = int(mean_logits.argmax())
            targets.append(state_id)
            predictions.append(prediction)
            records.append(
                {
                    "source": source,
                    "bootstrap_index": bootstrap_index,
                    "target_id": state_id,
                    "prediction_id": prediction,
                    "correct": prediction == state_id,
                    "member_generated_ids": [
                        rows[int(index)]["generated_id"]
                        for index in selected.tolist()
                    ],
                    "sampled_with_replacement": len(candidates) < bag_size,
                }
            )
        metrics = classification_metrics(
            targets,
            predictions,
            num_classes=num_classes,
        )
        accuracies.append(float(metrics["accuracy"]))
        macro_f1s.append(float(metrics["macro_f1"]))
    support = [
        len(by_state.get(class_id, []))
        for class_id in range(num_classes)
    ]
    return (
        {
            "n_cells": int(sum(support)),
            "class_support": support,
            "observed_classes": len(by_state),
            "bag_size": bag_size,
            "bootstrap_reps": bootstrap_reps,
            "bags_per_bootstrap": len(by_state),
            "accuracy": summarize_distribution(accuracies),
            "macro_f1": summarize_distribution(macro_f1s),
            "sampling": (
                "without replacement when class support >= bag size; "
                "with replacement otherwise"
            ),
        },
        records,
    )


def rate(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [bool(row[key]) for row in rows]
    return float(np.mean(values)) if values else None


def acsr_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(selected),
            "target_success_rate": rate(selected, "target_success"),
            "preservation_rate": rate(selected, "preservation_success"),
            "acsr": rate(selected, "acsr_success"),
        }

    return {
        "by_attribute": {
            attribute: summarize(
                [row for row in rows if row["target_attribute"] == attribute]
            )
            for attribute in ["identity", "state", "background"]
        },
        "in_domain_identity_state": summarize(
            [
                row
                for row in rows
                if row["target_attribute"] in {"identity", "state"}
            ]
        ),
        "background_compositional_ood": summarize(
            [row for row in rows if row["target_attribute"] == "background"]
        ),
    }


def format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def main() -> None:
    args = parse_args()
    root = args.exp4_root.resolve()
    generated_manifest = from_root(args.generated_manifest, root)
    protocol_dir = from_root(args.protocol_dir, root)
    final_protocol_path = from_root(args.final_protocol, root)
    shared_evaluator_dir = from_root(args.shared_evaluator_dir, root)
    bbbc_evaluator_dir = from_root(args.bbbc_evaluator_dir, root)
    clip_model_dir = from_root(args.clip_model_dir, root)
    output_dir = from_root(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_protocol = load_json(final_protocol_path)
    if final_protocol["status"] != "pass":
        raise RuntimeError("final control metric protocol has not passed")
    bag_size = int(final_protocol["SCA"]["group_size"])
    if bag_size != int(final_protocol["ACSR"]["samples_per_edit_condition"]):
        raise ValueError("SCA and ACSR group sizes differ")
    formal_sources = tuple(
        source["source"] for source in final_protocol["SCA"]["formal_sources"]
    )
    if formal_sources != FORMAL_STATE_SOURCES:
        raise ValueError(
            f"unexpected formal SCA sources: {formal_sources}; "
            f"expected {FORMAL_STATE_SOURCES}"
        )

    generated = read_jsonl(generated_manifest)
    if not generated:
        raise ValueError(f"{generated_manifest} is empty")
    generated_ids = [str(row["generated_id"]) for row in generated]
    if len(generated_ids) != len(set(generated_ids)):
        raise ValueError("generated IDs are not unique")
    mode, rows, target_manifest = attach_targets(
        generated,
        mode=args.mode,
        protocol_dir=protocol_dir,
        root=root,
        control_edits=args.control_edits,
    )
    vocab_path = protocol_dir / "label_vocab.json"
    vocab = load_json(vocab_path)
    identity_labels, identity_groups, identity_to_id = observable_identity_groups(
        vocab["identity"]["id_to_label"]
    )
    identity_targets, state_targets, background_targets = target_ids(
        rows,
        identity_to_id=identity_to_id,
        vocab=vocab,
    )

    image_root = generated_manifest.parent
    paths = [
        resolve_image_path(str(row["generated_image"]), image_root)
        for row in rows
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} generated images are missing; first={missing[0]}"
        )
    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )

    shared_checkpoint, shared_probe = load_probe(
        shared_evaluator_dir,
        vocab_path=vocab_path,
        device=device,
    )
    processor, encoder = load_clip_image_encoder(
        clip_model_dir,
        device,
        parameter_delta=shared_checkpoint.get("vision_parameter_delta"),
    )
    features = extract_clip_features(
        paths,
        processor=processor,
        model=encoder,
        device=device,
        batch_size=args.batch_size,
    )
    identity_logits, background_logits, dynamic_logits = shared_logits(
        features,
        model=shared_probe,
        device=device,
        batch_size=args.batch_size,
    )
    del features, encoder, processor, shared_probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    state_logits_by_index: dict[int, torch.Tensor] = {
        index: dynamic_logits[index]
        for index, row in enumerate(rows)
        if row["_target"]["state_group"] == "dynamicnuclearnet"
    }
    bbbc_indices = [
        index
        for index, row in enumerate(rows)
        if row["_target"]["state_group"] == "bbbc021"
    ]
    if bbbc_indices:
        bbbc_checkpoint, bbbc_probe = load_probe(
            bbbc_evaluator_dir,
            vocab_path=vocab_path,
            device=device,
        )
        processor, encoder = load_clip_image_encoder(
            clip_model_dir,
            device,
            parameter_delta=bbbc_checkpoint.get("vision_parameter_delta"),
        )
        bbbc_features = extract_clip_features(
            [paths[index] for index in bbbc_indices],
            processor=processor,
            model=encoder,
            device=device,
            batch_size=args.batch_size,
        )
        bbbc_logits = source_state_logits(
            bbbc_features,
            model=bbbc_probe,
            source="bbbc021",
            device=device,
            batch_size=args.batch_size,
        )
        state_logits_by_index.update(
            {
                global_index: bbbc_logits[local_index]
                for local_index, global_index in enumerate(bbbc_indices)
            }
        )
        del bbbc_features, bbbc_logits, encoder, processor, bbbc_probe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    identity_predictions = project_identity_predictions(
        identity_logits,
        identity_groups,
    ).tolist()
    background_predictions = background_logits.argmax(dim=1).tolist()
    per_image: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        state_group = row["_target"]["state_group"]
        state_prediction: str | None = None
        if index in state_logits_by_index:
            state_id = int(state_logits_by_index[index].argmax())
            state_prediction = vocab["state"][state_group]["id_to_label"][state_id]
        per_image.append(
            {
                "generated_id": row["generated_id"],
                "sample_id": row["sample_id"],
                "repeat": int(row.get("repeat", 0)),
                "generated_image": row["generated_image"],
                "target_identity": identity_labels[identity_targets[index]],
                "predicted_identity": identity_labels[identity_predictions[index]],
                "target_background": vocab["background"]["id_to_label"][
                    background_targets[index]
                ],
                "predicted_background": vocab["background"]["id_to_label"][
                    background_predictions[index]
                ],
                "state_group": state_group,
                "target_state": row["_target"]["state"],
                "predicted_state_diagnostic": state_prediction,
                "formal_single_image_state_decision": False,
            }
        )

    direct = {
        "ICA": classification_metrics(
            identity_targets,
            identity_predictions,
            num_classes=len(identity_labels),
        ),
        "BCA": classification_metrics(
            background_targets,
            background_predictions,
            num_classes=len(vocab["background"]["id_to_label"]),
        ),
    }
    population_records: list[dict[str, Any]] = []
    population_reports: dict[str, Any] = {}
    acsr_conditions: list[dict[str, Any]] = []

    if mode == "paired":
        for source_index, source in enumerate(FORMAL_STATE_SOURCES):
            count = len(vocab["state"][source]["id_to_label"])
            report, records = bootstrap_population_score(
                rows,
                source=source,
                state_targets=state_targets,
                logits_by_global_index=state_logits_by_index,
                num_classes=count,
                bag_size=bag_size,
                bootstrap_reps=args.sca_bootstrap_reps,
                seed=args.bootstrap_seed + source_index * 100,
            )
            for record in records:
                record["target"] = vocab["state"][source]["id_to_label"][
                    record["target_id"]
                ]
                record["prediction"] = vocab["state"][source]["id_to_label"][
                    record["prediction_id"]
                ]
            population_records.extend(records)
            population_reports[source] = report
        direct["SCA"] = {
            "by_source": population_reports,
            "source_macro_accuracy": float(
                np.mean(
                    [
                        report["accuracy"]["mean"]
                        for report in population_reports.values()
                    ]
                )
            ),
            "source_macro_f1": float(
                np.mean(
                    [
                        report["macro_f1"]["mean"]
                        for report in population_reports.values()
                    ]
                )
            ),
            "accuracy": float(
                np.mean(
                    [
                        report["accuracy"]["mean"]
                        for report in population_reports.values()
                    ]
                )
            ),
            "macro_f1": float(
                np.mean(
                    [
                        report["macro_f1"]["mean"]
                        for report in population_reports.values()
                    ]
                )
            ),
            "decision_level": f"{bag_size}-image mean-logit population",
            "pooling": "unweighted macro mean over formal sources",
        }
    else:
        by_edit: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            by_edit[str(row["sample_id"])].append(index)
        incomplete: list[tuple[str, int]] = []
        excluded_sources: dict[str, int] = defaultdict(int)
        for edit_id in sorted(by_edit):
            indices = sorted(
                by_edit[edit_id],
                key=lambda index: rows[index]["generated_id"],
            )
            first = rows[indices[0]]
            source = first["_target"]["state_group"]
            if source not in FORMAL_STATE_SOURCES:
                excluded_sources[source] += 1
                continue
            if len(indices) != bag_size:
                incomplete.append((edit_id, len(indices)))
                continue
            mean_identity = identity_logits[indices].mean(dim=0, keepdim=True)
            mean_background = background_logits[indices].mean(dim=0)
            mean_state = torch.stack(
                [state_logits_by_index[index] for index in indices]
            ).mean(dim=0)
            predictions = {
                "identity": int(
                    project_identity_predictions(
                        mean_identity,
                        identity_groups,
                    )[0]
                ),
                "state": int(mean_state.argmax()),
                "background": int(mean_background.argmax()),
            }
            targets = {
                "identity": identity_targets[indices[0]],
                "state": state_targets[indices[0]],
                "background": background_targets[indices[0]],
            }
            matches = {
                attribute: predictions[attribute] == targets[attribute]
                for attribute in ["identity", "state", "background"]
            }
            target_attribute = first["_target_attribute"]
            preserved = [
                attribute
                for attribute in matches
                if attribute != target_attribute
            ]
            target_success = matches[target_attribute]
            preservation_success = all(matches[key] for key in preserved)
            acsr_conditions.append(
                {
                    "edit_id": edit_id,
                    "anchor_sample_id": first["_anchor_sample_id"],
                    "source": source,
                    "edit_scope": first["_edit_scope"],
                    "target_attribute": target_attribute,
                    "num_generated": len(indices),
                    "member_generated_ids": [
                        rows[index]["generated_id"] for index in indices
                    ],
                    "target_labels": {
                        "identity": identity_labels[targets["identity"]],
                        "state": vocab["state"][source]["id_to_label"][
                            targets["state"]
                        ],
                        "background": vocab["background"]["id_to_label"][
                            targets["background"]
                        ],
                    },
                    "predicted_labels": {
                        "identity": identity_labels[predictions["identity"]],
                        "state": vocab["state"][source]["id_to_label"][
                            predictions["state"]
                        ],
                        "background": vocab["background"]["id_to_label"][
                            predictions["background"]
                        ],
                    },
                    "attribute_matches": matches,
                    "target_success": target_success,
                    "preservation_success": preservation_success,
                    "acsr_success": target_success and preservation_success,
                }
            )
        if incomplete and not args.allow_incomplete_edit_groups:
            first_id, first_count = incomplete[0]
            raise ValueError(
                f"{len(incomplete)} eligible edit conditions do not contain "
                f"exactly {bag_size} images; first={first_id}:{first_count}"
            )
        direct["ACSR"] = {
            **acsr_summary(acsr_conditions),
            "samples_per_condition": bag_size,
            "eligible_sources": list(FORMAL_STATE_SOURCES),
            "excluded_condition_counts_by_source": dict(
                sorted(excluded_sources.items())
            ),
            "skipped_incomplete_conditions": [
                {"edit_id": edit_id, "count": count}
                for edit_id, count in incomplete
            ],
        }

    summary = {
        "schema": "exp4.control_metrics.generated_report.final.v1",
        "status": "pass",
        "mode": mode,
        "generated_manifest": str(generated_manifest),
        "target_manifest": target_manifest,
        "final_protocol": str(final_protocol_path),
        "shared_evaluator": str(shared_evaluator_dir / "checkpoint_best.pt"),
        "bbbc_state_evaluator": str(
            bbbc_evaluator_dir / "checkpoint_best.pt"
        ),
        "n_generated": len(rows),
        "formal_protocol": {
            "ICA": "single-image 12-class entity+morphology",
            "SCA": f"{bag_size}-image mean-logit BBBC021/DynamicNuclearNet",
            "BCA": "single-image five-domain",
            "ACSR": f"{bag_size}-image mean-logit edit condition",
        },
        "metrics": direct,
    }
    write_jsonl(output_dir / "per_generated_predictions.jsonl", per_image)
    if population_records:
        write_jsonl(output_dir / "population_sca_bags.jsonl", population_records)
    if mode == "edits":
        write_jsonl(output_dir / "acsr_conditions.jsonl", acsr_conditions)
    write_json(output_dir / "summary.json", summary)

    markdown = f"""# Final controllable-generation metrics

- Mode: `{mode}`
- Generated images: {len(rows)}
- ICA protocol: 12-class entity + morphology
- SCA protocol: {bag_size}-image mean-logit population, BBBC021 and DynamicNuclearNet
- BCA protocol: five acquisition/background domains

| Metric | Accuracy | Macro-F1 |
|---|---:|---:|
| ICA | {direct['ICA']['accuracy']:.4f} | {direct['ICA']['macro_f1']:.4f} |
| BCA | {direct['BCA']['accuracy']:.4f} | {direct['BCA']['macro_f1']:.4f} |
"""
    if mode == "paired":
        markdown += (
            f"| SCA | {direct['SCA']['accuracy']:.4f} | "
            f"{direct['SCA']['macro_f1']:.4f} |\n\n"
        )
        markdown += "| SCA source | Bags | Accuracy | Macro-F1 |\n"
        markdown += "|---|---:|---:|---:|\n"
        for source, report in population_reports.items():
            markdown += (
                f"| {source} | {report['bags_per_bootstrap']} | "
                f"{report['accuracy']['mean']:.4f} | "
                f"{report['macro_f1']['mean']:.4f} |\n"
            )
    else:
        acsr = direct["ACSR"]["by_attribute"]
        markdown += "\n| Edit | Target success | Preservation | ACSR |\n"
        markdown += "|---|---:|---:|---:|\n"
        for attribute in ["identity", "state", "background"]:
            report = acsr[attribute]
            markdown += (
                f"| {attribute} | "
                f"{format_metric(report['target_success_rate'])} | "
                f"{format_metric(report['preservation_rate'])} | "
                f"{format_metric(report['acsr'])} |\n"
            )
        markdown += (
            "\nBackground edits are compositional OOD and are not pooled with "
            "in-domain identity/state edits.\n"
        )
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(f"saved {output_dir / 'summary.json'}")
    print(f"saved {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
