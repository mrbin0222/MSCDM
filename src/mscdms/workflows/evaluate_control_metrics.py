#!/usr/bin/env python3
"""Evaluate control metrics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mscdms.control_evaluator import (
    ControlProbe,
    classification_metrics,
    extract_clip_features,
    load_clip_image_encoder,
    majority_vote,
    read_jsonl,
    resolve_image_path,
    sha256_file,
    write_json,
    write_jsonl,
)


STATE_GROUPS = [
    "bbbc021",
    "idr0033",
    "tissuenet_wholecell",
    "tissuenet_nuclear",
    "dynamicnuclearnet",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp4_root", type=Path, default=root)
    ap.add_argument("--generated_manifest", type=Path, required=True)
    ap.add_argument(
        "--protocol_dir",
        type=Path,
        default=Path(
            "data/processed/control_metrics_v2"
        ),
    )
    ap.add_argument(
        "--evaluator_dir",
        type=Path,
        default=Path("outputs/v4_control_evaluators_finetuned_v2"),
    )
    ap.add_argument(
        "--clip_model_dir",
        type=Path,
        default=Path("models/clip-vit-large-patch14"),
    )
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["auto", "paired", "edits"], default="auto")
    ap.add_argument(
        "--control_edits",
        type=Path,
        help="Required for edit mode unless IDs uniquely match val or test edits",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--allow_failed_evaluator_gate", action="store_true")
    return ap.parse_args()


def from_root(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_control_edits(
    requested: Path | None,
    *,
    root: Path,
    protocol_dir: Path,
    sample_ids: set[str],
) -> tuple[Path, dict[str, dict[str, Any]]]:
    if requested is not None:
        candidates = [from_root(requested, root)]
    else:
        candidates = [
            protocol_dir / "val_control_edits.jsonl",
            protocol_dir / "test_control_edits.jsonl",
        ]
    matches: list[tuple[Path, dict[str, dict[str, Any]]]] = []
    for path in candidates:
        rows = read_jsonl(path)
        by_id = {row["edit_id"]: row for row in rows}
        if sample_ids <= set(by_id):
            matches.append((path, by_id))
    if len(matches) != 1:
        raise ValueError(
            "could not uniquely resolve control edit manifest; "
            "pass --control_edits explicitly"
        )
    return matches[0]


def attach_targets(
    generated: list[dict[str, Any]],
    *,
    mode: str,
    protocol_dir: Path,
    root: Path,
    control_edits: Path | None,
) -> tuple[str, list[dict[str, Any]], str]:
    sample_ids = {str(row["sample_id"]) for row in generated}
    evaluator_rows = read_jsonl(protocol_dir / "evaluator_manifest.jsonl")
    paired_map = {row["sample_id"]: row for row in evaluator_rows}
    paired_possible = sample_ids <= set(paired_map)
    edit_path: Path | None = None
    edit_map: dict[str, dict[str, Any]] | None = None
    try:
        edit_path, edit_map = resolve_control_edits(
            control_edits,
            root=root,
            protocol_dir=protocol_dir,
            sample_ids=sample_ids,
        )
        edits_possible = True
    except ValueError:
        edits_possible = False
    if mode == "auto":
        if paired_possible == edits_possible:
            raise ValueError(
                "auto mode is ambiguous or unmatched; pass --mode paired|edits"
            )
        mode = "paired" if paired_possible else "edits"
    if mode == "paired":
        if not paired_possible:
            missing = sorted(sample_ids - set(paired_map))
            raise ValueError(
                f"{len(missing)} sample IDs do not match evaluator manifest; "
                f"first={missing[0]}"
            )
        attached = []
        for row in generated:
            target = paired_map[str(row["sample_id"])]
            attached.append(
                {
                    **row,
                    "_target": {
                        "identity": target["identity_label"],
                        "state": target["state_label"],
                        "state_group": target["state_group"],
                        "background": target["background_label"],
                    },
                }
            )
        return mode, attached, str(protocol_dir / "evaluator_manifest.jsonl")
    if not edits_possible or edit_path is None or edit_map is None:
        raise ValueError("edit mode requires matching --control_edits")
    attached = []
    for row in generated:
        edit = edit_map[str(row["sample_id"])]
        attached.append(
            {
                **row,
                "_target": edit["target_labels"],
                "_original": edit["original_labels"],
                "_target_attribute": edit["target_attribute"],
                "_edit_scope": edit["edit_scope"],
                "_anchor_sample_id": edit["anchor_sample_id"],
            }
        )
    return mode, attached, str(edit_path)


def decode_predictions(
    rows: list[dict[str, Any]],
    features: torch.Tensor,
    *,
    model: ControlProbe,
    vocab: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    model.eval()
    decoded: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            stop = min(len(rows), start + batch_size)
            batch = features[start:stop].to(device=device, dtype=torch.float32)
            identity_logits, background_logits, state_logits = model(batch)
            identity_ids = identity_logits.argmax(dim=1).cpu().tolist()
            background_ids = background_logits.argmax(dim=1).cpu().tolist()
            for offset, row in enumerate(rows[start:stop]):
                state_group = row["_target"]["state_group"]
                state_id = int(state_logits[state_group][offset].argmax().cpu())
                prediction = {
                    "identity": vocab["identity"]["id_to_label"][
                        identity_ids[offset]
                    ],
                    "state": vocab["state"][state_group]["id_to_label"][state_id],
                    "state_group": state_group,
                    "background": vocab["background"]["id_to_label"][
                        background_ids[offset]
                    ],
                }
                decoded.append(prediction)
    return decoded


def label_ids(
    labels: dict[str, str],
    vocab: dict[str, Any],
) -> tuple[int, int, int]:
    group = labels["state_group"]
    return (
        int(vocab["identity"]["label_to_id"][labels["identity"]]),
        int(vocab["state"][group]["label_to_id"][labels["state"]]),
        int(vocab["background"]["label_to_id"][labels["background"]]),
    )


def direct_metrics(
    rows: list[dict[str, Any]],
    vocab: dict[str, Any],
) -> dict[str, Any]:
    identity_target: list[int] = []
    identity_prediction: list[int] = []
    background_target: list[int] = []
    background_prediction: list[int] = []
    state_target: list[int] = []
    state_prediction: list[int] = []
    state_group: list[str] = []
    for row in rows:
        target = row["target_labels"]
        prediction = row["predicted_labels"]
        target_ids = label_ids(target, vocab)
        prediction_ids = label_ids(prediction, vocab)
        identity_target.append(target_ids[0])
        identity_prediction.append(prediction_ids[0])
        state_target.append(target_ids[1])
        state_prediction.append(prediction_ids[1])
        state_group.append(target["state_group"])
        background_target.append(target_ids[2])
        background_prediction.append(prediction_ids[2])
    identity = classification_metrics(
        identity_target,
        identity_prediction,
        num_classes=len(vocab["identity"]["id_to_label"]),
    )
    background = classification_metrics(
        background_target,
        background_prediction,
        num_classes=len(vocab["background"]["id_to_label"]),
    )
    state_by_group: dict[str, Any] = {}
    combined_target: list[int] = []
    combined_prediction: list[int] = []
    offset = 0
    for group in STATE_GROUPS:
        indices = [index for index, value in enumerate(state_group) if value == group]
        group_target = [state_target[index] for index in indices]
        group_prediction = [state_prediction[index] for index in indices]
        count = len(vocab["state"][group]["id_to_label"])
        state_by_group[group] = classification_metrics(
            group_target,
            group_prediction,
            num_classes=count,
        )
        combined_target.extend(value + offset for value in group_target)
        combined_prediction.extend(value + offset for value in group_prediction)
        offset += count
    state = classification_metrics(
        combined_target,
        combined_prediction,
        num_classes=offset,
    )
    state["by_group"] = state_by_group
    return {"ICA": identity, "SCA": state, "BCA": background}


def add_success_flags(row: dict[str, Any]) -> None:
    target = row["target_labels"]
    prediction = row["predicted_labels"]
    matches = {
        key: prediction[key] == target[key]
        for key in ["identity", "state", "background"]
    }
    target_attribute = row["target_attribute"]
    preserved_attributes = [
        key for key in matches if key != target_attribute
    ]
    row["attribute_matches"] = matches
    row["target_success"] = bool(matches[target_attribute])
    row["preservation_success"] = bool(
        all(matches[key] for key in preserved_attributes)
    )
    row["acsr_success"] = bool(
        row["target_success"] and row["preservation_success"]
    )


def rate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "target_success_rate": None,
            "preservation_rate": None,
            "acsr": None,
        }
    return {
        "n": len(rows),
        "target_success_rate": float(
            np.mean([row["target_success"] for row in rows])
        ),
        "preservation_rate": float(
            np.mean([row["preservation_success"] for row in rows])
        ),
        "acsr": float(np.mean([row["acsr_success"] for row in rows])),
    }


def format_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def acsr_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_attribute = {
        attribute: rate_summary(
            [row for row in rows if row["target_attribute"] == attribute]
        )
        for attribute in ["identity", "state", "background"]
    }
    by_repeat = {
        str(repeat): {
            attribute: rate_summary(
                [
                    row
                    for row in rows
                    if int(row["repeat"]) == repeat
                    and row["target_attribute"] == attribute
                ]
            )
            for attribute in ["identity", "state", "background"]
        }
        for repeat in sorted({int(row["repeat"]) for row in rows})
    }
    in_domain = [
        row
        for row in rows
        if row["target_attribute"] in {"identity", "state"}
    ]
    background_ood = [
        row for row in rows if row["target_attribute"] == "background"
    ]
    return {
        "by_attribute": by_attribute,
        "by_repeat": by_repeat,
        "in_domain_subject_state_pooled": rate_summary(in_domain),
        "background_compositional_ood": rate_summary(background_ood),
        "reporting_policy": (
            "background ACSR is compositional OOD and is not pooled into the "
            "in-domain subject/state ACSR"
        ),
    }


def majority_rows(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["edit_id"]
            if mode == "edits"
            else str(row["sample_id"])
        )
        grouped[key].append(row)
    result: list[dict[str, Any]] = []
    for key in sorted(grouped):
        members = grouped[key]
        first = members[0]
        prediction = {
            attribute: majority_vote(
                member["predicted_labels"][attribute] for member in members
            )
            for attribute in ["identity", "state", "background"]
        }
        prediction["state_group"] = first["target_labels"]["state_group"]
        aggregate = {
            "sample_id": first["sample_id"],
            "generated_id": f"{key}__majority",
            "repeat": -1,
            "num_repeats": len(members),
            "target_labels": first["target_labels"],
            "predicted_labels": prediction,
        }
        if mode == "edits":
            aggregate.update(
                {
                    "edit_id": first["edit_id"],
                    "anchor_sample_id": first["anchor_sample_id"],
                    "target_attribute": first["target_attribute"],
                    "edit_scope": first["edit_scope"],
                    "original_labels": first["original_labels"],
                }
            )
            add_success_flags(aggregate)
        result.append(aggregate)
    return result


def main() -> None:
    args = parse_args()
    root = args.exp4_root.resolve()
    generated_manifest = from_root(args.generated_manifest, root)
    protocol_dir = from_root(args.protocol_dir, root)
    evaluator_dir = from_root(args.evaluator_dir, root)
    clip_model_dir = from_root(args.clip_model_dir, root)
    output_dir = from_root(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)

    gate_report = load_json(evaluator_dir / "real_test_report.json")
    if (
        gate_report["status"] != "pass"
        and not args.allow_failed_evaluator_gate
    ):
        raise RuntimeError(
            "real-image evaluator gate did not pass; generated ICA/SCA/BCA/ACSR "
            "would be unreliable. Pass --allow_failed_evaluator_gate only for "
            "diagnostic scoring."
        )
    vocab_path = protocol_dir / "label_vocab.json"
    vocab = load_json(vocab_path)
    checkpoint = torch.load(
        evaluator_dir / "checkpoint_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint["vocab_sha256"] != sha256_file(vocab_path):
        raise ValueError("evaluator checkpoint and protocol vocabulary differ")
    model = ControlProbe.from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = model.to(device).eval()

    generated = read_jsonl(generated_manifest)
    if not generated:
        raise ValueError(f"{generated_manifest} is empty")
    generated_ids = [str(row["generated_id"]) for row in generated]
    if len(generated_ids) != len(set(generated_ids)):
        raise ValueError("generated_manifest contains duplicate generated_id values")
    mode, attached, target_manifest = attach_targets(
        generated,
        mode=args.mode,
        protocol_dir=protocol_dir,
        root=root,
        control_edits=args.control_edits,
    )
    image_root = generated_manifest.parent
    paths = [
        resolve_image_path(str(row["generated_image"]), image_root)
        for row in attached
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} generated images are missing; first={missing[0]}"
        )
    processor, encoder = load_clip_image_encoder(
        clip_model_dir,
        device,
        parameter_delta=checkpoint.get("vision_parameter_delta"),
    )
    features = extract_clip_features(
        paths,
        processor=processor,
        model=encoder,
        device=device,
        batch_size=args.batch_size,
    )
    predictions = decode_predictions(
        attached,
        features,
        model=model,
        vocab=vocab,
        device=device,
        batch_size=args.batch_size,
    )

    scored: list[dict[str, Any]] = []
    for source, prediction in zip(attached, predictions):
        row = {
            "generated_id": source["generated_id"],
            "sample_id": source["sample_id"],
            "repeat": int(source.get("repeat", 0)),
            "generated_image": source["generated_image"],
            "target_labels": source["_target"],
            "predicted_labels": prediction,
        }
        if mode == "edits":
            row.update(
                {
                    "edit_id": source["sample_id"],
                    "anchor_sample_id": source["_anchor_sample_id"],
                    "target_attribute": source["_target_attribute"],
                    "edit_scope": source["_edit_scope"],
                    "original_labels": source["_original"],
                }
            )
            add_success_flags(row)
        scored.append(row)
    majority = majority_rows(scored, mode)
    summary: dict[str, Any] = {
        "schema": "exp4.control_metrics.generated_report.v1",
        "status": "pass",
        "mode": mode,
        "generated_manifest": str(generated_manifest),
        "target_manifest": target_manifest,
        "evaluator_checkpoint": str(evaluator_dir / "checkpoint_best.pt"),
        "evaluator_real_test_gate": gate_report["status"],
        "n_generated": len(scored),
        "n_unique_conditions": len(majority),
        "repeat_level": {
            "direct_metrics": direct_metrics(scored, vocab),
        },
        "condition_majority_vote": {
            "direct_metrics": direct_metrics(majority, vocab),
        },
    }
    if mode == "edits":
        summary["repeat_level"]["ACSR"] = acsr_metrics(scored)
        summary["condition_majority_vote"]["ACSR"] = acsr_metrics(majority)
    write_jsonl(output_dir / "per_generated_predictions.jsonl", scored)
    write_jsonl(output_dir / "per_condition_majority.jsonl", majority)
    write_json(output_dir / "summary.json", summary)

    direct = summary["repeat_level"]["direct_metrics"]
    markdown = f"""# Generated ICA/SCA/BCA/ACSR report

- Mode: `{mode}`
- Generated images: {len(scored)}
- Unique conditions: {len(majority)}
- Evaluator real-test gate: **{gate_report['status'].upper()}**

| Metric | Repeat-level accuracy | Repeat-level Macro-F1 |
|---|---:|---:|
| ICA | {direct['ICA']['accuracy']:.4f} | {direct['ICA']['macro_f1']:.4f} |
| SCA | {direct['SCA']['accuracy']:.4f} | {direct['SCA']['macro_f1']:.4f} |
| BCA | {direct['BCA']['accuracy']:.4f} | {direct['BCA']['macro_f1']:.4f} |
"""
    if mode == "edits":
        acsr = summary["repeat_level"]["ACSR"]
        markdown += f"""
| ACSR scope | Target success | Preservation | ACSR |
|---|---:|---:|---:|
| Identity edit (in-domain) | {format_rate(acsr['by_attribute']['identity']['target_success_rate'])} | {format_rate(acsr['by_attribute']['identity']['preservation_rate'])} | {format_rate(acsr['by_attribute']['identity']['acsr'])} |
| State edit (in-domain) | {format_rate(acsr['by_attribute']['state']['target_success_rate'])} | {format_rate(acsr['by_attribute']['state']['preservation_rate'])} | {format_rate(acsr['by_attribute']['state']['acsr'])} |
| Background edit (compositional OOD) | {format_rate(acsr['by_attribute']['background']['target_success_rate'])} | {format_rate(acsr['by_attribute']['background']['preservation_rate'])} | {format_rate(acsr['by_attribute']['background']['acsr'])} |

Background-edit ACSR is reported separately and is not pooled into the
in-domain subject/state ACSR.
"""
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(f"saved {output_dir / 'summary.json'}")
    print(f"saved {output_dir / 'summary.md'}")


