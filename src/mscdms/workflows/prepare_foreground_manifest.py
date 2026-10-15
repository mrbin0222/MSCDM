#!/usr/bin/env python3
"""Prepare foreground manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


FORMAL_SOURCES = ("bbbc021", "dynamicnuclearnet")
MORPHOLOGIES = ("elongated", "oval", "round")
UNKNOWN_STATE = "__unknown__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_manifest", type=Path, default=Path("data/processed/foreground_samples.jsonl"))
    parser.add_argument("--evaluator_manifest", type=Path, default=Path("data/processed/control_metrics_v2/evaluator_manifest.jsonl"))
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--audit_json", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_key(source: str, state_label: str | None) -> str:
    return f"{source}::{state_label if state_label is not None else UNKNOWN_STATE}"


def prepare_records(
    training_rows: list[dict[str, Any]],
    evaluator_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluator = {
        str(row["sample_id"]): row
        for row in evaluator_rows
        if row.get("source_name") in FORMAL_SOURCES
        and row.get("split") in {"train", "val"}
    }
    source_to_id = {source: index for index, source in enumerate(FORMAL_SOURCES)}
    state_keys = sorted(
        {
            state_key(str(row["source_name"]), row.get("state_label"))
            for row in evaluator.values()
            if row.get("split") == "train"
        }
    )
    state_to_id = {value: index for index, value in enumerate(state_keys)}
    records: list[dict[str, Any]] = []
    for row in training_rows:
        if row.get("source_name") not in FORMAL_SOURCES:
            continue
        if row.get("split") not in {"train", "val"}:
            continue
        sample_id = str(row["sample_id"])
        evaluation = evaluator.get(sample_id)
        if evaluation is None:
            raise ValueError(f"missing evaluator record: {sample_id}")
        source = str(row["source_name"])
        morphology = str(row.get("attributes", {}).get("morphology") or "").lower()
        if morphology not in MORPHOLOGIES:
            raise ValueError(f"invalid morphology for {sample_id}: {morphology!r}")
        key = state_key(source, evaluation.get("state_label"))
        if key not in state_to_id:
            raise ValueError(f"validation state is absent from train vocabulary: {key}")
        item = dict(row)
        item["foreground_renderer"] = {
            "schema": "exp4.foreground_renderer.stage26.record.v1",
            "source_id": source_to_id[source],
            "state_id": state_to_id[key],
            "state_key": key,
            "state_label": evaluation.get("state_label"),
            "state_formally_labeled": evaluation.get("state_label") is not None,
            "morphology": morphology,
            "paired_counterfactual_target": False,
        }
        records.append(item)
    records.sort(key=lambda row: (str(row["split"]), str(row["sample_id"])))
    split_counts = Counter(str(row["split"]) for row in records)
    expected = {"train": 10000, "val": 1000}
    if dict(split_counts) != expected:
        raise ValueError(f"unexpected split counts: {dict(split_counts)} != {expected}")
    if any(row.get("split") == "test" for row in records):
        raise ValueError("test records entered the foreground-renderer manifest")
    audit = {
        "schema": "exp4.foreground_renderer.stage26.manifest_audit.v1",
        "status": "pass",
        "split_counts": dict(split_counts),
        "source_split_counts": dict(
            Counter(f"{row['source_name']}:{row['split']}" for row in records)
        ),
        "morphology_split_counts": dict(
            Counter(
                f"{row['split']}:{row['foreground_renderer']['morphology']}"
                for row in records
            )
        ),
        "formal_state_split_counts": dict(
            Counter(
                f"{row['split']}:{row['source_name']}:"
                f"{'labeled' if row['foreground_renderer']['state_formally_labeled'] else 'unknown'}"
                for row in records
            )
        ),
        "source_to_id": source_to_id,
        "state_to_id": state_to_id,
        "state_vocabulary_fitted_on": "train_only",
        "paired_counterfactual_targets_used": False,
        "test_accessed": False,
    }
    return records, audit


def main() -> None:
    args = parse_args()
    records, audit = prepare_records(
        read_jsonl(args.training_manifest),
        read_jsonl(args.evaluator_manifest),
    )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit["inputs"] = {
        "training_manifest": str(args.training_manifest.resolve()),
        "training_manifest_sha256": sha256_file(args.training_manifest),
        "evaluator_manifest": str(args.evaluator_manifest.resolve()),
        "evaluator_manifest_sha256": sha256_file(args.evaluator_manifest),
    }
    audit["output_manifest"] = str(args.output_manifest.resolve())
    audit["output_manifest_sha256"] = sha256_file(args.output_manifest)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("saved", args.output_manifest)
    print("saved", args.audit_json)


if __name__ == "__main__":
    main()
