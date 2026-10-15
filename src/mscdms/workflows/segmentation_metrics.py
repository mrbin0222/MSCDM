#!/usr/bin/env python3
"""Segmentation metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.optimize import linear_sum_assignment


THRESHOLDS = tuple(round(0.5 + i * 0.05, 2) for i in range(10))


def overlap_iou(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    # Explicitly reserve background even if neither map contains a zero pixel.
    pids = np.unique(pred)
    gids = np.unique(truth)
    p = np.searchsorted(np.r_[0, pids[pids > 0]], pred.ravel())
    g = np.searchsorted(np.r_[0, gids[gids > 0]], truth.ravel())
    n, m = len(pids[pids > 0]) + 1, len(gids[gids > 0]) + 1
    table = np.bincount(p * m + g, minlength=n * m).reshape(n, m)
    inter = table[1:, 1:].astype(float)
    union = table.sum(1)[1:, None] + table.sum(0)[None, 1:] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_counts(matrix: np.ndarray, threshold: float) -> tuple[int, int, int]:
    n, m = matrix.shape
    tp = 0
    if n and m:
        eligible = matrix > threshold
        # Cardinality bonus dominates any possible sum of IoUs.
        r, c = linear_sum_assignment(-(eligible * (min(n, m) + 1) + matrix))
        tp = int(eligible[r, c].sum())
    return tp, n - tp, m - tp


def frame_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError('native full-frame 2D prediction and ST label shapes must match')
    for a in (pred, truth):
        if not np.issubdtype(a.dtype, np.integer) or a.min(initial=0) < 0:
            raise ValueError('nonnegative integer instance labels required')
    matrix = overlap_iou(pred, truth)
    scores, counts = {}, {}
    for t in THRESHOLDS:
        tp, fp, fn = match_counts(matrix, t)
        scores[f'{t:.2f}'] = tp / (tp + fp + fn) if tp + fp + fn else 1.0
        counts[f'{t:.2f}'] = {'tp': tp, 'fp': fp, 'fn': fn}
    pf, gf = pred > 0, truth > 0
    denom = int(pf.sum() + gf.sum())
    return {'AP50': scores['0.50'], 'AP75': scores['0.75'],
            'mAP_inst': float(np.mean(list(scores.values()))),
            'Dice_fg': 2 * int((pf & gf).sum()) / denom if denom else 1.0,
            'AP_by_threshold': scores, 'counts_by_threshold': counts,
            'pred_instances': matrix.shape[0], 'target_instances': matrix.shape[1]}


def evaluate_manifest(predictions: Path, test_pairs: Path, output: Path) -> dict:
    truth = {r['frame_id']: r for r in (json.loads(s) for s in test_pairs.read_text().splitlines())}
    pred_rows = [json.loads(s) for s in predictions.read_text().splitlines()]
    if len(pred_rows) != len(truth) or {r['frame_id'] for r in pred_rows} != set(truth):
        raise ValueError('prediction manifest must cover the entire fixed test sequence exactly once')
    details = []
    for r in pred_rows:
        path = Path(r['pred_instance_path'])
        path = path if path.is_absolute() else predictions.parent / path
        g = np.asarray(tifffile.imread(truth[r['frame_id']]['mask_path']))
        p = np.asarray(tifffile.imread(path))
        details.append({'frame_id': r['frame_id'], **frame_metrics(p, g)})
    summary = {k: float(np.mean([r[k] for r in details])) for k in ('AP50', 'AP75', 'mAP_inst', 'Dice_fg')}
    summary.update({'frames': len(details), 'reference': 'official CTC ST silver',
                    'aggregation': 'arithmetic frame macro mean', 'AP_is_COCO_ranked_AP': False,
                    'match_rule': 'one-to-one cardinality-first instance IoU strictly > threshold',
                    'thresholds': THRESHOLDS})
    output.mkdir(parents=True, exist_ok=False)
    (output / 'frame_metrics.json').write_text(json.dumps(details, indent=2) + '\n')
    (output / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--test-pairs', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(evaluate_manifest(a.predictions, a.test_pairs, a.out_dir), indent=2), flush=True)


if __name__ == '__main__':
    main()
