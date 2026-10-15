from __future__ import annotations
from pathlib import Path
from typing import Any,Iterable
import json,numpy as np,tifffile

CTC_DATASETS = ('DIC-C2DH-HeLa', 'Fluo-N2DL-HeLa', 'Fluo-N2DH-GOWT1', 'Fluo-N3DH-CHO', 'Fluo-N2DH-SIM+', 'PhC-C2DH-U373', 'PhC-C2DL-PSC')

def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')

def save_label_tiff(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_label = int(labels.max(initial=0))
    dtype = np.uint16 if max_label <= np.iinfo(np.uint16).max else np.uint32
    tifffile.imwrite(path, labels.astype(dtype, copy=False), compression='zlib')
