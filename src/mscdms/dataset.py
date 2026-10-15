from __future__ import annotations

import json
import random
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .cell_text import triplet_from_txt_path
from .text_encoder import MultiScaleTextEncoder


def re_match_id(name: str) -> str | None:
    m = re.search(r"cell_text_(\d+)\.txt$", name, re.IGNORECASE)
    return m.group(1) if m else None


def discover_image_text_pairs(root: str | Path) -> list[tuple[Path, Path]]:
    """
    在 root 下递归查找 cell_text_*.txt，并配对同索引的
    ../image/<channel>/cell_<id>.png（与当前 demo 目录结构一致）。
    """
    root = Path(root).resolve()
    pairs: list[tuple[Path, Path]] = []
    for text_path in root.rglob("cell_text_*.txt"):
        m = re_match_id(text_path.name)
        if not m:
            continue
        idx = m
        channel = text_path.parent.name
        cell_dir = text_path.parent.parent.parent
        image_path = cell_dir / "image" / channel / f"cell_{idx}.png"
        if image_path.is_file():
            pairs.append((image_path, text_path))
    pairs.sort(key=lambda x: (str(x[0]), str(x[1])))
    return pairs


def discover_manifest_records(
    root: str | Path,
    split: str,
    manifest_name: str = "manifest.jsonl",
) -> list[dict]:
    """读取 exp4 实例裁剪数据的显式划分及三路文本条件。"""
    root = Path(root).resolve()
    path = root / manifest_name
    if not path.is_file():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("split") != split:
                continue
            required = ("image_path", "subject", "state", "background")
            missing = [key for key in required if not str(rec.get(key, "")).strip()]
            if missing:
                raise ValueError(f"{path}:{line_no} 缺少字段: {', '.join(missing)}")
            image_path = Path(rec["image_path"])
            if not image_path.is_absolute():
                image_path = root / image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"{path}:{line_no} 指向的图像不存在: {image_path}")
            normalized = dict(rec)
            normalized["_resolved_image_path"] = str(image_path)
            records.append(normalized)
    records.sort(key=lambda item: str(item.get("sample_id", item["image_path"])))
    return records


def discover_dataset_size(root: str | Path, manifest_name: str = "manifest.jsonl") -> int:
    root = Path(root)
    path = root / manifest_name
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    return len(discover_image_text_pairs(root))


def split_indices(n: int, seed: int, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> tuple[list[int], list[int], list[int]]:
    """8:1:1 固定随机划分。"""
    assert abs(sum(ratios) - 1.0) < 1e-6
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    train_i = idx[:n_tr]
    val_i = idx[n_tr : n_tr + n_va]
    test_i = idx[n_tr + n_va :]
    return train_i, val_i, test_i


class CellMicroscopyDataset(Dataset):
    """
    单细胞 512 训练样本：读取配对图像与 txt，三路文本由 cell_text 模块生成；
    可选强度分位数拉伸（多来源亮度对齐）。
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        seed: int = 42,
        image_size: int = 512,
        normalize_mode: str = "percentile",
        p_lo: float = 1.0,
        p_hi: float = 99.0,
        manifest_name: str = "manifest.jsonl",
        return_target_mask: bool = False,
        return_object_type: bool = False,
        return_control_negatives: bool = False,
        return_morphology_mask_target: bool = False,
        return_morphology_class: bool = False,
        return_morphology_counterfactual_class: bool = False,
    ):
        super().__init__()
        assert split in ("train", "val", "test")
        assert normalize_mode in ("percentile", "minmax", "none")
        self.root = Path(root)
        self.image_size = image_size
        self.normalize_mode = normalize_mode
        self.p_lo = p_lo
        self.p_hi = p_hi
        self.return_target_mask = bool(return_target_mask)
        self.return_object_type = bool(return_object_type)
        self.return_control_negatives = bool(return_control_negatives)
        self.return_morphology_mask_target = bool(
            return_morphology_mask_target
        )
        self.return_morphology_class = bool(return_morphology_class)
        self.return_morphology_counterfactual_class = bool(
            return_morphology_counterfactual_class
        )

        self.uses_manifest = (self.root / manifest_name).is_file()
        self.records = discover_manifest_records(self.root, split, manifest_name)
        self.pairs: list[tuple[Path, Path]] = []
        if (
            self.return_object_type
            or self.return_target_mask
            or self.return_control_negatives
            or self.return_morphology_mask_target
            or self.return_morphology_class
            or self.return_morphology_counterfactual_class
        ) and not self.uses_manifest:
            raise ValueError(
                "object_type、target_mask 和 control_negatives "
                "只支持含 manifest 的数据集"
            )
        if self.return_control_negatives:
            for record in self.records:
                negatives = record.get("control_negatives")
                enabled = record.get("control_rank_enabled")
                if not isinstance(negatives, dict) or not isinstance(
                    enabled, dict
                ):
                    raise ValueError(
                        f"{record.get('sample_id')} 缺少控制反事实字段"
                    )
                for factor in ("subject", "state", "background"):
                    if not str(negatives.get(factor) or "").strip():
                        raise ValueError(
                            f"{record.get('sample_id')} 缺少 "
                            f"control_negatives.{factor}"
                        )
                    if not isinstance(enabled.get(factor), bool):
                        raise ValueError(
                            f"{record.get('sample_id')} 缺少布尔字段 "
                            f"control_rank_enabled.{factor}"
                        )
        if self.return_morphology_mask_target:
            for record in self.records:
                payload = record.get("morphology_mask_surrogate")
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"{record.get('sample_id')} 缺少 "
                        "morphology_mask_surrogate"
                    )
                enabled = bool(payload.get("enabled", False))
                source_id = int(payload.get("source_id", -1))
                if enabled and source_id not in {0, 1}:
                    raise ValueError(
                        f"{record.get('sample_id')} 的 mask surrogate "
                        f"source_id 无效: {source_id}"
                    )
                if not enabled:
                    record["_resolved_morphology_mask_path"] = ""
                    continue
                raw_path = str(payload.get("target_mask_path") or "").strip()
                if not raw_path:
                    raise ValueError(
                        f"{record.get('sample_id')} 缺少反事实 morphology mask"
                    )
                target_path = Path(raw_path)
                if not target_path.is_absolute():
                    target_path = self.root / target_path
                if not target_path.is_file():
                    raise FileNotFoundError(target_path)
                record["_resolved_morphology_mask_path"] = str(target_path)
        if self.return_morphology_class:
            for record in self.records:
                morphology = str(record.get("morphology_class") or "").strip()
                if morphology not in {"round", "oval", "elongated"}:
                    raise ValueError(
                        f"{record.get('sample_id')} morphology_class invalid: "
                        f"{morphology!r}"
                    )
        if self.return_morphology_counterfactual_class:
            for record in self.records:
                morphology = str(
                    record.get("morphology_counterfactual_class") or ""
                ).strip()
                if morphology not in {"round", "oval", "elongated"}:
                    raise ValueError(
                        f"{record.get('sample_id')} counterfactual morphology "
                        f"invalid: {morphology!r}"
                    )
        if self.return_target_mask:
            for record in self.records:
                raw_mask_path = str(record.get("target_mask_path") or "").strip()
                if not raw_mask_path:
                    raise ValueError(
                        f"{record.get('sample_id')} 缺少 target_mask_path"
                    )
                mask_path = Path(raw_mask_path)
                if not mask_path.is_absolute():
                    mask_path = self.root / mask_path
                if not mask_path.is_file():
                    raise FileNotFoundError(
                        f"{record.get('sample_id')} 的目标掩膜不存在: {mask_path}"
                    )
                record["_resolved_target_mask_path"] = str(mask_path)
        if not self.uses_manifest:
            pairs = discover_image_text_pairs(self.root)
            if not pairs:
                raise FileNotFoundError(
                    f"未在 {root} 下找到 {manifest_name}，也未找到配对的 cell_text_*.txt 与 cell_*.png"
                )
            n = len(pairs)
            tr, va, te = split_indices(n, seed)
            order = {"train": tr, "val": va, "test": te}[split]
            self.pairs = [pairs[i] for i in order]

    def __len__(self) -> int:
        return len(self.records) if self.uses_manifest else len(self.pairs)

    def _to_tensor_01(self, img: Image.Image) -> torch.Tensor:
        import numpy as np

        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        if self.normalize_mode == "none":
            return t * 2.0 - 1.0
        if self.normalize_mode == "percentile" and t.numel() > 0:
            flat = t.view(t.shape[0], -1)
            lo = torch.quantile(flat, self.p_lo / 100.0, dim=1)
            hi = torch.quantile(flat, self.p_hi / 100.0, dim=1)
            lo = lo.view(-1, 1, 1).clamp(min=0.0)
            hi = torch.maximum(hi.view(-1, 1, 1), lo + 1e-6)
            t = (t - lo) / (hi - lo)
            t = t.clamp(0, 1)
        elif self.normalize_mode == "minmax":
            t = t - t.amin(dim=(1, 2), keepdim=True)
            t = t / (t.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6))
        # [-1, 1]
        return t * 2.0 - 1.0

    def __getitem__(self, i: int):
        if self.uses_manifest:
            rec = self.records[i]
            img_path = Path(rec["_resolved_image_path"])
            subj = str(rec["subject"])
            st = str(rec["state"])
            bg = str(rec["background"])
        else:
            img_path, txt_path = self.pairs[i]
            subj, st, bg = triplet_from_txt_path(txt_path)
        im = Image.open(img_path).convert("RGB")
        im = im.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        x = self._to_tensor_01(im)
        result = (x, subj, st, bg, str(img_path))
        if self.return_object_type:
            object_type = str(self.records[i].get("object_type") or "").strip()
            if object_type not in {"nucleus", "whole_cell"}:
                raise ValueError(
                    f"{self.records[i].get('sample_id')} 的 object_type 无效: "
                    f"{object_type!r}"
                )
            result = (*result, object_type)
        if self.return_control_negatives:
            negatives = self.records[i]["control_negatives"]
            enabled = self.records[i]["control_rank_enabled"]
            result = (
                *result,
                str(negatives["subject"]),
                str(negatives["state"]),
                str(negatives["background"]),
                bool(enabled["subject"]),
                bool(enabled["state"]),
                bool(enabled["background"]),
            )
        if self.return_morphology_mask_target:
            import numpy as np

            payload = self.records[i]["morphology_mask_surrogate"]
            enabled = bool(payload.get("enabled", False))
            if enabled:
                with Image.open(
                    self.records[i]["_resolved_morphology_mask_path"]
                ) as mask_image:
                    mask_image = mask_image.convert("L").resize(
                        (128, 128),
                        Image.Resampling.NEAREST,
                    )
                    mask_array = np.asarray(mask_image, dtype=np.uint8)
                mask_target = torch.from_numpy(
                    (mask_array > 0).astype(np.float32)
                ).unsqueeze(0)
                if not bool(mask_target.any()):
                    raise ValueError(
                        f"反事实 morphology mask 为空: "
                        f"{self.records[i]['_resolved_morphology_mask_path']}"
                    )
                source_id = int(payload["source_id"])
            else:
                mask_target = torch.zeros(1, 128, 128, dtype=torch.float32)
                source_id = -1
            result = (
                *result,
                mask_target,
                torch.tensor(source_id, dtype=torch.long),
            )
        if self.return_morphology_class:
            morphology_to_id = {"round": 0, "oval": 1, "elongated": 2}
            result = (
                *result,
                torch.tensor(
                    morphology_to_id[self.records[i]["morphology_class"]],
                    dtype=torch.long,
                ),
            )
        if self.return_morphology_counterfactual_class:
            morphology_to_id = {"round": 0, "oval": 1, "elongated": 2}
            result = (
                *result,
                torch.tensor(
                    morphology_to_id[
                        self.records[i]["morphology_counterfactual_class"]
                    ],
                    dtype=torch.long,
                ),
            )
        if not self.return_target_mask:
            return result
        import numpy as np

        mask_path = Path(self.records[i]["_resolved_target_mask_path"])
        mask_image = Image.open(mask_path).convert("L")
        mask_image = mask_image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST,
        )
        mask_array = np.asarray(mask_image, dtype=np.uint8)
        target_mask = torch.from_numpy((mask_array > 0).astype(np.float32)).unsqueeze(0)
        if not bool(target_mask.any()):
            raise ValueError(f"目标掩膜为空: {mask_path}")
        return (*result, target_mask)


class CounterfactualPairDataset(Dataset):
    """Train-only real-image pairs that differ in exactly one prompt factor."""

    FACTOR_TO_ID = {"subject": 0, "state": 1, "background": 2}

    def __init__(
        self,
        root: str | Path,
        manifest_name: str,
        *,
        expected_factor: str,
        image_size: int = 512,
        normalize_mode: str = "percentile",
        p_lo: float = 1.0,
        p_hi: float = 99.0,
    ):
        super().__init__()
        if expected_factor not in self.FACTOR_TO_ID:
            raise ValueError(
                "counterfactual factor must be subject, state, or background, "
                f"got {expected_factor!r}"
            )
        if normalize_mode not in {"percentile", "minmax", "none"}:
            raise ValueError(f"invalid normalize_mode: {normalize_mode}")
        self.root = Path(root).resolve()
        manifest_path = self.root / manifest_name
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.expected_factor = expected_factor
        self.factor_id = self.FACTOR_TO_ID[expected_factor]
        self.image_size = int(image_size)
        self.normalize_mode = normalize_mode
        self.p_lo = float(p_lo)
        self.p_hi = float(p_hi)
        self.records: list[dict] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("split") != "train":
                    raise ValueError(
                        f"{manifest_path}:{line_number} is not train-only"
                    )
                if record.get("factor") != expected_factor:
                    raise ValueError(
                        f"{manifest_path}:{line_number} factor mismatch"
                    )
                required = (
                    "anchor_image_path",
                    "target_image_path",
                    "anchor_subject",
                    "anchor_state",
                    "anchor_background",
                    "target_subject",
                    "target_state",
                    "target_background",
                    "anchor_sample_id",
                    "target_sample_id",
                )
                missing = [
                    key for key in required if not str(record.get(key) or "").strip()
                ]
                if missing:
                    raise ValueError(
                        f"{manifest_path}:{line_number} missing {missing}"
                    )
                for role in ("anchor", "target"):
                    raw_path = Path(record[f"{role}_image_path"])
                    image_path = (
                        raw_path
                        if raw_path.is_absolute()
                        else self.root / raw_path
                    )
                    if not image_path.is_file():
                        raise FileNotFoundError(
                            f"{manifest_path}:{line_number} missing {image_path}"
                        )
                    record[f"_resolved_{role}_image_path"] = str(image_path)
                raw_target_paths = record.get("target_image_paths")
                allowed_exemplars = {4} if expected_factor == "state" else {1, 4}
                if (
                    not isinstance(raw_target_paths, list)
                    or len(raw_target_paths) not in allowed_exemplars
                ):
                    raise ValueError(
                        f"{manifest_path}:{line_number} requires "
                        f"target exemplar count in {sorted(allowed_exemplars)}"
                    )
                resolved_target_paths = []
                for raw_target_path in raw_target_paths:
                    target_path = Path(str(raw_target_path))
                    if not target_path.is_absolute():
                        target_path = self.root / target_path
                    if not target_path.is_file():
                        raise FileNotFoundError(
                            f"{manifest_path}:{line_number} missing {target_path}"
                        )
                    resolved_target_paths.append(str(target_path))
                record["_resolved_target_image_paths"] = resolved_target_paths
                raw_anchor_paths = record.get("anchor_image_paths")
                if raw_anchor_paths is None:
                    resolved_anchor_paths = [
                        record["_resolved_anchor_image_path"]
                    ]
                else:
                    if not isinstance(raw_anchor_paths, list) or not raw_anchor_paths:
                        raise ValueError(
                            f"{manifest_path}:{line_number} anchor_image_paths "
                            "must be a non-empty list"
                        )
                    resolved_anchor_paths = []
                    for raw_anchor_path in raw_anchor_paths:
                        anchor_path = Path(str(raw_anchor_path))
                        if not anchor_path.is_absolute():
                            anchor_path = self.root / anchor_path
                        if not anchor_path.is_file():
                            raise FileNotFoundError(
                                f"{manifest_path}:{line_number} missing "
                                f"{anchor_path}"
                            )
                        resolved_anchor_paths.append(str(anchor_path))
                record["_resolved_anchor_image_paths"] = resolved_anchor_paths
                unchanged = (
                    {"subject", "state", "background"} - {expected_factor}
                )
                for factor in unchanged:
                    if record[f"anchor_{factor}"] != record[f"target_{factor}"]:
                        raise ValueError(
                            f"{manifest_path}:{line_number} changes non-target "
                            f"factor {factor}"
                        )
                if (
                    record[f"anchor_{expected_factor}"]
                    == record[f"target_{expected_factor}"]
                ):
                    raise ValueError(
                        f"{manifest_path}:{line_number} does not change "
                        f"{expected_factor}"
                    )
                self.records.append(record)
        if not self.records:
            raise ValueError(f"counterfactual pair manifest is empty: {manifest_path}")
        self.records.sort(key=lambda row: str(row.get("pair_id") or ""))

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: str) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BICUBIC,
        )
        import numpy as np

        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        if self.normalize_mode == "none":
            return tensor * 2.0 - 1.0
        if self.normalize_mode == "percentile" and tensor.numel() > 0:
            flat = tensor.view(tensor.shape[0], -1)
            low = torch.quantile(flat, self.p_lo / 100.0, dim=1)
            high = torch.quantile(flat, self.p_hi / 100.0, dim=1)
            low = low.view(-1, 1, 1).clamp(min=0.0)
            high = torch.maximum(
                high.view(-1, 1, 1),
                low + 1e-6,
            )
            tensor = ((tensor - low) / (high - low)).clamp(0, 1)
        elif self.normalize_mode == "minmax":
            tensor = tensor - tensor.amin(dim=(1, 2), keepdim=True)
            tensor = tensor / tensor.amax(
                dim=(1, 2),
                keepdim=True,
            ).clamp(min=1e-6)
        return tensor * 2.0 - 1.0

    def __getitem__(self, index: int):
        record = self.records[index]
        anchor_images = [
            self._load_image(path)
            for path in record["_resolved_anchor_image_paths"]
        ]
        anchor_value = (
            anchor_images[0]
            if len(anchor_images) == 1
            else torch.stack(anchor_images)
        )
        return (
            anchor_value,
            torch.stack(
                [
                    self._load_image(path)
                    for path in record["_resolved_target_image_paths"]
                ]
            ),
            str(record["anchor_subject"]),
            str(record["anchor_state"]),
            str(record["anchor_background"]),
            str(record["target_subject"]),
            str(record["target_state"]),
            str(record["target_background"]),
            self.factor_id,
            str(record["anchor_sample_id"]),
            str(record["target_sample_id"]),
        )


def collate_mscdm(
    batch,
    device: torch.device,
    *,
    max_len: int = 64,
):
    xs = torch.stack([b[0] for b in batch]).to(device)
    subj = [b[1] for b in batch]
    st = [b[2] for b in batch]
    bg = [b[3] for b in batch]
    sid, stid, bgid = MultiScaleTextEncoder.texts_to_batch(subj, st, bg, device, max_len=max_len)
    paths = [b[4] for b in batch]
    if len(batch[0]) <= 5:
        return xs, sid, stid, bgid, paths
    target_masks = torch.stack([b[5] for b in batch]).to(device)
    return xs, sid, stid, bgid, paths, target_masks
