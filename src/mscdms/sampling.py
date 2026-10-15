from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterator

import torch
from torch.utils.data import Sampler


class SourceBalancedSmallAreaSampler(Sampler[int]):
    """Preserve source counts while oversampling small instances in one source.

    Non-target sources are shuffled without replacement. The target source keeps
    the same total number of draws, but samples with the smallest bbox areas are
    drawn with a larger relative probability. All ranks reconstruct the same
    deterministic global index stream and take disjoint strided slices.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        target_source: str,
        area_quantile: float,
        hard_multiplier: float,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ) -> None:
        if not records:
            raise ValueError("records must not be empty")
        if not 0.0 < area_quantile < 1.0:
            raise ValueError("area_quantile must be in (0, 1)")
        if hard_multiplier <= 1.0:
            raise ValueError("hard_multiplier must be greater than 1")
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), found {rank}")

        grouped: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            source = str(record.get("source_name") or "").strip()
            if not source:
                raise ValueError(f"record {index} is missing source_name")
            bbox = record.get("bbox_xywh")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError(f"record {index} is missing bbox_xywh")
            grouped.setdefault(source, []).append(index)
        if target_source not in grouped:
            raise ValueError(f"target source not found: {target_source}")

        self.records = records
        self.target_source = target_source
        self.area_quantile = float(area_quantile)
        self.hard_multiplier = float(hard_multiplier)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.source_indices = {
            source: tuple(indices) for source, indices in sorted(grouped.items())
        }
        target_indices = sorted(
            self.source_indices[target_source],
            key=lambda index: (
                self._bbox_area(records[index]),
                str(records[index].get("sample_id") or ""),
                index,
            ),
        )
        hard_count = max(1, int(math.ceil(len(target_indices) * area_quantile)))
        self.hard_indices = frozenset(target_indices[:hard_count])
        self.target_weights = torch.tensor(
            [
                hard_multiplier if index in self.hard_indices else 1.0
                for index in self.source_indices[target_source]
            ],
            dtype=torch.double,
        )
        self.dataset_size = len(records)
        self.num_samples = int(math.ceil(self.dataset_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    @staticmethod
    def _bbox_area(record: dict[str, Any]) -> int:
        bbox = record["bbox_xywh"]
        return int(bbox[2]) * int(bbox[3])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def global_indices_for_epoch(self, epoch: int | None = None) -> list[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + (self.epoch if epoch is None else int(epoch)))
        selected: list[int] = []
        for source, indices_tuple in self.source_indices.items():
            indices = list(indices_tuple)
            if source == self.target_source:
                positions = torch.multinomial(
                    self.target_weights,
                    num_samples=len(indices),
                    replacement=True,
                    generator=generator,
                ).tolist()
                selected.extend(indices[position] for position in positions)
            else:
                permutation = torch.randperm(
                    len(indices),
                    generator=generator,
                ).tolist()
                selected.extend(indices[position] for position in permutation)
        permutation = torch.randperm(len(selected), generator=generator).tolist()
        selected = [selected[position] for position in permutation]
        if self.total_size > len(selected):
            selected.extend(selected[: self.total_size - len(selected)])
        return selected

    def __iter__(self) -> Iterator[int]:
        indices = self.global_indices_for_epoch()
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def audit(self, epoch: int = 0) -> dict[str, Any]:
        global_indices = self.global_indices_for_epoch(epoch)[: self.dataset_size]
        source_counts = Counter(
            str(self.records[index]["source_name"]) for index in global_indices
        )
        target_draws = [
            index
            for index in global_indices
            if str(self.records[index]["source_name"]) == self.target_source
        ]
        hard_draws = sum(index in self.hard_indices for index in target_draws)
        hard_count = len(self.hard_indices)
        target_count = len(self.source_indices[self.target_source])
        expected_fraction = (
            self.hard_multiplier * hard_count
            / (self.hard_multiplier * hard_count + target_count - hard_count)
        )
        return {
            "strategy": "small_area_source_v1",
            "target_source": self.target_source,
            "area_quantile": self.area_quantile,
            "hard_multiplier": self.hard_multiplier,
            "hard_unique_count": hard_count,
            "target_unique_count": target_count,
            "expected_hard_draw_fraction": expected_fraction,
            "epoch": int(epoch),
            "global_draw_count": len(global_indices),
            "source_draw_counts": dict(sorted(source_counts.items())),
            "target_hard_draw_count": hard_draws,
            "target_hard_draw_fraction": hard_draws / max(1, len(target_draws)),
        }


class SourceMorphologyBalancedSampler(Sampler[int]):
    """Balance morphology classes within each source without changing source totals."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ) -> None:
        if not records:
            raise ValueError("records must not be empty")
        if num_replicas < 1 or rank < 0 or rank >= num_replicas:
            raise ValueError("invalid distributed sampler configuration")
        grouped: dict[tuple[str, str], list[int]] = {}
        source_counts: Counter[str] = Counter()
        for index, record in enumerate(records):
            source = str(record.get("source_name") or "").strip()
            morphology = str(record.get("morphology_class") or "").strip()
            if not source or morphology not in {"round", "oval", "elongated"}:
                raise ValueError(
                    f"record {index} has invalid source/morphology: "
                    f"{source!r}/{morphology!r}"
                )
            grouped.setdefault((source, morphology), []).append(index)
            source_counts[source] += 1
        expected_groups = {
            (source, morphology)
            for source in source_counts
            for morphology in ("elongated", "oval", "round")
        }
        if set(grouped) != expected_groups:
            raise ValueError(
                "every source must contain elongated/oval/round records"
            )
        targets: dict[tuple[str, str], int] = {}
        for source, source_count in sorted(source_counts.items()):
            base, remainder = divmod(source_count, 3)
            for offset, morphology in enumerate(("elongated", "oval", "round")):
                targets[(source, morphology)] = base + (offset < remainder)

        self.records = records
        self.groups = {
            key: tuple(indices) for key, indices in sorted(grouped.items())
        }
        self.target_counts = targets
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.dataset_size = len(records)
        self.num_samples = int(math.ceil(self.dataset_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def global_indices_for_epoch(self, epoch: int | None = None) -> list[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + (self.epoch if epoch is None else int(epoch)))
        selected: list[int] = []
        for key, indices_tuple in self.groups.items():
            indices = list(indices_tuple)
            needed = self.target_counts[key]
            permutation = torch.randperm(len(indices), generator=generator).tolist()
            if needed <= len(indices):
                draws = [indices[position] for position in permutation[:needed]]
            else:
                draws = [indices[position] for position in permutation]
                extras = torch.randint(
                    len(indices),
                    (needed - len(indices),),
                    generator=generator,
                ).tolist()
                draws.extend(indices[position] for position in extras)
            selected.extend(draws)
        if len(selected) != self.dataset_size:
            raise RuntimeError("morphology-balanced sampler changed epoch size")
        permutation = torch.randperm(len(selected), generator=generator).tolist()
        selected = [selected[position] for position in permutation]
        if self.total_size > len(selected):
            selected.extend(selected[: self.total_size - len(selected)])
        return selected

    def __iter__(self) -> Iterator[int]:
        indices = self.global_indices_for_epoch()
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def audit(self, epoch: int = 0) -> dict[str, Any]:
        indices = self.global_indices_for_epoch(epoch)[: self.dataset_size]
        group_counts = Counter(
            (
                str(self.records[index]["source_name"]),
                str(self.records[index]["morphology_class"]),
            )
            for index in indices
        )
        source_counts = Counter(source for source, _ in group_counts.elements())
        return {
            "strategy": "source_morphology_balanced_v1",
            "epoch": int(epoch),
            "global_draw_count": len(indices),
            "source_draw_counts": dict(sorted(source_counts.items())),
            "source_morphology_draw_counts": {
                f"{source}:{morphology}": count
                for (source, morphology), count in sorted(group_counts.items())
            },
            "source_morphology_unique_counts": {
                f"{source}:{morphology}": len(self.groups[(source, morphology)])
                for source, morphology in sorted(self.groups)
            },
        }


class SupportCellBalancedSampler(Sampler[int]):
    """Balance factual subject/state/background support cells per epoch.

    A support cell is the exact tuple ``source_dataset × subject_key ×
    state_family_key × background_key``.  The sampler keeps the epoch length
    unchanged, assigns every factual cell either ``floor(N/K)`` or
    ``ceil(N/K)`` draws, and never invents a condition combination that is not
    present in the manifest.  Small cells are traversed once before any
    replacement draws are added, so every unique bridge sample is exposed in
    each epoch.

    All distributed ranks reconstruct the same deterministic global stream and
    take strided slices, matching PyTorch's DistributedSampler convention.
    """

    CELL_FIELDS = (
        "source_dataset",
        "subject_key",
        "state_family_key",
        "background_key",
    )

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ) -> None:
        if not records:
            raise ValueError("records must not be empty")
        if num_replicas < 1 or rank < 0 or rank >= num_replicas:
            raise ValueError("invalid distributed sampler configuration")

        grouped: dict[tuple[str, str, str, str], list[int]] = {}
        for index, record in enumerate(records):
            values = tuple(str(record.get(field) or "").strip() for field in self.CELL_FIELDS)
            if any(not value for value in values):
                missing = [
                    field
                    for field, value in zip(self.CELL_FIELDS, values)
                    if not value
                ]
                raise ValueError(
                    f"record {index} is missing support-cell fields: "
                    f"{', '.join(missing)}"
                )
            grouped.setdefault(values, []).append(index)
        if len(grouped) < 2:
            raise ValueError("support-cell balancing requires at least two factual cells")

        ordered_keys = tuple(sorted(grouped))
        base, remainder = divmod(len(records), len(ordered_keys))
        targets = {
            key: base + (offset < remainder)
            for offset, key in enumerate(ordered_keys)
        }

        self.records = records
        self.groups = {
            key: tuple(grouped[key])
            for key in ordered_keys
        }
        self.target_counts = targets
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.dataset_size = len(records)
        self.num_samples = int(math.ceil(self.dataset_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    @staticmethod
    def cell_name(key: tuple[str, str, str, str]) -> str:
        return "|".join(key)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def global_indices_for_epoch(self, epoch: int | None = None) -> list[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + (self.epoch if epoch is None else int(epoch)))
        selected: list[int] = []
        for key, indices_tuple in self.groups.items():
            indices = list(indices_tuple)
            needed = self.target_counts[key]
            permutation = torch.randperm(len(indices), generator=generator).tolist()
            draws = [indices[position] for position in permutation[:needed]]
            if needed > len(indices):
                extras = torch.randint(
                    len(indices),
                    (needed - len(indices),),
                    generator=generator,
                ).tolist()
                draws.extend(indices[position] for position in extras)
            selected.extend(draws)
        if len(selected) != self.dataset_size:
            raise RuntimeError("support-cell-balanced sampler changed epoch size")
        permutation = torch.randperm(len(selected), generator=generator).tolist()
        selected = [selected[position] for position in permutation]
        if self.total_size > len(selected):
            selected.extend(selected[: self.total_size - len(selected)])
        return selected

    def __iter__(self) -> Iterator[int]:
        indices = self.global_indices_for_epoch()
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def audit(self, epoch: int = 0) -> dict[str, Any]:
        indices = self.global_indices_for_epoch(epoch)[: self.dataset_size]
        cell_draws: Counter[tuple[str, str, str, str]] = Counter()
        cell_distinct: dict[tuple[str, str, str, str], set[int]] = {
            key: set() for key in self.groups
        }
        source_draws: Counter[str] = Counter()
        source_unique: Counter[str] = Counter()
        for key, unique_indices in self.groups.items():
            source_unique[key[0]] += len(unique_indices)
        for index in indices:
            record = self.records[index]
            key = tuple(
                str(record[field]).strip() for field in self.CELL_FIELDS
            )
            cell_draws[key] += 1
            cell_distinct[key].add(index)
            source_draws[key[0]] += 1

        draw_values = list(cell_draws.values())
        bridge_draws = sum(
            count
            for source, count in source_draws.items()
            if source in {"BBBC021", "BBBC022"}
        )
        return {
            "strategy": "support_cell_balanced_v1",
            "epoch": int(epoch),
            "global_draw_count": len(indices),
            "support_cell_count": len(self.groups),
            "support_cell_min_draw_count": min(draw_values),
            "support_cell_max_draw_count": max(draw_values),
            "source_unique_counts": dict(sorted(source_unique.items())),
            "source_draw_counts": dict(sorted(source_draws.items())),
            "bridge_source_draw_fraction": bridge_draws / max(1, len(indices)),
            "support_cell_unique_counts": {
                self.cell_name(key): len(self.groups[key])
                for key in self.groups
            },
            "support_cell_draw_counts": {
                self.cell_name(key): cell_draws[key]
                for key in self.groups
            },
            "support_cell_distinct_drawn_counts": {
                self.cell_name(key): len(cell_distinct[key])
                for key in self.groups
            },
            "support_cell_repeat_draw_counts": {
                self.cell_name(key): cell_draws[key] - len(cell_distinct[key])
                for key in self.groups
            },
        }
