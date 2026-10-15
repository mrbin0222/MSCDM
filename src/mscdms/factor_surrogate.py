from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F


def latent_descriptor(latent: torch.Tensor, grid_size: int = 8) -> torch.Tensor:
    """Match the differentiable descriptor used to fit the latent probes."""
    if latent.ndim != 4:
        raise ValueError(f"expected BCHW latent, got {tuple(latent.shape)}")
    mean_grid = F.adaptive_avg_pool2d(latent, (grid_size, grid_size))
    second_grid = F.adaptive_avg_pool2d(
        latent.float().square(),
        (grid_size, grid_size),
    )
    std_grid = (
        second_grid - mean_grid.float().square()
    ).clamp_min(0).sqrt()
    return torch.cat([mean_grid.float(), std_grid], dim=1).flatten(1)


def dino_descriptor(last_hidden_state: torch.Tensor) -> torch.Tensor:
    if last_hidden_state.ndim != 3 or last_hidden_state.shape[1] < 2:
        raise ValueError(
            "DINO last_hidden_state must have shape [batch,tokens,channels]"
        )
    cls = last_hidden_state[:, 0].float()
    patches = last_hidden_state[:, 1:].float()
    return torch.cat(
        [
            cls,
            patches.mean(dim=1),
            patches.std(dim=1, unbiased=False),
        ],
        dim=1,
    )


class FrozenFeatureProbe(nn.Module):
    def __init__(self, checkpoint: str | Path) -> None:
        super().__init__()
        payload = torch.load(
            str(checkpoint),
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("schema") != "exp4.latent_factor_probe.checkpoint.v1":
            raise ValueError(f"unsupported factor probe checkpoint: {checkpoint}")
        config = payload["model_config"]
        feature_dim = int(config["feature_dim"])
        hidden_dim = int(config["hidden_dim"])
        num_classes = int(config["num_classes"])
        dropout = float(config["dropout"])
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        saved_state = payload["model"]
        if not isinstance(saved_state, dict) or not saved_state:
            raise ValueError(
                f"factor probe checkpoint has no model state: {checkpoint}"
            )
        # Probe-training checkpoints save FeatureProbe.state_dict(), whose
        # Sequential layers live below ``network.*``.  FrozenFeatureProbe
        # reconstructs that Sequential directly, so remove exactly one
        # wrapper prefix.  Also accept early checkpoints that stored the
        # Sequential state without the wrapper.
        has_network_prefix = [
            str(key).startswith("network.") for key in saved_state
        ]
        if all(has_network_prefix):
            network_state = {
                str(key)[len("network.") :]: value
                for key, value in saved_state.items()
            }
        elif not any(has_network_prefix):
            network_state = saved_state
        else:
            raise ValueError(
                "factor probe checkpoint mixes prefixed and unprefixed "
                f"model keys: {checkpoint}"
            )
        self.network.load_state_dict(network_state, strict=True)
        self.register_buffer(
            "feature_mean",
            torch.as_tensor(payload["feature_mean"], dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "feature_std",
            torch.as_tensor(payload["feature_std"], dtype=torch.float32),
            persistent=True,
        )
        if self.feature_mean.shape != (feature_dim,):
            raise ValueError("factor probe feature mean has the wrong shape")
        if self.feature_std.shape != (feature_dim,):
            raise ValueError("factor probe feature std has the wrong shape")
        self.task = str(payload["task"])
        self.branch = str(payload["branch"])
        self.num_classes = num_classes
        self.requires_grad_(False)
        self.eval()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (
            features.float() - self.feature_mean
        ) / self.feature_std.clamp_min(1e-6)
        return self.network(normalized)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def population_energy_distance_to_bank(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    population_size: int,
    bank_logits: torch.Tensor,
    bank_labels: torch.Tensor,
    bank_mean: torch.Tensor,
    bank_std: torch.Tensor,
    bank_within_distance: torch.Tensor,
) -> torch.Tensor:
    """Match generated factor-logit populations to train-real distributions.

    The biased empirical energy distance is used because it constrains both
    location and spread, unlike hard cross-entropy which only pushes samples
    across a decision boundary. The real-real term is frozen and therefore
    contributes no generator gradient.
    """
    if logits.ndim != 2:
        raise ValueError("population logits must have shape [batch,classes]")
    if targets.ndim != 1 or len(targets) != len(logits):
        raise ValueError("population targets must have shape [batch]")
    if population_size <= 1 or len(logits) % population_size != 0:
        raise ValueError("invalid population size for energy distance")
    if bank_logits.ndim != 2 or bank_logits.shape[1] != logits.shape[1]:
        raise ValueError("factor-logit bank shape is incompatible")
    if bank_labels.shape != (len(bank_logits),):
        raise ValueError("factor-logit bank labels have the wrong shape")
    if bank_mean.shape != (logits.shape[1],):
        raise ValueError("factor-logit bank mean has the wrong shape")
    if bank_std.shape != (logits.shape[1],):
        raise ValueError("factor-logit bank std has the wrong shape")
    grouped_targets = targets.view(-1, population_size)
    if not bool((grouped_targets == grouped_targets[:, :1]).all()):
        raise ValueError("each generated population must share one target")
    if bool((bank_std <= 0).any()):
        raise ValueError("factor-logit bank std must be positive")

    normalized_logits = (logits.float() - bank_mean) / bank_std
    normalized_bank = (bank_logits.float() - bank_mean) / bank_std
    grouped_logits = normalized_logits.view(
        -1,
        population_size,
        logits.shape[-1],
    )
    losses: list[torch.Tensor] = []
    for generated_group, target in zip(
        grouped_logits,
        grouped_targets[:, 0],
    ):
        class_id = int(target.item())
        real_group = normalized_bank[bank_labels == class_id]
        if len(real_group) < population_size:
            raise ValueError(
                f"factor-logit bank class {class_id} has too few samples"
            )
        cross_distance = torch.cdist(
            generated_group,
            real_group,
            p=2,
        ).mean()
        generated_within = torch.cdist(
            generated_group,
            generated_group,
            p=2,
        ).mean()
        real_within = bank_within_distance[class_id]
        energy = (
            2.0 * cross_distance
            - generated_within
            - real_within
        )
        # Empirical energy distance is non-negative; clamp only protects
        # against tiny floating-point negatives.
        losses.append(energy.clamp_min(0.0))
    return torch.stack(losses).mean()


class AdaptedDinoStateJudge(nn.Module):
    """Frozen microscopy-adapted DINOv2 state judge with image gradients."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        dino_model_dir: str | Path,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        payload = torch.load(
            str(checkpoint),
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("schema") != "exp4.dnn_dinov2_state_surrogate.v1":
            raise ValueError(
                f"unsupported adapted DNN state checkpoint: {checkpoint}"
            )
        model_path = Path(dino_model_dir)
        if payload["base_model_sha256"] != sha256_file(
            model_path / "model.safetensors"
        ):
            raise ValueError("adapted DNN state DINO base hash mismatch")
        self.dino = AutoModel.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        incompatible = self.dino.load_state_dict(
            payload["trainable_vision_state"],
            strict=False,
        )
        if incompatible.unexpected_keys:
            raise ValueError(
                "unexpected adapted DINO tensors: "
                f"{incompatible.unexpected_keys[:5]}"
            )
        config = payload["probe_config"]
        self.network = nn.Sequential(
            nn.LayerNorm(int(config["feature_dim"])),
            nn.Linear(
                int(config["feature_dim"]),
                int(config["hidden_dim"]),
            ),
            nn.GELU(),
            nn.Dropout(float(config["dropout"])),
            nn.Linear(
                int(config["hidden_dim"]),
                int(config["num_classes"]),
            ),
        )
        saved_probe = payload["probe_state"]
        if not all(str(key).startswith("network.") for key in saved_probe):
            raise ValueError("adapted DNN state probe keys lack network prefix")
        self.network.load_state_dict(
            {
                str(key)[len("network.") :]: value
                for key, value in saved_probe.items()
            },
            strict=True,
        )
        self.num_classes = int(config["num_classes"])
        if self.num_classes != 6:
            raise ValueError("adapted DNN state judge must have six classes")
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        hidden = self.dino(pixel_values=pixels).last_hidden_state
        return self.network(dino_descriptor(hidden))


class FactorSurrogateBundle(nn.Module):
    """Independent frozen factor judges used only for generator supervision."""

    def __init__(
        self,
        *,
        subject_probe: str | Path,
        dnn_state_probe: str | Path,
        background_probe: str | Path,
        dino_model_dir: str | Path,
        dnn_state_dino_checkpoint: str | Path | None = None,
        dnn_state_distribution_bank: str | Path | None = None,
        subject_image_probe: str | Path | None = None,
        subject_distribution_bank: str | Path | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.subject_probe = FrozenFeatureProbe(subject_probe)
        self.dnn_state_probe = FrozenFeatureProbe(dnn_state_probe)
        self.background_probe = FrozenFeatureProbe(background_probe)
        if self.subject_probe.task != "subject_low":
            raise ValueError("subject probe checkpoint has the wrong task")
        if self.dnn_state_probe.task != "state_dynamicnuclearnet_mid":
            raise ValueError("DNN state probe checkpoint has the wrong task")
        if self.background_probe.task != "background_decoded_dinov2":
            raise ValueError("background probe checkpoint has the wrong task")
        self.dino = AutoModel.from_pretrained(
            str(dino_model_dir),
            local_files_only=True,
        )
        self.dnn_state_image_judge = (
            AdaptedDinoStateJudge(
                checkpoint=dnn_state_dino_checkpoint,
                dino_model_dir=dino_model_dir,
            )
            if dnn_state_dino_checkpoint is not None
            else None
        )
        self.subject_image_probe = (
            FrozenFeatureProbe(subject_image_probe)
            if subject_image_probe is not None
            else None
        )
        if (
            self.subject_image_probe is not None
            and self.subject_image_probe.task != "identity_decoded_dinov2"
        ):
            raise ValueError("subject image probe checkpoint has the wrong task")
        self.subject_distribution_bank_path = (
            str(subject_distribution_bank)
            if subject_distribution_bank is not None
            else None
        )
        if subject_distribution_bank is not None:
            if subject_image_probe is None:
                raise ValueError(
                    "subject distribution bank requires the subject image probe"
                )
            subject_payload = torch.load(
                str(subject_distribution_bank),
                map_location="cpu",
                weights_only=False,
            )
            if (
                subject_payload.get("schema")
                != "exp4.subject.real_identity_logit_bank.v1"
            ):
                raise ValueError("unsupported subject identity-logit bank")
            if subject_payload.get("split") != "train":
                raise ValueError("subject identity-logit bank must be train-only")
            if int(subject_payload.get("num_classes", -1)) != 12:
                raise ValueError("subject identity-logit bank must have 12 classes")
            if subject_payload.get("probe_checkpoint_sha256") != sha256_file(
                subject_image_probe
            ):
                raise ValueError("subject identity-logit bank probe hash mismatch")
            if subject_payload.get("base_model_sha256") != sha256_file(
                Path(dino_model_dir) / "model.safetensors"
            ):
                raise ValueError("subject identity-logit bank DINO hash mismatch")
            subject_logits = torch.as_tensor(
                subject_payload["logits"],
                dtype=torch.float32,
            )
            subject_labels = torch.as_tensor(
                subject_payload["labels"],
                dtype=torch.long,
            )
            subject_mean = torch.as_tensor(
                subject_payload["global_mean"],
                dtype=torch.float32,
            )
            subject_std = torch.as_tensor(
                subject_payload["global_std"],
                dtype=torch.float32,
            )
            subject_within = torch.as_tensor(
                subject_payload["class_within_distance"],
                dtype=torch.float32,
            )
            if subject_logits.ndim != 2 or subject_logits.shape[1] != 12:
                raise ValueError(
                    "subject identity-logit bank must have shape [n,12]"
                )
            if subject_labels.shape != (len(subject_logits),):
                raise ValueError("subject identity-logit bank labels mismatch")
            if set(subject_labels.tolist()) != set(range(12)):
                raise ValueError(
                    "subject identity-logit bank must cover 12 classes"
                )
            if subject_mean.shape != (12,) or subject_std.shape != (12,):
                raise ValueError(
                    "subject identity-logit normalization shape mismatch"
                )
            if subject_within.shape != (12,):
                raise ValueError(
                    "subject identity-logit within-distance shape mismatch"
                )
            self.register_buffer(
                "subject_bank_logits",
                subject_logits,
                persistent=True,
            )
            self.register_buffer(
                "subject_bank_labels",
                subject_labels,
                persistent=True,
            )
            self.register_buffer(
                "subject_bank_mean",
                subject_mean,
                persistent=True,
            )
            self.register_buffer(
                "subject_bank_std",
                subject_std.clamp_min(1e-6),
                persistent=True,
            )
            self.register_buffer(
                "subject_bank_within_distance",
                subject_within,
                persistent=True,
            )
        else:
            self.subject_bank_logits = None
            self.subject_bank_labels = None
            self.subject_bank_mean = None
            self.subject_bank_std = None
            self.subject_bank_within_distance = None
        self.dnn_state_distribution_bank_path = (
            str(dnn_state_distribution_bank)
            if dnn_state_distribution_bank is not None
            else None
        )
        if dnn_state_distribution_bank is not None:
            if dnn_state_dino_checkpoint is None:
                raise ValueError(
                    "DNN state distribution bank requires the adapted image judge"
                )
            bank_payload = torch.load(
                str(dnn_state_distribution_bank),
                map_location="cpu",
                weights_only=False,
            )
            if (
                bank_payload.get("schema")
                != "exp4.dnn.real_state_logit_bank.v1"
            ):
                raise ValueError("unsupported DNN state-logit bank checkpoint")
            if bank_payload.get("split") != "train":
                raise ValueError("DNN state-logit bank must be train-only")
            if bank_payload.get("id_to_label") != [
                "3t3",
                "hek293",
                "hela",
                "pc-3",
                "raw264",
                "rpe",
            ]:
                raise ValueError("DNN state-logit bank class order mismatch")
            if bank_payload.get("judge_checkpoint_sha256") != sha256_file(
                dnn_state_dino_checkpoint
            ):
                raise ValueError("DNN state-logit bank judge hash mismatch")
            bank_logits = torch.as_tensor(
                bank_payload["logits"],
                dtype=torch.float32,
            )
            bank_labels = torch.as_tensor(
                bank_payload["labels"],
                dtype=torch.long,
            )
            bank_mean = torch.as_tensor(
                bank_payload["global_mean"],
                dtype=torch.float32,
            )
            bank_std = torch.as_tensor(
                bank_payload["global_std"],
                dtype=torch.float32,
            )
            bank_within = torch.as_tensor(
                bank_payload["class_within_distance"],
                dtype=torch.float32,
            )
            if bank_logits.ndim != 2 or bank_logits.shape[1] != 6:
                raise ValueError("DNN state-logit bank must have shape [n,6]")
            if bank_labels.shape != (len(bank_logits),):
                raise ValueError("DNN state-logit bank labels mismatch")
            if set(bank_labels.tolist()) != set(range(6)):
                raise ValueError("DNN state-logit bank must cover six classes")
            if bank_mean.shape != (6,) or bank_std.shape != (6,):
                raise ValueError("DNN state-logit normalization shape mismatch")
            if bank_within.shape != (6,):
                raise ValueError("DNN state-logit within-distance shape mismatch")
            self.register_buffer(
                "dnn_state_bank_logits",
                bank_logits,
                persistent=True,
            )
            self.register_buffer(
                "dnn_state_bank_labels",
                bank_labels,
                persistent=True,
            )
            self.register_buffer(
                "dnn_state_bank_mean",
                bank_mean,
                persistent=True,
            )
            self.register_buffer(
                "dnn_state_bank_std",
                bank_std.clamp_min(1e-6),
                persistent=True,
            )
            self.register_buffer(
                "dnn_state_bank_within_distance",
                bank_within,
                persistent=True,
            )
        else:
            self.dnn_state_bank_logits = None
            self.dnn_state_bank_labels = None
            self.dnn_state_bank_mean = None
            self.dnn_state_bank_std = None
            self.dnn_state_bank_within_distance = None
        self.register_buffer(
            "dino_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
                1, 3, 1, 1
            ),
            persistent=False,
        )
        self.register_buffer(
            "dino_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
                1, 3, 1, 1
            ),
            persistent=False,
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # These modules are frozen judges. Gradients must flow only to their
        # image/latent inputs, never to the surrogate parameters.
        super().train(False)
        return self

    def subject_logits(self, latent: torch.Tensor) -> torch.Tensor:
        return self.subject_probe(latent_descriptor(latent))

    def dnn_state_logits(self, latent: torch.Tensor) -> torch.Tensor:
        return self.dnn_state_probe(latent_descriptor(latent))

    def _image_pixels(self, decoded_rgb: torch.Tensor) -> torch.Tensor:
        if decoded_rgb.ndim != 4 or decoded_rgb.shape[1] != 3:
            raise ValueError("decoded RGB must have shape [batch,3,height,width]")
        pixels = (decoded_rgb.float() * 0.5 + 0.5).clamp(0.0, 1.0)
        pixels = F.interpolate(
            pixels,
            size=(256, 256),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        pixels = pixels[:, :, 16:240, 16:240]
        return (pixels - self.dino_mean) / self.dino_std

    def dnn_state_image_logits(
        self,
        decoded_rgb: torch.Tensor,
    ) -> torch.Tensor:
        if self.dnn_state_image_judge is None:
            raise ValueError("adapted DNN state image judge is not configured")
        return self.dnn_state_image_judge(self._image_pixels(decoded_rgb))

    def subject_image_logits(
        self,
        decoded_rgb: torch.Tensor,
    ) -> torch.Tensor:
        if self.subject_image_probe is None:
            raise ValueError("subject image probe is not configured")
        pixels = self._image_pixels(decoded_rgb)
        hidden = self.dino(pixel_values=pixels).last_hidden_state
        return self.subject_image_probe(dino_descriptor(hidden))

    def has_subject_distribution_bank(self) -> bool:
        return self.subject_bank_logits is not None

    def subject_distribution_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        population_size: int,
    ) -> torch.Tensor:
        if not self.has_subject_distribution_bank():
            raise ValueError(
                "subject identity-logit distribution bank is not configured"
            )
        assert self.subject_bank_logits is not None
        assert self.subject_bank_labels is not None
        assert self.subject_bank_mean is not None
        assert self.subject_bank_std is not None
        assert self.subject_bank_within_distance is not None
        return population_energy_distance_to_bank(
            logits,
            targets,
            population_size=population_size,
            bank_logits=self.subject_bank_logits,
            bank_labels=self.subject_bank_labels,
            bank_mean=self.subject_bank_mean,
            bank_std=self.subject_bank_std,
            bank_within_distance=self.subject_bank_within_distance,
        )

    def has_dnn_state_distribution_bank(self) -> bool:
        return self.dnn_state_bank_logits is not None

    def dnn_state_distribution_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        population_size: int,
    ) -> torch.Tensor:
        if not self.has_dnn_state_distribution_bank():
            raise ValueError("DNN state-logit distribution bank is not configured")
        assert self.dnn_state_bank_logits is not None
        assert self.dnn_state_bank_labels is not None
        assert self.dnn_state_bank_mean is not None
        assert self.dnn_state_bank_std is not None
        assert self.dnn_state_bank_within_distance is not None
        return population_energy_distance_to_bank(
            logits,
            targets,
            population_size=population_size,
            bank_logits=self.dnn_state_bank_logits,
            bank_labels=self.dnn_state_bank_labels,
            bank_mean=self.dnn_state_bank_mean,
            bank_std=self.dnn_state_bank_std,
            bank_within_distance=self.dnn_state_bank_within_distance,
        )

    def background_logits(self, decoded_rgb: torch.Tensor) -> torch.Tensor:
        pixels = self._image_pixels(decoded_rgb)
        hidden = self.dino(pixel_values=pixels).last_hidden_state
        return self.background_probe(dino_descriptor(hidden))

    def audit(self) -> dict[str, Any]:
        return {
            "subject_classes": self.subject_probe.num_classes,
            "dnn_state_classes": self.dnn_state_probe.num_classes,
            "background_classes": self.background_probe.num_classes,
            "dnn_state_image_judge": (
                self.dnn_state_image_judge is not None
            ),
            "subject_image_probe": self.subject_image_probe is not None,
            "subject_distribution_bank": (
                self.has_subject_distribution_bank()
            ),
            "subject_distribution_bank_path": (
                self.subject_distribution_bank_path
            ),
            "dnn_state_distribution_bank": (
                self.has_dnn_state_distribution_bank()
            ),
            "dnn_state_distribution_bank_path": (
                self.dnn_state_distribution_bank_path
            ),
            "all_parameters_frozen": all(
                not parameter.requires_grad for parameter in self.parameters()
            ),
        }
