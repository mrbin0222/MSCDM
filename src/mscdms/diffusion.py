from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


def make_beta_schedule(
    schedule: str,
    num_timesteps: int,
    beta_start: float,
    beta_end: float,
) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_timesteps)
    raise ValueError(schedule)


class GaussianDiffusion(nn.Module):
    """DDPM 前向与 DDIM(η=0) 反向。"""

    def __init__(self, num_train_timesteps: int, beta_schedule: str, beta_start: float, beta_end: float):
        super().__init__()
        betas = make_beta_schedule(beta_schedule, num_train_timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_ac", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_om_ac", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self.sqrt_ac[t][:, None, None, None]
        s2 = self.sqrt_om_ac[t][:, None, None, None]
        return s1 * x0 + s2 * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (x_t - self.sqrt_om_ac[t][:, None, None, None] * eps) / self.sqrt_ac[t][:, None, None, None]

    def velocity_target(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """v = alpha * epsilon - sigma * x0."""
        alpha = self.sqrt_ac[t][:, None, None, None]
        sigma = self.sqrt_om_ac[t][:, None, None, None]
        return alpha * noise - sigma * x0

    def training_target(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
        prediction_type: str,
    ) -> torch.Tensor:
        if prediction_type == "epsilon":
            return noise
        if prediction_type == "v":
            return self.velocity_target(x0, t, noise)
        raise ValueError(f"Unsupported prediction_type: {prediction_type}")

    def model_output_to_x0_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
        prediction_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prediction_type == "epsilon":
            eps = model_output
            x0 = self.predict_x0_from_eps(x_t, t, eps)
            return x0, eps
        if prediction_type == "v":
            alpha = self.sqrt_ac[t][:, None, None, None]
            sigma = self.sqrt_om_ac[t][:, None, None, None]
            # 与 x_t = alpha*x0 + sigma*epsilon 联立求解。
            x0 = alpha * x_t - sigma * model_output
            eps = sigma * x_t + alpha * model_output
            return x0, eps
        raise ValueError(f"Unsupported prediction_type: {prediction_type}")

    def ddim_timesteps(self, num_inference_steps: int) -> torch.Tensor:
        """Return the exact integer schedule used by deterministic sampling."""
        if not 0 < num_inference_steps <= self.betas.shape[0]:
            raise ValueError(
                "num_inference_steps must be positive and no larger than the "
                "training schedule"
            )
        return torch.linspace(
            self.betas.shape[0] - 1,
            0,
            num_inference_steps,
            device=self.betas.device,
        ).long()

    def ddim_eta0_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
        *,
        prediction_type: str,
        t_next: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one differentiable DDIM eta=0 update.

        Training-time trajectory objectives call the same helper as inference,
        so their update formula cannot silently drift away from the sampler.
        """
        if x_t.shape != model_output.shape:
            raise ValueError("x_t and model_output must have identical shapes")
        if t.shape != (x_t.shape[0],):
            raise ValueError("t must have shape [batch]")
        if t_next is not None and t_next.shape != t.shape:
            raise ValueError("t_next must be None or have the same shape as t")
        if t.dtype != torch.long or (
            t_next is not None and t_next.dtype != torch.long
        ):
            raise ValueError("t and t_next must use torch.long indices")
        if bool((t < 0).any()) or bool((t >= self.betas.shape[0]).any()):
            raise ValueError("t contains indices outside the diffusion schedule")
        if t_next is not None and (
            bool((t_next < 0).any())
            or bool((t_next >= self.betas.shape[0]).any())
            or bool((t_next >= t).any())
        ):
            raise ValueError("t_next must be in range and strictly smaller than t")
        x0, eps = self.model_output_to_x0_eps(
            x_t,
            t,
            model_output,
            prediction_type,
        )
        if t_next is None:
            return x0, eps, x0
        alpha_cumprod_next = self.alphas_cumprod[t_next][
            :, None, None, None
        ]
        x_next = (
            torch.sqrt(alpha_cumprod_next) * x0
            + torch.sqrt(1.0 - alpha_cumprod_next) * eps
        )
        return x0, eps, x_next

    @torch.no_grad()
    def ddim_sample(
        self,
        denoise_fn,
        shape: tuple[int, ...],
        cond,
        cfg_scale: float | torch.Tensor,
        num_inference_steps: int,
        null_cond=None,
        generator: torch.Generator | None = None,
        prediction_type: str = "epsilon",
        trace_callback: Callable[
            [int, torch.Tensor, torch.Tensor], None
        ] | None = None,
    ) -> torch.Tensor:
        device = self.betas.device
        b = shape[0]
        ts = self.ddim_timesteps(num_inference_steps)
        x = torch.randn(shape, device=device, generator=generator)
        for i in range(num_inference_steps):
            t = ts[i].expand(b)

            def eps_fn(c) -> torch.Tensor:
                return denoise_fn(x, t, c)

            if isinstance(cfg_scale, torch.Tensor):
                if cfg_scale.ndim == 0:
                    scale = cfg_scale.to(device=device, dtype=x.dtype)
                elif cfg_scale.shape == (b,):
                    scale = cfg_scale.to(
                        device=device,
                        dtype=x.dtype,
                    )[:, None, None, None]
                else:
                    raise ValueError(
                        "tensor cfg_scale must be scalar or shape [batch], "
                        f"got {tuple(cfg_scale.shape)}"
                    )
                use_guidance = bool(torch.any(scale != 1.0).item())
            else:
                scale = float(cfg_scale)
                use_guidance = scale != 1.0
            if null_cond is not None and use_guidance:
                e_null = eps_fn(null_cond)
                e_cond = eps_fn(cond)
                e = e_null + scale * (e_cond - e_null)
            else:
                e = eps_fn(cond)

            t_next = (
                ts[i + 1].expand(b)
                if i < num_inference_steps - 1
                else None
            )
            x0, _eps, x_next = self.ddim_eta0_step(
                x,
                t,
                e,
                prediction_type=prediction_type,
                t_next=t_next,
            )
            if trace_callback is not None:
                # The callback receives the clean-latent estimate at every
                # inference step.  It is diagnostic-only and must not alter
                # the sampler state.
                trace_callback(i, t, x0)
            # t=0 still contains beta_0 noise; the helper returns clean x0 on
            # the final step, exactly matching the previous sampler behavior.
            x = x_next
        return x
