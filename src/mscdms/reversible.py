from __future__ import annotations

import contextlib

import torch
import torch.nn as nn


def _autocast_off(t: torch.Tensor):
    """在自定义 backward 内关闭 AMP，避免可逆重算时 GroupNorm 等出现 Half/Float 混用。"""
    if t.is_cuda:
        return torch.amp.autocast("cuda", enabled=False)
    return contextlib.nullcontext()


class CondFiLM(nn.Module):
    """条件向量 -> 通道仿射参数。"""

    def __init__(self, cond_dim: int, ch: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, ch)
        self.to_beta = nn.Linear(cond_dim, ch)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        g = self.to_gamma(c).unsqueeze(-1).unsqueeze(-1)
        b = self.to_beta(c).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + g) + b


class InvertibleFunction(torch.autograd.Function):
    """
    自定义 autograd Function：前向时不存储中间激活，
    反向时利用可逆块的逆运算从输出重构输入，再重计算前向以获取梯度。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, cond: torch.Tensor, *params_and_meta):
        block = params_and_meta[-1]
        num_params = len(params_and_meta) - 1
        with torch.no_grad():
            y = block._raw_forward(x, cond)
        ctx.save_for_backward(y, cond)
        ctx.block = block
        ctx.num_params = num_params
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        y, cond = ctx.saved_tensors
        block = ctx.block
        w_dtype = next(block.parameters()).dtype

        with torch.no_grad():
            with _autocast_off(y):
                y_i = y.to(dtype=w_dtype)
                cond_i = cond.to(dtype=w_dtype)
                x = block._invert(y_i, cond_i)

        x = x.detach().requires_grad_(True)
        cond_d = cond.detach().to(dtype=w_dtype).requires_grad_(True)
        with torch.enable_grad():
            with _autocast_off(y):
                y_hat = block._raw_forward(x, cond_d)

        params = list(block.parameters())
        trainable_param_indices = [i for i, p in enumerate(params) if p.requires_grad]
        grad_inputs = [x, cond_d] + [params[i] for i in trainable_param_indices]
        grads = torch.autograd.grad(y_hat, grad_inputs, grad_y, allow_unused=True)

        grad_x = grads[0]
        grad_cond = grads[1]
        param_grads: list[torch.Tensor | None] = [None] * len(params)
        for param_index, grad in zip(trainable_param_indices, grads[2:]):
            param_grads[param_index] = grad
        return (grad_x, grad_cond) + tuple(param_grads) + (None,)


class ReversibleCondBlock(nn.Module):
    """
    可逆残差块（式 4-25–4-28 思想）：通道二分 + 两个子网络。
    当 save_memory=True 且处于训练模式时，使用 InvertibleFunction 自定义反向传播，
    反向时通过逆运算重构输入而非存储中间激活，从而降低显存开销。
    """

    def __init__(self, ch: int, cond_dim: int, save_memory: bool = True):
        super().__init__()
        c1, c2 = ch // 2, ch - ch // 2
        self.c1, self.c2 = c1, c2
        self.save_memory = save_memory
        self.f = nn.Sequential(
            nn.GroupNorm(8, c2),
            nn.SiLU(),
            nn.Conv2d(c2, c1, 3, padding=1),
        )
        self.g = nn.Sequential(
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv2d(c1, c2, 3, padding=1),
        )
        self.film_f = CondFiLM(cond_dim, c1)
        self.film_g = CondFiLM(cond_dim, c2)

    def _raw_forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.split([self.c1, self.c2], dim=1)
        y1 = x1 + self.film_f(self.f(x2), cond)
        y2 = x2 + self.film_g(self.g(y1), cond)
        return torch.cat([y1, y2], dim=1)

    def _invert(self, y: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """从输出恢复输入（精确可逆）。"""
        y1, y2 = y.split([self.c1, self.c2], dim=1)
        x2 = y2 - self.film_g(self.g(y1), cond)
        x1 = y1 - self.film_f(self.f(x2), cond)
        return torch.cat([x1, x2], dim=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.training and self.save_memory:
            return InvertibleFunction.apply(x, cond, *self.parameters(), self)
        return self._raw_forward(x, cond)

    @staticmethod
    def invert(y: torch.Tensor, block: "ReversibleCondBlock", cond: torch.Tensor) -> torch.Tensor:
        """向后兼容的静态方法，内部委托给实例方法。"""
        return block._invert(y, cond)
