"""Self-contained LoRA (Low-Rank Adaptation) modules.

Implemented from scratch so the whole training path has a minimal dependency
surface and full control over memory behaviour (no peft/trl/accelerate).
Also handles GPT2's Conv1D modules, which are stored as (in, out) while
nn.Linear stores weights as (out, in).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap an nn.Linear with a trainable low-rank adapter. Base weights frozen."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(in_f, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_f))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(p=dropout) if dropout else nn.Identity()
        for p in base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = x @ self.lora_A @ self.lora_B
        return base_out + self.dropout(delta) * self.scaling

    def merge_into_base(self) -> None:
        # forward delta = x @ A @ B has shape (..., out).
        # nn.Linear.weight is (out, in), so the weight-space delta is (A@B).T = B.T @ A.T.
        with torch.no_grad():
            delta = self.lora_B.detach().t() @ self.lora_A.detach().t()
            self.base.weight.add_(delta, alpha=self.scaling)


def _getattr(obj: nn.Module, path: str) -> nn.Module:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _convert_conv1d_to_linear(model: nn.Module) -> list[str]:
    """Replace GPT2 Conv1D modules with nn.Linear (weight transposed).
    Returns the names of converted modules so they can be restored on save."""
    conv_names: list[str] = []
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ == "Conv1D":
            parent, _, leaf = name.rpartition(".")
            container = model if not parent else _getattr(model, parent)
            w, b = module.weight, module.bias  # w: (in, out), b: (out,)
            lin = nn.Linear(w.size(0), w.size(1))
            with torch.no_grad():
                lin.weight.copy_(w.t())
                if b is not None:
                    lin.bias.copy_(b)
            setattr(container, leaf, lin)
            conv_names.append(name)
    return conv_names


def _restore_conv1d(model: nn.Module, names: Sequence[str]) -> None:
    """Convert the given nn.Linear modules back to GPT2 Conv1D before saving."""
    if not names:
        return
    from transformers.pytorch_utils import Conv1D

    for name in names:
        parent, _, leaf = name.rpartition(".")
        container = model if not parent else _getattr(model, parent)
        module = getattr(container, leaf)  # nn.Linear, weight (out, in)
        conv = Conv1D(nf=module.out_features, nx=module.in_features)
        with torch.no_grad():
            conv.weight.copy_(module.weight.t())
            if module.bias is not None:
                conv.bias.copy_(module.bias)
        setattr(container, leaf, conv)


def wrap_linear_layers(
    model: nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.0, target_modules: Sequence[str] | None = None
) -> tuple[list[tuple[str, LoRALinear]], list[str]]:
    """In-place wrap all nn.Linear (optionally filtered) with LoRA.

    Returns ((name, wrapper) pairs, converted Conv1D names). lm_head/embed
    are skipped by default. Call merge_and_unwrap then restore_conv1d on save.
    """
    conv_names = _convert_conv1d_to_linear(model)
    names: list[str] = []

    def should_wrap(name: str) -> bool:
        if "lm_head" in name or "embed" in name:
            return False
        if target_modules is None:
            return True
        return any(t in name for t in target_modules)

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and should_wrap(name):
            parent, _, leaf = name.rpartition(".")
            container = model if not parent else _getattr(model, parent)
            setattr(container, leaf, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
            names.append(name)

    wrapped: list[tuple[str, LoRALinear]] = []
    for name in names:
        obj = _getattr(model, name)
        if isinstance(obj, LoRALinear):
            wrapped.append((name, obj))
    return wrapped, conv_names


def merge_and_unwrap(model: nn.Module, wrapped: Iterable[tuple[str, LoRALinear]]) -> None:
    """Merge LoRA deltas back into base weights and restore plain nn.Linear."""
    for name, wrapper in wrapped:
        wrapper.merge_into_base()
        parent, _, leaf = name.rpartition(".")
        container = model if not parent else _getattr(model, parent)
        setattr(container, leaf, wrapper.base)


def restore_conv1d(model: nn.Module, conv_names: Sequence[str]) -> None:
    """Restore GPT2 Conv1D modules after merging (so saved weights reload cleanly)."""
    _restore_conv1d(model, conv_names)
