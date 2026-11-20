"""LoRA (Low-Rank Adaptation), written from scratch for GPT-2.

A frozen weight W gets a trainable low-rank update:  y = x W + (alpha / r) * x A^T B^T
A is (r x in) with small random values and B is (out x r) with zeros, so at the start the model behaves
exactly like the base model. Only A and B are trained: a tiny fraction of the parameters.

GPT-2 uses `transformers.pytorch_utils.Conv1D` (weight shaped (in, out)) instead of nn.Linear
(weight shaped (out, in)); LoRALinear handles both.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

DEFAULT_TARGETS = ("c_attn", "c_proj", "c_fc")   # attention qkv, output projections, MLP in-projection


class LoRALinear(nn.Module):
    """Wraps a frozen Linear/Conv1D layer and adds a trainable low-rank update."""

    def __init__(self, base: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        if isinstance(base, Conv1D):
            in_f, out_f = base.weight.shape
        elif isinstance(base, nn.Linear):
            out_f, in_f = base.weight.shape
        else:
            raise TypeError(f"LoRA supports Linear and Conv1D, not {type(base).__name__}")
        self.base = base
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return self.base(x) + update * self.scaling

    def delta_weight(self) -> torch.Tensor:
        """The update in the base layer's own weight layout."""
        delta = (self.lora_B @ self.lora_A) * self.scaling           # (out, in)
        return delta.T if isinstance(self.base, Conv1D) else delta


def inject_lora(model: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.05,
                targets: tuple[str, ...] = DEFAULT_TARGETS) -> int:
    """Freeze the whole model, then wrap every layer whose name ends with one of `targets`.
    Returns how many layers were wrapped."""
    for p in model.parameters():
        p.requires_grad = False
    to_wrap = [(name, mod) for name, mod in model.named_modules()
               if name.split(".")[-1] in targets and isinstance(mod, (nn.Linear, Conv1D))]
    for name, mod in to_wrap:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(mod, r=r, alpha=alpha, dropout=dropout))
    model.lora_config = {"r": r, "alpha": alpha, "dropout": dropout, "targets": list(targets)}
    return len(to_wrap)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def save_adapter(model: nn.Module, path: str | Path, extra: dict | None = None) -> Path:
    """Save only the LoRA weights (about a megabyte) plus the config needed to re-inject them."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), path / "adapter.pt")
    (path / "adapter_config.json").write_text(json.dumps({**model.lora_config, **(extra or {})}, indent=2))
    return path


def load_adapter(model: nn.Module, path: str | Path) -> dict:
    """Inject LoRA layers into a fresh base model and load saved adapter weights. Returns the config."""
    path = Path(path)
    cfg = json.loads((path / "adapter_config.json").read_text())
    inject_lora(model, r=cfg["r"], alpha=cfg["alpha"], dropout=cfg["dropout"], targets=tuple(cfg["targets"]))
    missing, unexpected = model.load_state_dict(torch.load(path / "adapter.pt", map_location="cpu"), strict=False)
    if unexpected or any("lora_" in k for k in missing):
        raise ValueError(f"adapter does not fit this model: missing={missing[:3]} unexpected={unexpected[:3]}")
    return cfg


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every adapter into its base weight (W <- W + delta) and swap the LoRA wrapper back out
    for the plain layer. The result is an ordinary transformers model: same outputs, no extra matmuls,
    and it can be saved with save_pretrained() and loaded without any LoRA code."""
    wrapped = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, LoRALinear)]
    for name, mod in wrapped:
        mod.base.weight += mod.delta_weight().to(mod.base.weight.dtype)
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, mod.base)
    if hasattr(model, "lora_config"):
        del model.lora_config
    for p in model.parameters():
        p.requires_grad = True
    return model
