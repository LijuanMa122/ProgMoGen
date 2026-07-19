"""
Heterogeneous constraint interface for MIC.

Each Constraint provides:
  - type: objective (Eq.10, differentiable) or criterion (Eq.9, forward-only)
  - v_d / E: energy / terminal cost on clean motion estimate
  - mask M_k: spatial-temporal scope in latent space [1,C,1,T]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Sequence, Union

import torch


class ConstraintType(str, Enum):
    OBJECTIVE = "objective"
    CRITERION = "criterion"


EnergyFn = Callable[..., Union[torch.Tensor, float]]


@dataclass
class Constraint:
    name: str
    type: ConstraintType
    # Energy / evaluator. Signature: (x0_hat, *, diffusion=None, joints=None) -> scalar
    energy_fn: EnergyFn
    # Scope mask in motion latent space; None → all-ones
    mask: Optional[torch.Tensor] = None
    # Optional: build mask lazily from (shape, length, device)
    mask_fn: Optional[Callable] = None

    @property
    def is_criterion(self) -> bool:
        return self.type == ConstraintType.CRITERION

    @property
    def is_objective(self) -> bool:
        return self.type == ConstraintType.OBJECTIVE

    def ensure_mask(self, shape, length: int, device) -> torch.Tensor:
        if self.mask is not None:
            return self.mask.to(device)
        if self.mask_fn is not None:
            self.mask = self.mask_fn(shape, length, device)
            return self.mask
        self.mask = torch.ones(shape, device=device)
        return self.mask

    def energy(self, x0_hat: torch.Tensor, diffusion=None) -> torch.Tensor:
        """Terminal cost E / v_d on clean estimate (may be non-diff for criterion)."""
        val = self.energy_fn(x0_hat, diffusion=diffusion)
        if not torch.is_tensor(val):
            val = torch.tensor(float(val), device=x0_hat.device)
        return val.reshape(()).to(x0_hat.device)

    def violation(self, x0_hat: torch.Tensor, diffusion=None) -> torch.Tensor:
        """c̃ = max(0, v_d(ẑ_T)) for regulator."""
        return torch.clamp(self.energy(x0_hat, diffusion=diffusion), min=0.0)

    # aliases used in README
    def v_d(self, x0_hat, diffusion=None):
        return self.energy(x0_hat, diffusion=diffusion)

    def E(self, x0_hat, diffusion=None):
        return self.energy(x0_hat, diffusion=diffusion)


def ones_mask(shape, length, device):
    return torch.ones(shape, device=device)


def latent_mask_from_joint_frames(
    shape,
    length: int,
    device,
    joint_ids: Sequence[int],
    frames: Optional[Sequence[int]] = None,
    feat_dim: int = 263,
):
    """
    Coarse latent-space mask: activate all feature channels on selected frames.
    (Fine joint→feature mapping can be refined later; skeleton uses frame scope.)
    shape: [B, C, 1, T]
    """
    mask = torch.zeros(shape, device=device)
    T = shape[-1]
    if frames is None:
        frame_idx = list(range(min(length, T)))
    else:
        frame_idx = [f for f in frames if 0 <= f < min(length, T)]
    if len(frame_idx) == 0:
        frame_idx = [0]
    mask[..., frame_idx] = 1.0
    return mask


def build_constraints_from_task(task_module, shape, length: int, device) -> List[Constraint]:
    """
    Load constraint list from a task_configs_mic module.

    Expected API on task module:
      build_mic_constraints(shape, length, device) -> List[Constraint]
    """
    if hasattr(task_module, "build_mic_constraints"):
        cons = task_module.build_mic_constraints(shape, length, device)
    else:
        raise AttributeError(
            f"{task_module} must define build_mic_constraints(shape, length, device)"
        )
    for c in cons:
        c.ensure_mask(shape, length, device)
    return cons
