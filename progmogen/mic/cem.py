"""Cross-Entropy Method for criterion control proposal q (paper Eq.9)."""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class CEMState:
    """
    Diagonal-Gaussian proposal q = N(μ, diag(σ²)).
    Updated each denoising step from an elite subset (elite_ratio=20%).
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        eps: float = 1e-4,
        momentum: float = 0.5,
    ):
        self.shape = shape
        self.device = device
        self.eps = eps
        self.momentum = momentum
        self.mu = torch.zeros(shape, device=device)
        self.log_std = torch.zeros(shape, device=device)  # σ=1 initially

    @property
    def std(self) -> torch.Tensor:
        return self.log_std.exp().clamp_min(self.eps)

    def sample(self, M: int) -> torch.Tensor:
        """Draw M noise increments; returns [M, *shape]."""
        eps = torch.randn(M, *self.shape, device=self.device)
        return self.mu.unsqueeze(0) + self.std.unsqueeze(0) * eps

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [M, *shape] → log q(x) summed over all dims except batch → [M]
        """
        var = (self.std ** 2).unsqueeze(0)
        mu = self.mu.unsqueeze(0)
        log_p = -0.5 * (
            math.log(2.0 * math.pi) + torch.log(var) + (x - mu) ** 2 / var
        )
        return log_p.flatten(1).sum(dim=1)

    def update_elite(
        self,
        samples: torch.Tensor,
        weights: torch.Tensor,
        elite_ratio: float = 0.2,
    ):
        """
        samples: [M, *shape]
        weights: [M] importance weights π (already normalized) or scores
        """
        M = samples.shape[0]
        n_elite = max(1, int(round(M * elite_ratio)))
        # select by highest weight
        _, idx = torch.topk(weights, k=n_elite, largest=True)
        elite = samples[idx]  # [n_elite, *shape]

        mu_new = elite.mean(dim=0)
        std_new = elite.std(dim=0, unbiased=False).clamp_min(self.eps)
        log_std_new = std_new.log()

        m = self.momentum
        self.mu = m * self.mu + (1.0 - m) * mu_new
        self.log_std = m * self.log_std + (1.0 - m) * log_std_new


def standard_normal_log_prob(x: torch.Tensor) -> torch.Tensor:
    """log p0(x) for N(0,I); x [M,*] → [M]."""
    # math.pi: torch.pi unavailable on PyTorch < 1.8
    log_2pi = math.log(2.0 * math.pi)
    log_p = -0.5 * (log_2pi + x ** 2)
    return log_p.flatten(1).sum(dim=1)


def stable_softmax(log_w: torch.Tensor) -> torch.Tensor:
    return F.softmax(log_w, dim=0)
