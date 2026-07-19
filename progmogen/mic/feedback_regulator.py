"""Feedback regulator: integral weight update (paper Eq.11)."""

from typing import Sequence

import torch

from .constraints import Constraint


class FeedbackRegulator:
    """
    W_{k,t+1} = Π_{[0, W_max]}( W_{k,t} + γ · c_{k,t} )

    where c̃_{k,t} = max(0, v_{d_k}(ẑ_T)) and c_{k,t} is EMA-normalized.
    """

    def __init__(
        self,
        K: int,
        gamma: float = 0.1,
        W_max: float = 10.0,
        W_init: float = 0.0,
        ema_alpha: float = 0.95,
        eps: float = 1e-6,
        device: torch.device = None,
    ):
        self.K = K
        self.gamma = gamma
        self.W_max = W_max
        self.W_init = float(W_init)
        self.ema_alpha = ema_alpha
        self.eps = eps
        device = device or torch.device("cpu")
        self.W = torch.full((K,), self.W_init, device=device)
        self.scale = torch.ones(K, device=device)

    def to(self, device: torch.device):
        self.W = self.W.to(device)
        self.scale = self.scale.to(device)
        return self

    def reset(self, device: torch.device = None):
        if device is not None:
            self.to(device)
        self.W.fill_(self.W_init)
        self.scale.fill_(1.0)

    @torch.no_grad()
    def update(
        self,
        x0_hat: torch.Tensor,
        constraints: Sequence[Constraint],
        diffusion=None,
    ) -> torch.Tensor:
        """Evaluate violations on Tweedie clean estimate and update weights."""
        assert len(constraints) == self.K
        for k, c in enumerate(constraints):
            c_tilde = c.violation(x0_hat, diffusion=diffusion)
            if not torch.is_tensor(c_tilde):
                c_tilde = torch.tensor(float(c_tilde), device=self.W.device)
            else:
                c_tilde = c_tilde.detach().to(self.W.device).reshape(())
            c_tilde = torch.clamp(c_tilde, min=0.0)

            self.scale[k] = (
                self.ema_alpha * self.scale[k]
                + (1.0 - self.ema_alpha) * torch.clamp(c_tilde, min=self.eps)
            )
            c_norm = c_tilde / (self.scale[k] + self.eps)
            self.W[k] = torch.clamp(self.W[k] + self.gamma * c_norm, 0.0, self.W_max)
        return self.W.clone()
