"""Control allocator: weighted least-squares over scopes (paper Eq.12–13)."""

from typing import List, Optional, Sequence

import torch


def allocate(
    u_list: Sequence[torch.Tensor],
    W: torch.Tensor,
    masks: Sequence[Optional[torch.Tensor]],
    lam: float = 1.0,
) -> torch.Tensor:
    """
    Closed-form elementwise solution of Eq.13 when M_k are diagonal masks:

        u_i = (Σ_k W_k² M_{k,i} u_{k,i}) / (λ + Σ_k W_k² M_{k,i})

    Args:
        u_list: K control tensors, same shape as motion latent (e.g. [1,263,1,T])
        W: [K] weights from feedback regulator
        masks: K tensors broadcastable to u; None → all-ones
        lam: λ > 0 regularizer on ||u||²
    """
    assert len(u_list) == len(W) == len(masks)
    assert len(u_list) >= 1

    ref = u_list[0]
    numer = torch.zeros_like(ref)
    denom = torch.full_like(ref, float(lam))

    for uk, Wk, Mk in zip(u_list, W, masks):
        if Mk is None:
            Mk = torch.ones_like(uk)
        else:
            Mk = Mk.to(uk.device).to(uk.dtype)
            if Mk.shape != uk.shape:
                Mk = torch.broadcast_to(Mk, uk.shape)
        w2 = float(Wk.item() if torch.is_tensor(Wk) else Wk) ** 2
        numer = numer + w2 * Mk * uk
        denom = denom + w2 * Mk

    return numer / denom.clamp_min(1e-12)


def average_controls(u_list: Sequence[torch.Tensor]) -> torch.Tensor:
    """Ablation: w/o coordination → plain average."""
    return torch.stack(list(u_list), dim=0).mean(dim=0)


def weighted_sum_controls(
    u_list: Sequence[torch.Tensor],
    W: torch.Tensor,
) -> torch.Tensor:
    """Ablation: w/o allocation → Σ_k W_k u_k (no masks / no λ)."""
    out = torch.zeros_like(u_list[0])
    for uk, Wk in zip(u_list, W):
        out = out + float(Wk.item()) * uk
    return out
