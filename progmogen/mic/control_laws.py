"""
Instantiate per-constraint control signals.

  Eq.10 — continuous objective-based (gradient on Tweedie x0)
  Eq.9  — criterion-based (importance sampling + CEM)
"""

from typing import Optional

import torch

from .cem import CEMState, standard_normal_log_prob, stable_softmax
from .constraints import Constraint
from .tweedie import perturb_x0_with_noise, sigma_of_t


def objective_control_eq10(
    diffusion,
    x_t: torch.Tensor,
    t: torch.Tensor,
    x0_hat: torch.Tensor,
    constraint: Constraint,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """
    u_t dt ≈ σ(t) ∇_z log p(d | ẑ_T) dt
            ≈ -σ(t) ∇_{ẑ_T} v_d(ẑ_T)
    """
    x0 = x0_hat.detach().requires_grad_(True)
    # Prefer energy that backprops through latent; task may decode inside energy_fn.
    loss = constraint.energy(x0, diffusion=diffusion)
    if loss.numel() != 1:
        loss = loss.mean()
    grad = torch.autograd.grad(loss, x0, retain_graph=False, create_graph=False)[0]
    sigma = sigma_of_t(diffusion, t, x0.shape)
    u = -sigma * grad
    return (guidance_scale * u).detach()


@torch.no_grad()
def criterion_control_eq9(
    diffusion,
    model,
    x_t: torch.Tensor,
    t: torch.Tensor,
    x0_hat: torch.Tensor,
    constraint: Constraint,
    cem: CEMState,
    M: int = 16,
    elite_ratio: float = 0.2,
    model_kwargs=None,
) -> torch.Tensor:
    """
    u_t dt ≈ Σ_m π^{(m)} dε^{(m)}
    π̃ ∝ exp(-E(ẑ_T^{(m)})) · p0(dε)/q(dε)
    """
    d_eps = cem.sample(M)  # [M, *shape]
    # Match single-sample shape [1,C,1,T]
    assert d_eps.shape[1:] == x_t.shape

    log_w_list = []
    for m in range(M):
        x0_m = perturb_x0_with_noise(diffusion, x_t, t, x0_hat, d_eps[m])
        E_m = constraint.energy(x0_m, diffusion=diffusion)
        E_m = E_m.reshape(()).float()
        # importance weight in log space
        # log π̃ = -E + log p0 - log q
        lp0 = standard_normal_log_prob(d_eps[m : m + 1]).reshape(())
        lq = cem.log_prob(d_eps[m : m + 1]).reshape(())
        log_w_list.append(-E_m + lp0 - lq)

    log_w = torch.stack(log_w_list, dim=0)
    pi = stable_softmax(log_w)  # [M]
    u = (pi.view(M, *([1] * (d_eps.dim() - 1))) * d_eps).sum(dim=0)

    cem.update_elite(d_eps, pi, elite_ratio=elite_ratio)
    return u
