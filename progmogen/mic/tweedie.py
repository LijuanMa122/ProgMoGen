"""Tweedie clean-motion estimate helpers (paper Eq.9–10)."""

from typing import Optional

import torch

from diffusion.gaussian_diffusion_v2 import _extract_into_tensor


def get_pred_xstart(diffusion, model, x, t, model_kwargs=None, clip_denoised=False):
    """ẑ_T ≈ pred_xstart from MDM (x0-parameterization)."""
    out = diffusion.p_mean_variance(
        model,
        x,
        t,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs,
    )
    return out["pred_xstart"], out


def sigma_of_t(diffusion, t, shape) -> torch.Tensor:
    """
    Discrete stand-in for σ(t)=√β(T-t) in paper Eq.4/8/10.
    Uses √(1-α̅_t) which matches score-guidance scale in condition_score.
    """
    return _extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod, t, shape)


def beta_of_t(diffusion, t, shape) -> torch.Tensor:
    return _extract_into_tensor(diffusion.betas, t, shape)


def perturb_x0_with_noise(
    diffusion,
    x_t: torch.Tensor,
    t: torch.Tensor,
    x0_hat: torch.Tensor,
    d_eps: torch.Tensor,
) -> torch.Tensor:
    """
    Construct a candidate clean estimate for criterion importance sampling.

    Skeleton: re-estimate x0 after a score-like noise perturbation on x_t,
    using the closed-form Tweedie relation with a proposed noise increment.

    For MDM x0-pred models a practical surrogate (aligned with path-integral
    IS) is to evaluate the terminal cost on:
        x0_m = x0_hat + scale * d_eps
    or decode from a one-step mean shifted by d_eps.

    Here we use: ẑ_T^{(m)} = x0_hat + √(1-α̅_t) * d_eps  (broadcast-safe).
    Refine later if matching a specific supplementary discretization.
    """
    scale = sigma_of_t(diffusion, t, x0_hat.shape)
    return x0_hat + scale * d_eps
