"""Inject applied control u_t into one DDIM step (paper Eq.3)."""

from typing import Optional

import torch

from diffusion.gaussian_diffusion_v2 import _extract_into_tensor
from .tweedie import sigma_of_t


def h_from_u(u: torch.Tensor, diffusion, t: torch.Tensor) -> torch.Tensor:
    """
    Paper: u_t = √β(T-t) · h_{T-t}  ⇒  h = u / σ(t).
    Use √(1-α̅_t) as discrete σ.
    """
    sigma = sigma_of_t(diffusion, t, u.shape).clamp_min(1e-8)
    return u / sigma


def ddim_step_with_control(
    diffusion,
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    out_orig: dict,
    u: torch.Tensor,
    model_kwargs=None,
    eta: float = 0.0,
    guidance_scale: float = 1.0,
    inject_mode: str = "condition_score",
) -> torch.Tensor:
    """
    One controlled DDIM step x_t → x_{t-1}.

    inject_mode:
      - "condition_score": Scheme A — reuse condition_score with h = u/σ
      - "mean_add": Scheme B — x_{t-1} = mean_pred + scale * u
    """
    u = guidance_scale * u

    if inject_mode == "condition_score":
        h = h_from_u(u, diffusion, t)

        def cond_fn(x_in, t_in, **kwargs):
            return h

        out = diffusion.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
    elif inject_mode == "mean_add":
        out = out_orig
    else:
        raise ValueError(f"Unknown inject_mode: {inject_mode}")

    eps = diffusion._predict_eps_from_xstart(x, t, out["pred_xstart"])
    alpha_bar = _extract_into_tensor(diffusion.alphas_cumprod, t, x.shape)
    alpha_bar_prev = _extract_into_tensor(diffusion.alphas_cumprod_prev, t, x.shape)
    sigma = (
        eta
        * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
        * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
    )
    mean_pred = (
        out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
        + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
    )

    if inject_mode == "mean_add":
        # Scheme B: add control on the mean (scale by √(1-α̅_{t-1}) for magnitude)
        mean_pred = mean_pred + torch.sqrt(1 - alpha_bar_prev).clamp_min(0.0) * u

    nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
    noise = torch.zeros_like(x) if eta == 0.0 else torch.randn_like(x)
    sample = mean_pred + nonzero_mask * sigma * noise
    return sample
