"""
MIC overall inference loop (paper Sec. 3.4).

At each denoising step t:
  (i)   compute u_{k,t} via Eq.9 (criterion) or Eq.10 (objective)
  (ii)  update weights via feedback regulator Eq.11
  (iii) allocate applied control via Eq.12–13
  (iv)  inject u_t into DDIM dynamics
"""

from typing import List, Optional, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from .cem import CEMState
from .config import MICConfig
from .constraints import Constraint, ConstraintType
from .control_allocator import allocate, average_controls, weighted_sum_controls
from .control_laws import criterion_control_eq9, objective_control_eq10
from .feedback_regulator import FeedbackRegulator
from .inject import ddim_step_with_control
from .tweedie import get_pred_xstart
from .warm_start import prog_mogen_goal_relaxed_warm_start, prog_mogen_warm_start


def run_mic_sample_loop(
    diffusion,
    model,
    shape,
    model_kwargs,
    constraints: Sequence[Constraint],
    mic_cfg: MICConfig,
    *,
    init_image=None,
    clip_denoised: bool = False,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = next(model.parameters()).device
    K = len(constraints)
    assert K >= 1, "MIC requires at least one constraint"

    if not hasattr(diffusion, "d_mean") or getattr(diffusion, "d_mean", None) is None:
        diffusion.load_inv_normalization_data(device)

    diffusion.n_noise = 0
    length = model_kwargs["y"]["lengths"].item()
    diffusion.length = length

    # ensure masks
    for c in constraints:
        c.ensure_mask(shape, length, device)

    target_relaxed = None
    pred_res_0 = None

    # ---- warm-start ----
    if mic_cfg.warm_start:
        mode = getattr(mic_cfg, "warm_start_mode", "noise_opt") or "noise_opt"
        if mode == "goal_relaxed":
            if mic_cfg.verbose:
                print("[MIC] warm-start mode=goal_relaxed (ProgMoGen multi-epoch)")
            noise_init, _, pred_res_0, target_relaxed = prog_mogen_goal_relaxed_warm_start(
                diffusion,
                model,
                shape,
                model_kwargs,
                init_image=init_image,
                verbose=mic_cfg.verbose,
            )
            # MIC objective / success use the relaxed goal; hard target after transform
            diffusion.target_mic = target_relaxed
            x = noise_init
        else:
            ws_lr = mic_cfg.warm_start_lr
            if ws_lr is None:
                ws_lr = getattr(diffusion, "lr", 0.005)
            ws_it = mic_cfg.warm_start_iterations
            if ws_it is None:
                ws_it = max(1, int(getattr(diffusion, "iterations", 50) // 2))
            if mic_cfg.verbose:
                print(f"[MIC] warm-start mode=noise_opt: lr={ws_lr}, iterations={ws_it}")
            noise_init, _ = prog_mogen_warm_start(
                diffusion,
                model,
                shape,
                model_kwargs,
                lr=ws_lr,
                iterations=ws_it,
                decay_steps=getattr(diffusion, "decay_steps", None),
                init_image=init_image,
                clip_denoised=clip_denoised,
                use_goal=hasattr(diffusion, "target_gt"),
                verbose=mic_cfg.verbose,
            )
            x = noise_init
    else:
        if noise is not None:
            x = noise.to(device)
        else:
            rng = np.random.default_rng(getattr(diffusion, "np_seed", 0))
            x = torch.FloatTensor(rng.standard_normal(size=shape)).to(device)

    # ---- coordination state ----
    regulator = FeedbackRegulator(
        K=K,
        gamma=mic_cfg.gamma,
        W_max=mic_cfg.W_max,
        W_init=mic_cfg.W_init,
        ema_alpha=mic_cfg.ema_alpha,
        device=device,
    )
    cem_states = {}
    for k, c in enumerate(constraints):
        if c.is_criterion:
            cem_states[k] = CEMState(
                shape=shape,
                device=device,
                eps=mic_cfg.cem_eps,
                momentum=mic_cfg.cem_momentum,
            )

    indices = list(range(diffusion.num_timesteps))[::-1]
    if mic_cfg.progress:
        indices = tqdm(indices, desc="MIC denoising")

    for i in indices:
        t = torch.tensor([i] * shape[0], device=device)
        with torch.no_grad():
            x0_hat, out = get_pred_xstart(
                diffusion, model, x, t, model_kwargs=model_kwargs, clip_denoised=clip_denoised
            )

        # per-constraint controls
        u_list = []
        active_constraints = list(constraints)
        if mic_cfg.ablation == "objective_only":
            active_idx = [k for k, c in enumerate(constraints) if c.is_objective]
        elif mic_cfg.ablation == "criterion_only":
            active_idx = [k for k, c in enumerate(constraints) if c.is_criterion]
        else:
            active_idx = list(range(K))

        # Always keep K slots for regulator; inactive → zero control
        for k, c in enumerate(constraints):
            if k not in active_idx:
                u_list.append(torch.zeros_like(x))
                continue
            if c.is_criterion:
                u_k = criterion_control_eq9(
                    diffusion,
                    model,
                    x,
                    t,
                    x0_hat.detach(),
                    c,
                    cem_states[k],
                    M=mic_cfg.M,
                    elite_ratio=mic_cfg.elite_ratio,
                    model_kwargs=model_kwargs,
                )
            else:
                u_k = objective_control_eq10(
                    diffusion,
                    x,
                    t,
                    x0_hat,
                    c,
                    guidance_scale=1.0,
                )
            u_list.append(u_k)

        # regulator
        if mic_cfg.ablation == "no_regulation":
            W = torch.ones(K, device=device)
        else:
            W = regulator.update(x0_hat.detach(), constraints, diffusion=diffusion)

        # allocator
        masks = [c.mask for c in constraints]
        if mic_cfg.ablation == "no_coordination":
            u = average_controls(u_list)
        elif mic_cfg.ablation == "no_allocation":
            u = weighted_sum_controls(u_list, W)
        else:
            u = allocate(u_list, W, masks, lam=mic_cfg.lambda_)

        # inject
        with torch.no_grad():
            x = ddim_step_with_control(
                diffusion,
                model,
                x,
                t,
                out,
                u,
                model_kwargs=model_kwargs,
                eta=mic_cfg.eta,
                guidance_scale=mic_cfg.guidance_scale,
                inject_mode="condition_score",
            )

    # GEO/HOI: rigid map from relaxed goal → hard target (ProgMoGen transform_sample)
    do_post = bool(getattr(mic_cfg, "post_transform", False))
    if do_post and target_relaxed is not None and hasattr(diffusion, "transform_sample"):
        with torch.no_grad():
            # refresh relaxed plane/trajectory from final MIC sample (matches ProgMoGen)
            if hasattr(diffusion, "update_goal"):
                target_relaxed = diffusion.update_goal(x, diffusion.target_gt, target_relaxed, None)
            x_out = diffusion.transform_sample(x, target_relaxed, diffusion.target_gt)
            if mic_cfg.verbose:
                print(f"[MIC] post_transform: {tuple(x.shape)} → {tuple(x_out.shape)}")
            x = x_out

    # final metric via f_eval if available
    if hasattr(diffusion, "f_eval"):
        with torch.no_grad():
            target = getattr(diffusion, "target_gt", None)
            loss_val = None
            xz = bool(do_post and x.shape[1] == 265)
            if xz:
                try:
                    loss_val = diffusion.f_eval(
                        x, pred_res_0 if pred_res_0 is not None else x, target, True
                    )
                except TypeError:
                    loss_val = None
            if loss_val is None:
                for call in (
                    (lambda: diffusion.f_eval(x, x, target, False)),
                    (lambda: diffusion.f_eval(x, x, target)),
                    (lambda: diffusion.f_eval(x, x)),
                    (lambda: diffusion.f_eval(x)),
                ):
                    try:
                        loss_val = call()
                        break
                    except TypeError:
                        continue
            if loss_val is None:
                diffusion.loss_ret_val = np.zeros((1,), dtype=np.float32)
            elif torch.is_tensor(loss_val):
                diffusion.loss_ret_val = loss_val.detach().cpu().numpy()
            else:
                diffusion.loss_ret_val = np.array([float(loss_val)])
    else:
        diffusion.loss_ret_val = np.zeros((1,), dtype=np.float32)

    if hasattr(diffusion, "target_mic"):
        delattr(diffusion, "target_mic")

    if mic_cfg.verbose:
        print(f"[MIC] done. loss_ret_val mean={float(np.mean(diffusion.loss_ret_val)):.6f}")
        print(f"[MIC] final W={regulator.W.detach().cpu().numpy()}")

    return x.detach()
