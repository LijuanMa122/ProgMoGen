"""
Warm-start via ProgMoGen / DNO-style noise optimization (paper Implementation Details).

Runs a shortened `ddim_sample_loop_opt_fn` (or full goal_relaxed) then returns the
optimized `noise_init` to initialize the MIC reverse loop.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.optim as optim


def prog_mogen_warm_start(
    diffusion,
    model,
    shape,
    model_kwargs,
    *,
    lr: float,
    iterations: int,
    decay_steps=None,
    init_image=None,
    clip_denoised: bool = False,
    use_goal: bool = False,
    verbose: bool = True,
):
    """
    Optimize noise_init for `iterations` steps with existing ProgMoGen loss,
    return (noise_init.detach(), pred_res.detach()).
    """
    device = next(model.parameters()).device
    rng = np.random.default_rng(getattr(diffusion, "np_seed", 0))
    noise_list_npy = [rng.standard_normal(size=shape) for _ in range(diffusion.num_timesteps)]
    noise_init_npy = rng.standard_normal(size=shape)
    noise_list = [torch.FloatTensor(a).to(device) for a in noise_list_npy]
    noise_init = torch.FloatTensor(noise_init_npy).to(device)
    noise_init.requires_grad_(True)

    if not hasattr(diffusion, "d_mean") or diffusion.d_mean is None:
        diffusion.load_inv_normalization_data(device)

    # ddim_sample_known_noise increments this counter
    diffusion.n_noise = 0
    diffusion.length = model_kwargs["y"]["lengths"].item()
    optimizer = optim.Adam([noise_init], lr)
    if decay_steps is None:
        decay_steps = iterations

    pred_res_ret = None
    pred_res_0 = None
    for it in range(iterations):
        diffusion.adjust_learning_rate(optimizer, lr, it, step=decay_steps)
        optimizer.zero_grad()
        pred_res, res_list, pred_x0_list = diffusion.f_forward_return_middle_list(
            model,
            shape,
            noise_list,
            noise_init,
            init_image,
            model_kwargs,
            eta=0.0,
            progress=False,
            clip_denoised=clip_denoised,
        )
        pred_res_ret = pred_res.detach().clone()
        if it == 0:
            pred_res_0 = pred_res.detach().clone()

        if use_goal and hasattr(diffusion, "target_gt"):
            loss = diffusion.f_loss(pred_res, pred_res_0, it, diffusion.target_gt)
        else:
            # match signatures used across task configs (HSI-1/2/3 ...)
            try:
                loss = diffusion.f_loss(pred_res, pred_res_0, it)
            except TypeError:
                try:
                    loss = diffusion.f_loss(pred_res)
                except TypeError:
                    loss = diffusion.f_loss(pred_res, pred_res_0)

        if verbose:
            print(f"[MIC warm-start] it={it}, loss={float(loss.item()):.6f}")
        loss.backward()
        optimizer.step()
        del pred_res, res_list, pred_x0_list

    return noise_init.detach(), pred_res_ret


def prog_mogen_goal_relaxed_warm_start(
    diffusion,
    model,
    shape,
    model_kwargs,
    *,
    init_image=None,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full ProgMoGen GEO/HOI warm-start: epoch_relax × update_goal + Adam(noise_init).

    Does NOT call transform_sample (that happens after MIC). Returns:
      noise_init, pred_res_ret, pred_res_0, target_relaxed
    """
    device = next(model.parameters()).device
    rng = np.random.default_rng(getattr(diffusion, "np_seed", 0))
    noise_list_npy = [rng.standard_normal(size=shape) for _ in range(diffusion.num_timesteps)]
    noise_init_npy = rng.standard_normal(size=shape)
    noise_list = [torch.FloatTensor(a).to(device) for a in noise_list_npy]
    noise_init = torch.FloatTensor(noise_init_npy).to(device)

    if not hasattr(diffusion, "d_mean") or diffusion.d_mean is None:
        diffusion.load_inv_normalization_data(device)

    diffusion.n_noise = 0
    diffusion.length = model_kwargs["y"]["lengths"].item()

    target = diffusion.target_gt
    target_relaxed = None
    epoch_relax = int(getattr(diffusion, "epoch_relax", 1))
    max_it = int(getattr(diffusion, "iterations", 20))
    if verbose:
        print(f"[MIC goal_relaxed warm-start] epoch_relax={epoch_relax}, max_it={max_it}")

    pred_res_ret = None
    pred_res_0 = None

    for i_k in range(epoch_relax):
        # ---- relax: fit soft target from current sample ----
        with torch.no_grad():
            pred_res, res_list, pred_x0_list = diffusion.f_forward_return_middle_list(
                model,
                shape,
                noise_list,
                noise_init,
                init_image,
                model_kwargs,
                eta=0.0,
                progress=False,
            )
            if i_k == 0:
                pred_res_0 = pred_res.detach().clone()
            target_relaxed = diffusion.update_goal(pred_res, target, target_relaxed, i_k)
            pred_res_ret = pred_res.detach().clone()
            del pred_res, res_list, pred_x0_list
            d0 = diffusion.f_eval(pred_res_ret, pred_res_0, target_relaxed)
            if verbose:
                print(
                    f"[MIC goal_relaxed] epoch={i_k} relax d.mean={float(d0.mean()):.6f}"
                )

        noise_init.requires_grad_(True)
        base_lr = diffusion.get_lr_schedule(i_k)
        optimizer = optim.Adam([noise_init], base_lr)

        for it in range(max_it):
            diffusion.adjust_learning_rate(optimizer, base_lr, it, step=max_it)
            optimizer.zero_grad()
            pred_res, res_list, pred_x0_list = diffusion.f_forward_return_middle_list(
                model,
                shape,
                noise_list,
                noise_init,
                init_image,
                model_kwargs,
                eta=0.0,
                progress=False,
            )
            loss = diffusion.f_loss(pred_res, pred_res_0, it, target_relaxed)
            if verbose and (it == 0 or it == max_it - 1 or it % 5 == 0):
                loss_eval = diffusion.f_eval(pred_res, pred_res_0, target_relaxed).mean()
                print(
                    f"[MIC goal_relaxed] epoch={i_k} it={it}, "
                    f"loss={float(loss.item()):.6f}, loss_eval={float(loss_eval.item()):.6f}"
                )
            loss.backward()
            optimizer.step()
            pred_res_ret = pred_res.detach().clone()
            del pred_res, res_list, pred_x0_list

        noise_init.requires_grad_(False)
        with torch.no_grad():
            d = diffusion.f_eval(pred_res_ret, pred_res_0, target_relaxed)
            if verbose:
                print(
                    f"[MIC goal_relaxed] epoch={i_k} last d.mean={float(d.mean()):.6f}"
                )

    assert target_relaxed is not None and pred_res_ret is not None and pred_res_0 is not None
    return noise_init.detach(), pred_res_ret, pred_res_0, target_relaxed.detach()
