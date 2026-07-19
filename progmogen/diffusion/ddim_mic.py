"""
MIC-enabled diffusion class.

Inherits ProgMoGen `InpaintingGaussianDiffusion` (noise-opt warm-start helpers)
and exposes `ddim_sample_loop_mic` as the main sampling entry.
"""

from typing import List, Optional, Sequence

import torch as th

from diffusion.ddim import InpaintingGaussianDiffusion
from mic.config import MICConfig, default_mic_config
from mic.constraints import Constraint, build_constraints_from_task
from mic.sample_loop_mic import run_mic_sample_loop


class InpaintingGaussianDiffusionMIC(InpaintingGaussianDiffusion):
    """ProgMoGen DDIM + MIC step-wise control sampling."""

    def ddim_sample_loop_mic(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
        ref_motions=None,
        constraints: Optional[Sequence[Constraint]] = None,
        mic_cfg: Optional[MICConfig] = None,
        task_module=None,
    ):
        """
        MIC sampling. Signature mirrors `ddim_sample_loop_opt_fn` so it can be
        swapped in eval entry points with minimal changes.
        """
        if dump_steps is not None:
            raise NotImplementedError("dump_steps not supported in MIC loop")
        if const_noise:
            raise NotImplementedError("const_noise not supported in MIC loop")
        if skip_timesteps:
            raise NotImplementedError("skip_timesteps not supported in MIC skeleton")

        if mic_cfg is None:
            mic_cfg = getattr(self, "mic_cfg", None) or default_mic_config()
        if progress:
            mic_cfg.progress = True

        if constraints is None:
            if task_module is None:
                task_module = getattr(self, "mic_task_module", None)
            if task_module is None:
                raise ValueError(
                    "ddim_sample_loop_mic requires `constraints` or "
                    "`task_module` with build_mic_constraints()"
                )
            length = model_kwargs["y"]["lengths"].item()
            constraints = build_constraints_from_task(
                task_module, shape, length, next(model.parameters()).device
            )

        return run_mic_sample_loop(
            self,
            model,
            shape,
            model_kwargs,
            constraints,
            mic_cfg,
            init_image=init_image,
            clip_denoised=clip_denoised,
            noise=noise,
        )
