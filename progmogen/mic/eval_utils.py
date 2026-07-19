"""Helpers shared by MIC evaluation entry scripts."""

from __future__ import annotations

from mic.config import default_mic_config


def renorm(data, dataset):
    """
    Remap motion features from train norm to eval norm.
    Must use the `dataset` argument (do not rely on a gen_loader global).
    """
    mean = dataset.mean[None, None, :]
    std = dataset.std[None, None, :]
    mean_for_eval = dataset.mean_for_eval[None, None, :]
    std_for_eval = dataset.std_for_eval[None, None, :]
    data = data * std + mean
    data = (data - mean_for_eval) / std_for_eval
    return data


def load_mic_cfg(task_module, args):
    overrides = {}
    if hasattr(task_module, "MIC_GAMMA"):
        overrides["gamma"] = task_module.MIC_GAMMA
    if hasattr(task_module, "MIC_LAMBDA"):
        overrides["lambda_"] = task_module.MIC_LAMBDA
    if hasattr(task_module, "MIC_W_MAX"):
        overrides["W_max"] = task_module.MIC_W_MAX
    if hasattr(task_module, "MIC_WARM_START_ITERATIONS"):
        overrides["warm_start_iterations"] = task_module.MIC_WARM_START_ITERATIONS
    if hasattr(task_module, "MIC_WARM_START_MODE"):
        overrides["warm_start_mode"] = task_module.MIC_WARM_START_MODE
    if hasattr(task_module, "MIC_POST_TRANSFORM"):
        overrides["post_transform"] = bool(task_module.MIC_POST_TRANSFORM)
    if getattr(args, "mic_ablation", None):
        overrides["ablation"] = args.mic_ablation
    if getattr(args, "mic_no_warm_start", False):
        overrides["warm_start"] = False
    return default_mic_config(**overrides)


def bind_task_hooks(diffusion, task_config_name, import_class, *, goal_relaxed=False):
    import types

    f_loss = import_class(f"{task_config_name}.f_loss")
    f_eval = import_class(f"{task_config_name}.f_eval")
    diffusion.f_loss = types.MethodType(f_loss, diffusion)
    diffusion.f_eval = types.MethodType(f_eval, diffusion)

    if goal_relaxed:
        for name in ("update_goal", "transform_sample", "get_lr_schedule", "f_sample_random_plane"):
            try:
                fn = import_class(f"{task_config_name}.{name}")
                setattr(diffusion, name, types.MethodType(fn, diffusion))
            except AttributeError:
                pass

    diffusion.lr = import_class(f"{task_config_name}.lr")
    diffusion.iterations = import_class(f"{task_config_name}.iterations")
    try:
        diffusion.decay_steps = import_class(f"{task_config_name}.decay_steps")
    except AttributeError:
        diffusion.decay_steps = None
    try:
        diffusion.epoch_relax = import_class(f"{task_config_name}.epoch_relax")
    except AttributeError:
        pass
