"""MIC evaluation entry for GEO-1 / HOI-1 (goal + optional relax hooks)."""

import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../task_configs_eval"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../task_configs_mic"))

from eval_task import (
    f_add_args,
    get_slice_model_kwargs,
    import_class,
    save_to_npy_with_motion_gen,
)
# GEO/HOI return 265-dim (263 + XZ offset); need XZ-aware DataTransform
from eval_task_goal_relaxed import DataTransform
from config_data import EVAL_SAMPLE32_FILE_NAME, EVAL_HOI1_FILE_NAME
from utils.parser_util import evaluation_inpainting_parser_add_args
from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util_v2 import load_model_blending_and_diffusion
from data_loaders.get_data import get_dataset_loader, pad_or_trim_to_batch_size
from data_loaders.humanml_utils import get_inpainting_mask
from diffusion import logger
from diffusion.respace import SpacedDiffusion
from mic.eval_utils import bind_task_hooks, load_mic_cfg, renorm


def f_add_mic_args(parser):
    parser = f_add_args(parser)
    parser.add_argument("--eval_task", default="none", type=str, help="geo1|hoi1")
    parser.add_argument("--mic_ablation", default="none", type=str)
    parser.add_argument("--mic_no_warm_start", action="store_true")
    return parser


def get_gen_motion_mic(args, model, diffusion, dataloader, num_samples_limit, scale):
    clip_denoised = False
    real_num_batches = len(dataloader)
    if num_samples_limit is not None:
        real_num_batches = num_samples_limit // dataloader.batch_size + 1
    print("real_num_batches", real_num_batches)

    generated_motion, loss_list, length_list, text_list = [], [], [], []
    constraint_list = []
    caption_list, tokens_list, cap_len_list = [], [], []

    model.eval()
    for v in model.parameters():
        v.requires_grad = False

    task_module = import_class(args.task_config)
    mic_cfg = load_mic_cfg(task_module, args)
    diffusion.mic_cfg = mic_cfg
    diffusion.mic_task_module = task_module

    eval_task = args.eval_task
    assert eval_task in ["hoi1", "geo1"]

    for i, (motion, model_kwargs) in enumerate(dataloader):
        if num_samples_limit is not None and len(generated_motion) >= real_num_batches:
            break

        motion, model_kwargs = pad_or_trim_to_batch_size(
            motion, model_kwargs, dataloader.batch_size
        )

        if eval_task == "geo1":
            ref_n32_data = np.load(EVAL_SAMPLE32_FILE_NAME, allow_pickle=True)
            model_kwargs["y"]["text"] = [e[0] for e in ref_n32_data]
            model_kwargs["y"]["tokens"] = [e[1] for e in ref_n32_data]
            model_kwargs["y"]["lengths"] = torch.LongTensor([int(e[2]) for e in ref_n32_data])
            tokens = [t.split("_") for t in model_kwargs["y"]["tokens"]]
        else:
            ref_n32_data = np.load(EVAL_HOI1_FILE_NAME, allow_pickle=True)
            model_kwargs["y"]["text"] = [e[0] for e in ref_n32_data]
            model_kwargs["y"]["tokens"] = [e[1] for e in ref_n32_data]
            model_kwargs["y"]["lengths"] = torch.LongTensor([int(e[2]) for e in ref_n32_data])
            tokens = [[] for _ in model_kwargs["y"]["tokens"]]
            target_list_ref = np.array([e[3] for e in ref_n32_data])
            model_kwargs["y"]["target_list"] = target_list_ref

        if scale != 1.0:
            model_kwargs["y"]["scale"] = torch.ones(motion.shape[0], device=dist_util.dev()) * scale

        model_kwargs["y"]["inpainted_motion"] = motion.to(dist_util.dev())
        model_kwargs["y"]["inpainting_mask"] = torch.tensor(
            get_inpainting_mask(args.inpainting_mask, motion.shape)
        ).float().to(dist_util.dev())

        diffusion.load_inv_normalization_data(dist_util.dev())
        bind_task_hooks(diffusion, args.task_config, import_class, goal_relaxed=True)

        sample, loss, constraint = [], [], []
        demo_num = getattr(task_module, "DEMO_NUM", 32)
        bs = motion.shape[0]

        for ii in range(bs):
            print(ii, bs)
            diffusion.np_seed = np.random.randint(0, 1000) + 1
            model_kwargs_each = get_slice_model_kwargs(model_kwargs, ii)
            diffusion.length = model_kwargs_each["y"]["lengths"].item()

            if eval_task == "hoi1":
                target_gt_list = model_kwargs_each["y"]["target_list"]
                target_gt = torch.FloatTensor(target_gt_list).to(dist_util.dev())
            else:
                target_gt_list = diffusion.f_sample_random_plane(r_range=3, seed=diffusion.np_seed)
                target_gt = torch.FloatTensor(target_gt_list).to(dist_util.dev())
            diffusion.target_gt = target_gt

            sample_each = diffusion.ddim_sample_loop_mic(
                model,
                motion[ii : ii + 1].shape,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs_each,
                skip_timesteps=0,
                init_image=None,
                progress=True,
                task_module=task_module,
                mic_cfg=mic_cfg,
            )
            sample.append(sample_each)
            loss.append(diffusion.loss_ret_val)
            constraint.append(np.array(target_gt_list).reshape(-1).tolist())
            print("-->loss_ret_val each = ", np.mean(diffusion.loss_ret_val))
            if demo_num is not None and len(sample) >= demo_num:
                break

        sample = torch.cat(sample, 0)
        lengths = model_kwargs["y"]["lengths"]
        texts = model_kwargs["y"]["text"]
        if demo_num is not None:
            sample = sample[:demo_num]
            lengths = lengths[:demo_num]
            texts = texts[:demo_num]

        generated_motion.append(sample.data.cpu().detach())
        length_list.append(lengths.data.cpu().detach())
        text_list += texts
        loss_list.append(np.array(loss))
        constraint_list.append(np.array(constraint))
        caption_list += texts
        tokens_list += tokens[: len(texts)]
        cap_len_list += [len(tokens[j]) if j < len(tokens) else 0 for j in range(len(texts))]
        break

    generated_motion = torch.cat(generated_motion, 0)
    length_list = torch.cat(length_list, 0)
    loss_list = np.concatenate(loss_list, 0)
    constraint_list = np.concatenate(constraint_list, 0)
    return (
        [generated_motion, loss_list, constraint_list],
        length_list,
        text_list,
        [caption_list, tokens_list, cap_len_list],
    )


def main():
    args_list = evaluation_inpainting_parser_add_args(f_add_mic_args)
    args = args_list[0]
    fixseed(args.seed)
    args.batch_size = 32

    assert args.use_ddim_tag in [0, 1]
    mask_type = args.mask_type
    assert mask_type in ["root_horizontal", "left_wrist"]
    args_list[0].inpainting_mask = mask_type
    args.inpainting_mask = mask_type

    dist_util.setup_dist(args.device)
    logger.configure()
    logger.log("creating data loader...")
    split = args.text_split
    gen_loader = get_dataset_loader(
        name=args.dataset, batch_size=args.batch_size, num_frames=None,
        split=split, load_mode="eval", drop_last=False,
    )

    logger.log("Creating MIC relax model and diffusion...")
    from diffusion.ddim_relax_mic import InpaintingGaussianDiffusionRelaxMIC

    DiffusionClass = (
        InpaintingGaussianDiffusionRelaxMIC if args.filter_noise else SpacedDiffusion
    )
    model, diffusion = load_model_blending_and_diffusion(
        args_list, gen_loader, dist_util.dev(), DiffusionClass=DiffusionClass
    )

    data_transform = DataTransform(device="cpu")
    motion_gen_all, length_gen, texts_gen, _ = get_gen_motion_mic(
        args, model, diffusion, gen_loader, args.num_samples_limit, args.guidance_param
    )
    motion_gen, loss_head_gen, constraint_gen = motion_gen_all

    if args.ret_type == "pos":
        if motion_gen.shape[1] == 265:
            motion_gen_joints = data_transform.sample_to_joints_with_XZ_offset(motion_gen)
            motion_gen = motion_gen[:, :263, :, :]
        else:
            motion_gen_joints = data_transform.sample_to_joints(motion_gen)
    else:
        motion_gen_joints = data_transform.sample_to_joints_from_rot(motion_gen)
    motion_gen_joints_copy = motion_gen_joints.detach().clone()

    motion_gen = motion_gen.squeeze(2).permute(0, 2, 1).contiguous()
    motion_gen = renorm(motion_gen, gen_loader.dataset)

    os.makedirs(args.save_fig_dir, exist_ok=True)
    save_npy_path = os.path.join(args.save_fig_dir, "gen.npy")
    save_to_npy_with_motion_gen(
        save_npy_path,
        all_motions=motion_gen_joints_copy.data.cpu().numpy(),
        all_text=list(texts_gen),
        all_lengths=length_gen.data.cpu().numpy(),
        fid=None,
        motion_gen=motion_gen.data.cpu().numpy(),
        loss=loss_head_gen,
        constraint=constraint_gen,
    )
    print(f"[MIC {args.eval_task}] saved → {save_npy_path}")


if __name__ == "__main__":
    main()
