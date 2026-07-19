"""
MIC evaluation entry (skeleton).

Usage (from progmogen/):
  python3 tasks/eval_task_mic.py --task_config eval_task_hsi2_mic_config ...

Binds diffusion.ddim_sample_loop_mic instead of ddim_sample_loop_opt_fn.
"""

import os
import sys
import types

import numpy as np
import torch

# paths: mirror eval_task.py
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../task_configs_eval"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../task_configs_mic"))

import eval_task as base
from eval_task import (
    DataTransform,
    EVAL_SAMPLE32_FILE_NAME,
    f_add_args,
    get_slice_model_kwargs,
    import_class,
    save_to_npy_with_motion_gen,
)
from utils.parser_util import evaluation_inpainting_parser_add_args
from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util_v2 import load_model_blending_and_diffusion
from data_loaders.get_data import get_dataset_loader, pad_or_trim_to_batch_size
from data_loaders.humanml_utils import get_inpainting_mask
from diffusion import logger
from diffusion.respace import SpacedDiffusion
from mic.eval_utils import load_mic_cfg, renorm


def get_gen_motion_mic(args, model, diffusion, dataloader, num_samples_limit, scale, init_motion_type):
    clip_denoised = False
    real_num_batches = len(dataloader)
    if num_samples_limit is not None:
        real_num_batches = num_samples_limit // dataloader.batch_size + 1
    print("real_num_batches", real_num_batches)

    generated_motion = []
    loss_list = []
    length_list = []
    text_list = []
    constraint_list = []
    caption_list, tokens_list, cap_len_list = [], [], []

    model.eval()
    for v in model.parameters():
        v.requires_grad = False

    task_module = import_class(args.task_config)
    mic_cfg = load_mic_cfg(task_module, args)
    diffusion.mic_cfg = mic_cfg
    diffusion.mic_task_module = task_module

    for _ in range(1):
        for i, (motion, model_kwargs) in enumerate(dataloader):
            if num_samples_limit is not None and len(generated_motion) >= real_num_batches:
                break

            motion, model_kwargs = pad_or_trim_to_batch_size(
                motion, model_kwargs, dataloader.batch_size
            )

            ref_n32_data = np.load(EVAL_SAMPLE32_FILE_NAME, allow_pickle=True)
            ref_text_prompt_list = [each_sample[0] for each_sample in ref_n32_data]
            ref_tokens_list = [each_sample[1] for each_sample in ref_n32_data]
            ref_length_list = [int(each_sample[2]) for each_sample in ref_n32_data]

            model_kwargs["y"]["text"] = ref_text_prompt_list
            model_kwargs["y"]["tokens"] = ref_tokens_list
            model_kwargs["y"]["lengths"] = torch.LongTensor(ref_length_list)
            tokens = [t.split("_") for t in model_kwargs["y"]["tokens"]]

            if scale != 1.0:
                model_kwargs["y"]["scale"] = (
                    torch.ones(motion.shape[0], device=dist_util.dev()) * scale
                )

            model_kwargs["y"]["inpainted_motion"] = motion.to(dist_util.dev())
            model_kwargs["y"]["inpainting_mask"] = torch.tensor(
                get_inpainting_mask(args.inpainting_mask, motion.shape)
            ).float().to(dist_util.dev())

            diffusion.load_inv_normalization_data(dist_util.dev())

            f_loss = import_class(f"{args.task_config}.f_loss")
            f_eval = import_class(f"{args.task_config}.f_eval")
            diffusion.f_loss = types.MethodType(f_loss, diffusion)
            diffusion.f_eval = types.MethodType(f_eval, diffusion)

            sample_fn = diffusion.ddim_sample_loop_mic

            sample, loss, constraint = [], [], []
            bs = motion.shape[0]
            demo_num = 32

            for ii in range(bs):
                print(f"[MIC] sample {ii}/{bs}")
                diffusion.np_seed = np.random.randint(0, 1000) + 1
                model_kwargs_each = get_slice_model_kwargs(model_kwargs, ii)
                length_each = model_kwargs_each["y"]["lengths"].item()
                diffusion.length = length_each

                diffusion.lr = import_class(f"{args.task_config}.lr")
                diffusion.iterations = import_class(f"{args.task_config}.iterations")
                diffusion.decay_steps = import_class(f"{args.task_config}.decay_steps")

                sample_each = sample_fn(
                    model,
                    motion[ii : ii + 1].shape,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs_each,
                    skip_timesteps=0,
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                    task_module=task_module,
                    mic_cfg=mic_cfg,
                )
                sample.append(sample_each)
                loss.append(diffusion.loss_ret_val)
                constraint.append([0, 0, 0])
                print("-->loss_ret_val each = ", np.mean(diffusion.loss_ret_val))

                if demo_num is not None and len(sample) >= demo_num:
                    break

            sample = torch.cat(sample, 0)
            lengths = model_kwargs["y"]["lengths"]
            texts = model_kwargs["y"]["text"]
            loss = np.array(loss)
            constraint = np.array(constraint)

            if demo_num is not None:
                sample = sample[:demo_num]
                lengths = lengths[:demo_num]
                texts = texts[:demo_num]

            generated_motion.append(sample.data.cpu().detach())
            length_list.append(lengths.data.cpu().detach())
            text_list += texts

            if demo_num is not None:
                caption = model_kwargs["y"]["text"][:demo_num]
                tokens = tokens[:demo_num]
                cap_len = [len(tokens[bs_i]) for bs_i in range(len(tokens))]
            else:
                caption = model_kwargs["y"]["text"]
                cap_len = [len(tokens[bs_i]) for bs_i in range(len(tokens))]

            caption_list += caption
            tokens_list += tokens
            cap_len_list += cap_len
            loss_list.append(loss)
            constraint_list.append(constraint)

            break  # one batch for skeleton / open-set 32

    generated_motion = torch.cat(generated_motion, 0)
    length_list = torch.cat(length_list, 0)
    loss_list = np.concatenate(loss_list, 0)
    constraint_list = np.concatenate(constraint_list, 0)
    assert len(loss_list) == len(length_list)
    return (
        [generated_motion, loss_list, constraint_list],
        length_list,
        text_list,
        [caption_list, tokens_list, cap_len_list],
    )


def f_add_mic_args(parser):
    parser = f_add_args(parser)
    parser.add_argument("--mic_ablation", default="none", type=str, help="MIC ablation mode")
    parser.add_argument("--mic_no_warm_start", action="store_true", help="disable ProgMoGen warm-start")
    return parser


def main():
    args_list = evaluation_inpainting_parser_add_args(f_add_mic_args)
    args = args_list[0]
    fixseed(args.seed)
    args.batch_size = 32

    if args.use_ddim_tag == 1:
        use_ddim_tag = True
    elif args.use_ddim_tag == 0:
        use_ddim_tag = False
    else:
        raise ValueError()

    mask_type = args.mask_type
    assert mask_type in ["root_horizontal", "left_wrist"]
    args_list[0].inpainting_mask = mask_type
    args.inpainting_mask = mask_type

    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    log_file = os.path.join(
        os.path.dirname(args.model_path),
        f"debug_mic_ddim{int(use_ddim_tag)}_{args.save_tag}_eval_humanml_{name}_{niter}",
    )
    if args.guidance_param != 1.0:
        log_file += f"_gscale{args.guidance_param}"
    if args.inpainting_mask != "":
        log_file += f"_mask_{args.inpainting_mask}"
    log_file += f"_{args.eval_mode}.log"
    print(f"Will save to log file [{log_file}]")
    if os.path.exists(log_file):
        os.remove(log_file)

    dist_util.setup_dist(args.device)
    logger.configure()

    logger.log("creating data loader...")
    split = args.text_split
    gt_loader = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=None,
        split=split,
        load_mode="gt",
        drop_last=False,
    )
    gen_loader = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=None,
        split=split,
        load_mode="eval",
        drop_last=False,
    )
    print(f"dataset split={split}, len={len(gen_loader.dataset)}, n_batches={len(gen_loader)}")
    if len(gen_loader) == 0:
        raise RuntimeError(f"Empty dataloader for split={split}")

    logger.log("Creating MIC model and diffusion...")
    from diffusion.ddim_mic import InpaintingGaussianDiffusionMIC

    DiffusionClass = (
        InpaintingGaussianDiffusionMIC if args.filter_noise else SpacedDiffusion
    )
    model, diffusion = load_model_blending_and_diffusion(
        args_list, gen_loader, dist_util.dev(), DiffusionClass=DiffusionClass
    )

    data_transform = DataTransform(device="cpu")
    num_samples_limit = args.num_samples_limit

    motion_gen_all, length_gen, texts_gen, _meta = get_gen_motion_mic(
        args,
        model,
        diffusion,
        gen_loader,
        num_samples_limit,
        args.guidance_param,
        init_motion_type=None,
    )
    motion_gen, loss_head_gen, constraint_gen = motion_gen_all
    print("constraint_gen.shape = ", constraint_gen.shape)

    if args.ret_type == "pos":
        motion_gen_joints = data_transform.sample_to_joints(motion_gen)
    elif args.ret_type == "rot":
        motion_gen_joints = data_transform.sample_to_joints_from_rot(motion_gen)
    else:
        raise ValueError()
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
    print(f"[MIC] saved → {save_npy_path}")


if __name__ == "__main__":
    main()
