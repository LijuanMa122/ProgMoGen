#!/usr/bin/env bash
# MIC evaluation on HSI-3 (limited space)
set -eu

eval_method="mic"
ret_type="pos"
text_split="test_plane_v0_id"
num_samples_limit=32
save_tag="limited_space_mic"

save_fig_dir="result/eval/${save_tag}_n${num_samples_limit}/${eval_method}_${ret_type}_npy"
task_config="eval_task_hsi3_mic_config"

python3 tasks/eval_task_mic.py \
    --use_ddim_tag 1 \
    --mask_type 'root_horizontal' \
    --eval_mode "debug" \
    --save_tag "${save_tag}" \
    --ret_type "${ret_type}" \
    --save_fig_dir "${save_fig_dir}" \
    --text_split "${text_split}" \
    --num_samples_limit ${num_samples_limit} \
    --task_config ${task_config}

python3 eval/main_eval_hsi3_mic.py --input_path "${save_fig_dir}/gen.npy"
