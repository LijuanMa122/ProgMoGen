#!/usr/bin/env bash
# Run MIC evaluation for all benchmark tasks (HSI-1/2/3, GEO-1, HOI-1)
set -eu

cd "$(dirname "$0")/.."

sh script_eval/eval_task_hsi2_mic.sh
sh script_eval/eval_task_hsi3_mic.sh
sh script_eval/eval_task_geo1_relax_mic.sh
sh script_eval/eval_task_hoi1_relax_mic.sh
# HSI-1 is slower (512 samples); run last / optionally comment out
sh script_eval/eval_task_hsi1_mic.sh
