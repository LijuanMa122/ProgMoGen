"""HSI-1 MIC metrics: Skating, Max Acc, C.Err, Unsucc.Rate (+ Pass stub)."""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from metrics2 import get_jittor_stat, get_skate_stat
from config_data import EVAL_HSI1_FILE_NAME


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", default="")
    p.add_argument("--threshold", type=float, default=0.05)
    return p.parse_args()


def main():
    args = get_parser()
    file_name = args.input_path
    print("=" * 80)
    print("->", file_name)
    get_skate_stat(file_name)
    get_jittor_stat(file_name, order=2, stat_type="max")

    # reuse ProgMoGen HSI-1 eval for mae + unsuccess
    from main_eval_hsi1 import get_head_height_loss

    constraint = np.load(EVAL_HSI1_FILE_NAME, allow_pickle=True)
    constraint = np.array([each[3] for each in constraint])
    # trim to generated count
    data = np.load(file_name, allow_pickle=True).item()
    n = len(data["lengths"])
    constraint = constraint[:n]
    get_head_height_loss(file_name, constraint)
    print("pass_rate = nan  # TODO: MuJoCo")


if __name__ == "__main__":
    main()
