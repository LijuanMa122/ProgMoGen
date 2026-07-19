"""
HSI-2 MIC evaluation: Skating, Max Acc, C.Err, Unsucc.Rate.
(Pass / MuJoCo left as stub — see MIC_REPRODUCTION_README.md)
"""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from metrics2 import get_jittor_stat, get_skate_stat, read_all_sample
from atomic_lib.math_utils import *


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="", help="path to gen.npy")
    parser.add_argument("--eps", type=float, default=0.05, help="success threshold slack")
    return parser.parse_args()


def read_loss(npy_file_name):
    data_npy = np.load(npy_file_name, allow_pickle=True).item()
    loss = data_npy["loss"].mean()
    print(f"constraint_error = {loss:.5f}")
    return loss


def check_success_one(sample, length, eps=0.05):
    """sample: [1,22,3,T] numpy or torch-compatible indexing."""
    t0, tmid, tend = 0, length // 2, length - 1
    hy0 = sample[0, head, 1, t0]
    hym = sample[0, head, 1, tmid]
    hy1 = sample[0, head, 1, tend]
    fy_l = sample[0, left_foot, 1, tmid]
    fy_r = sample[0, right_foot, 1, tmid]
    return (
        (hy0 > 1.5 - eps)
        and (hy1 > 1.5 - eps)
        and (hym < 0.5 + eps)
        and (fy_l < 0.0 + eps)
        and (fy_r < 0.0 + eps)
    )


def unsuccess_rate(npy_file_name, eps=0.05):
    sample_list, length_list = read_all_sample(npy_file_name)
    bs = len(length_list)
    ok = []
    for i in range(bs):
        ok.append(check_success_one(sample_list[i : i + 1], int(length_list[i]), eps=eps))
    rate = 1.0 - float(np.mean(ok))
    print(f"unsuccess_rate = {rate:.4f}")
    return rate


def pass_rate_stub(npy_file_name):
    """MuJoCo physical pass — not implemented in skeleton."""
    print("pass_rate = nan  # TODO: MuJoCo simulation check")
    return float("nan")


def main():
    args = get_parser()
    file_name = args.input_path
    print("=" * 80)
    print("->", file_name)
    get_skate_stat(file_name)
    get_jittor_stat(file_name, order=2, stat_type="max")
    read_loss(file_name)
    unsuccess_rate(file_name, eps=args.eps)
    pass_rate_stub(file_name)


if __name__ == "__main__":
    main()
