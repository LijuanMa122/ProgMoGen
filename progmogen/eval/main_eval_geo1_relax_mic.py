"""GEO-1 MIC metrics: Skating, Max Acc, C.Err, Unsucc.Rate (+ Pass stub)."""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from metrics2 import get_jittor_stat, get_skate_stat, read_all_sample


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", default="")
    p.add_argument("--thresh", type=float, default=0.05)
    return p.parse_args()


def read_loss(path):
    loss = np.load(path, allow_pickle=True).item()["loss"].mean()
    print(f"constraint_error = {loss:.5f}")
    return float(loss)


def unsuccess_rate(path, thresh=0.05):
    """Unsuccess if per-sample mean loss >= thresh (loss stored in gen.npy)."""
    data = np.load(path, allow_pickle=True).item()
    loss = data["loss"]
    # loss may be [N] or [N,1,...]
    loss = np.array(loss).reshape(len(data["lengths"]), -1).mean(axis=1)
    rate = float((loss >= thresh).mean())
    print(f"unsuccess_rate = {rate:.4f}")
    return rate


def main():
    args = get_parser()
    print("=" * 80)
    print("->", args.input_path)
    get_skate_stat(args.input_path)
    get_jittor_stat(args.input_path, order=2, stat_type="max")
    read_loss(args.input_path)
    unsuccess_rate(args.input_path, thresh=args.thresh)
    print("pass_rate = nan  # TODO: MuJoCo")


if __name__ == "__main__":
    main()
