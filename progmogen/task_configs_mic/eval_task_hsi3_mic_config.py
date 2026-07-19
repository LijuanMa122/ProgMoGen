"""
HSI-3 MIC: limited-area walking (objective) + skating / success (criterion).
"""

from __future__ import annotations

from atomic_lib.math_utils import *
from mic.common_criteria import criterion_skating, feet_mask_fn
from mic.constraints import Constraint, ConstraintType, ones_mask

lr = 0.005
iterations = 50
decay_steps = None

MIC_WARM_START_ITERATIONS = max(1, iterations // 2)
MIC_GAMMA = 0.1
MIC_LAMBDA = 1.0
MIC_W_MAX = 10.0
SUCCESS_EPS = 0.05


def loss_limited_space(joints):
    loss_1 = less_than(dimX(joints), 1.0).mean()
    loss_2 = greater_than(dimX(joints), -1.0).mean()
    loss_3 = less_than(dimZ(joints), 1.0).mean()
    loss_4 = greater_than(dimZ(joints), -1.0).mean()
    return (loss_1 + loss_2 + loss_3 + loss_4) / 4


def f_loss(self, sample, sample_0, step):
    return loss_limited_space(self.sample_to_joints(sample))


def f_eval(self, sample, sample_0):
    return loss_limited_space(self.sample_to_joints(sample))


def _objective_space(x0_hat, diffusion=None, **kwargs):
    return loss_limited_space(diffusion.sample_to_joints(x0_hat))


def _criterion_success(x0_hat, diffusion=None, eps=SUCCESS_EPS, **kwargs):
    """Success if all joints stay in [-1-eps, 1+eps] on X and Z over valid length."""
    joints = diffusion.sample_to_joints(x0_hat)
    L = int(diffusion.length)
    xyz = joints[0, :, :, :L]  # [22,3,L]
    x = xyz[:, 0, :]
    z = xyz[:, 2, :]
    ok = (
        (x.min() >= -1.0 - eps)
        and (x.max() <= 1.0 + eps)
        and (z.min() >= -1.0 - eps)
        and (z.max() <= 1.0 + eps)
    )
    return 0.0 if bool(ok) else 1.0


def build_mic_constraints(shape, length: int, device):
    return [
        Constraint(
            name="limited_space",
            type=ConstraintType.OBJECTIVE,
            energy_fn=_objective_space,
            mask_fn=ones_mask,
        ),
        Constraint(
            name="foot_skating",
            type=ConstraintType.CRITERION,
            energy_fn=criterion_skating,
            mask_fn=feet_mask_fn,
        ),
        Constraint(
            name="success_check",
            type=ConstraintType.CRITERION,
            energy_fn=_criterion_success,
            mask_fn=ones_mask,
        ),
    ]
