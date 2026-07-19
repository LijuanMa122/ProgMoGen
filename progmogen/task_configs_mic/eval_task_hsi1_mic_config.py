"""
HSI-1 MIC: keyframe head-height (objective) + skating / success (criterion).
"""

from __future__ import annotations

from atomic_lib.math_utils import *
from mic.common_criteria import criterion_skating, feet_mask_fn
from mic.constraints import Constraint, ConstraintType, latent_mask_from_joint_frames

lr = 0.005
iterations = 50
decay_steps = None

MIC_WARM_START_ITERATIONS = max(1, iterations // 2)
MIC_GAMMA = 0.1
MIC_LAMBDA = 1.0
MIC_W_MAX = 10.0
SUCCESS_THRESHOLD = 0.05


def f_loss(self, sample):
    def loss_head(sample, h0, t):
        joints = self.sample_to_joints(sample)
        h_pred = keyframe(dimY(get_joint(joints, head)), t)
        return equal(h_pred, h0)

    t_all = self.length - 1
    t_0 = 0
    t_middle = self.length // 2
    return (
        loss_head(sample, h0=self.h_gt_list[1], t=t_middle) * 2
        + loss_head(sample, h0=self.h_gt_list[0], t=t_0)
        + loss_head(sample, h0=self.h_gt_list[2], t=t_all)
    )


def f_eval(self, sample):
    return f_loss(self, sample)


def _objective_head(x0_hat, diffusion=None, **kwargs):
    return f_loss(diffusion, x0_hat)


def _criterion_success(x0_hat, diffusion=None, threshold=SUCCESS_THRESHOLD, **kwargs):
    joints = diffusion.sample_to_joints(x0_hat)
    L = int(diffusion.length)
    t0, tmid, tend = 0, L // 2, L - 1
    h_gt = diffusion.h_gt_list
    preds = [
        float(keyframe(dimY(get_joint(joints, head)), t0).item()),
        float(keyframe(dimY(get_joint(joints, head)), tmid).item()),
        float(keyframe(dimY(get_joint(joints, head)), tend).item()),
    ]
    err = [abs(preds[i] - float(h_gt[i])) for i in range(3)]
    return 0.0 if all(e < threshold for e in err) else 1.0


def build_mic_constraints(shape, length: int, device):
    t0, tmid, tend = 0, length // 2, length - 1

    def mask_head(shape_, length_, device_):
        return latent_mask_from_joint_frames(
            shape_, length_, device_, joint_ids=[head], frames=[t0, tmid, tend]
        )

    return [
        Constraint(
            name="head_height",
            type=ConstraintType.OBJECTIVE,
            energy_fn=_objective_head,
            mask_fn=mask_head,
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
            mask_fn=mask_head,
        ),
    ]
