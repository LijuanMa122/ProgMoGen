"""
HSI-2 (overhead barrier) MIC task config.

Constraints:
  k=0 objective : overhead barrier energy (same as ProgMoGen)
  k=1 criterion : foot skating ratio
  k=2 criterion : discrete success check
"""

from __future__ import annotations

from atomic_lib.math_utils import *
from mic.constraints import Constraint, ConstraintType, latent_mask_from_joint_frames

# ---- ProgMoGen-compatible warm-start / eval hooks ----
lr = 0.005
iterations = 100
decay_steps = None

MIC_WARM_START_ITERATIONS = max(1, iterations // 2)
MIC_GAMMA = 0.1
MIC_LAMBDA = 1.0
MIC_W_MAX = 10.0


def loss_overhead_barrier(joints, length):
    t_all = length - 1
    t_0 = 0
    t_middle = length // 2

    loss_head = (
        greater_than(keyframe(dimY(get_joint(joints, head)), t_0), 1.5)
        + greater_than(keyframe(dimY(get_joint(joints, head)), t_all), 1.5)
        + less_than(keyframe(dimY(get_joint(joints, head)), t_middle), 0.5)
    ) / 3

    loss_foot = (
        less_than(keyframe(dimY(get_joint(joints, left_foot)), t_middle), 0.0)
        + less_than(keyframe(dimY(get_joint(joints, right_foot)), t_middle), 0.0)
    ) / 2

    return loss_foot + loss_head


def f_loss(self, sample, sample_0, step):
    joints = self.sample_to_joints(sample)
    assert joints.shape[0] == 1
    return loss_overhead_barrier(joints, self.length)


def f_eval(self, sample, sample_0):
    joints = self.sample_to_joints(sample)
    assert joints.shape[0] == 1
    return loss_overhead_barrier(joints, self.length)


def _objective_barrier(x0_hat, diffusion=None, **kwargs):
    joints = diffusion.sample_to_joints(x0_hat)
    return loss_overhead_barrier(joints, diffusion.length)


def _criterion_skating(x0_hat, diffusion=None, **kwargs):
    from eval.metrics import calculate_skating_ratio

    joints = diffusion.sample_to_joints(x0_hat)
    L = int(diffusion.length)
    joints = joints[:, :, :, :L]
    ratio, _ = calculate_skating_ratio(joints)
    return float(ratio.mean())


def _criterion_success(x0_hat, diffusion=None, eps: float = 0.05, **kwargs):
    joints = diffusion.sample_to_joints(x0_hat)
    L = int(diffusion.length)
    t0, tmid, tend = 0, L // 2, L - 1
    hy0 = float(keyframe(dimY(get_joint(joints, head)), t0).item())
    hym = float(keyframe(dimY(get_joint(joints, head)), tmid).item())
    hy1 = float(keyframe(dimY(get_joint(joints, head)), tend).item())
    fy_l = float(keyframe(dimY(get_joint(joints, left_foot)), tmid).item())
    fy_r = float(keyframe(dimY(get_joint(joints, right_foot)), tmid).item())
    ok = (
        (hy0 > 1.5 - eps)
        and (hy1 > 1.5 - eps)
        and (hym < 0.5 + eps)
        and (fy_l < 0.0 + eps)
        and (fy_r < 0.0 + eps)
    )
    return 0.0 if ok else 1.0


def build_mic_constraints(shape, length: int, device):
    t0, tmid, tend = 0, length // 2, length - 1

    def mask_barrier(shape_, length_, device_):
        return latent_mask_from_joint_frames(
            shape_,
            length_,
            device_,
            joint_ids=[head, left_foot, right_foot],
            frames=[t0, tmid, tend],
        )

    def mask_feet(shape_, length_, device_):
        return latent_mask_from_joint_frames(
            shape_,
            length_,
            device_,
            joint_ids=[left_foot, right_foot],
            frames=None,
        )

    return [
        Constraint(
            name="overhead_barrier",
            type=ConstraintType.OBJECTIVE,
            energy_fn=_objective_barrier,
            mask_fn=mask_barrier,
        ),
        Constraint(
            name="foot_skating",
            type=ConstraintType.CRITERION,
            energy_fn=_criterion_skating,
            mask_fn=mask_feet,
        ),
        Constraint(
            name="success_check",
            type=ConstraintType.CRITERION,
            energy_fn=_criterion_success,
            mask_fn=mask_barrier,
        ),
    ]
