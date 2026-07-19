"""
HOI-1 MIC: move object A→B via left wrist (objective) + skating / success (criterion).
"""

from __future__ import annotations

import numpy as np
import torch as th

from atomic_lib.math_utils import *
from atomic_lib.relax_geometry import apply_RT_on_joints
from mic.common_criteria import criterion_skating, feet_mask_fn
from mic.constraints import Constraint, ConstraintType, latent_mask_from_joint_frames

DEMO_NUM = 32
lr = 0.02
epoch_relax = 3
iterations = 30

MIC_WARM_START_MODE = "goal_relaxed"
MIC_POST_TRANSFORM = True
MIC_WARM_START_ITERATIONS = None  # unused in goal_relaxed mode
MIC_GAMMA = 0.1
MIC_LAMBDA = 1.0
MIC_W_MAX = 10.0
SUCCESS_THRESH = 0.05


def get_lr_schedule(self, i_k):
    if i_k in [0, 1]:
        return 0.02
    return 0.002


def get_target_pred_from_pos(joints, length, joint_control):
    t_all = length - 1
    t_0 = 0
    p0 = keyframe(get_joint(joints, joint_control), t_0).squeeze()
    p1 = keyframe(get_joint(joints, joint_control), t_all).squeeze()
    return [p0, p1]


def f_loss(self, sample, sample_0, it, target):
    joints = self.sample_to_joints(sample)
    assert joints.shape[2] == 3
    p0_pred, p1_pred = get_target_pred_from_pos(joints, self.length, left_wrist)
    p0_gt, p1_gt = target[0, :3], target[0, 3:]
    return (equal_sum(p0_pred, p0_gt) + equal_sum(p1_pred, p1_gt)) / 2


def f_eval(self, sample, sample_0, target, XZ_offset=False):
    if XZ_offset is False:
        joints = self.sample_to_joints(sample)
    else:
        joints = self.sample_to_joints_with_XZ_offset(sample)
    p0_pred, p1_pred = get_target_pred_from_pos(joints, self.length, left_wrist)
    p0_gt, p1_gt = target[0, :3], target[0, 3:]
    loss = (equal_sum(p0_pred, p0_gt) + equal_sum(p1_pred, p1_gt)) / 2
    return loss.sqrt()


def get_xz_constraint(p0_gt_0, p1_gt_0, p0_pred_0, p1_pred_0):
    p0_gt, p1_gt = p0_gt_0.clone(), p1_gt_0.clone()
    p0_pred, p1_pred = p0_pred_0.clone(), p1_pred_0.clone()
    p0_gt[1] = p1_gt[1] = p0_pred[1] = p1_pred[1] = 0
    center_pred = (p0_pred + p1_pred) / 2
    dir_p0 = (p0_pred - center_pred) / th.linalg.norm(p0_pred - center_pred)
    dir_p1 = (p1_pred - center_pred) / th.linalg.norm(p1_pred - center_pred)
    center_gt = (p0_gt + p1_gt) / 2
    len_p0_gt = th.linalg.norm(p0_gt - center_gt)
    len_p1_gt = th.linalg.norm(p1_gt - center_gt)
    p0_gt_relax = center_gt + len_p0_gt * dir_p0
    p1_gt_relax = center_gt + len_p1_gt * dir_p1
    p0_gt_relax[1] = p0_gt_0[1]
    p1_gt_relax[1] = p1_gt_0[1]
    return [p0_gt_relax, p1_gt_relax]


def update_goal(self, sample, target, target_relaxed, i_k):
    p0_pred, p1_pred = get_target_pred_from_pos(
        self.sample_to_joints(sample), self.length, left_wrist
    )
    if i_k == 0:
        return target.clone()
    p0_gt, p1_gt = target_relaxed[0, :3], target_relaxed[0, 3:]
    relax = get_xz_constraint(p0_gt, p1_gt, p0_pred, p1_pred)
    return th.cat([relax[0].reshape(1, -1), relax[1].reshape(1, -1)], 1)


def calc_RT_between_goals(target_gt_list_relax, target_gt_list):
    device = target_gt_list_relax[0].device
    x1, z1 = float(target_gt_list_relax[0][0]), float(target_gt_list_relax[0][2])
    x2, z2 = float(target_gt_list_relax[1][0]), float(target_gt_list_relax[1][2])
    x1_gt, z1_gt = float(target_gt_list[0][0]), float(target_gt_list[0][2])
    x2_gt, z2_gt = float(target_gt_list[1][0]), float(target_gt_list[1][2])
    A_mat = np.array(
        [[x1, -z1, 1.0, 0.0], [z1, x1, 0.0, 1.0], [x2, -z2, 1.0, 0.0], [z2, x2, 0.0, 1.0]],
        dtype=np.float32,
    )
    b_mat = np.array([x1_gt, z1_gt, x2_gt, z2_gt], dtype=np.float32).reshape(4, 1)
    a, b, c, d = [float(x) for x in np.linalg.solve(A_mat, b_mat).reshape(-1)]
    R = th.zeros((3, 3), device=device)
    translation = th.zeros((3, 1), device=device)
    R[0, 0], R[0, 2], R[1, 1], R[2, 0], R[2, 2] = a, -b, 1.0, b, a
    translation[0, 0], translation[2, 0] = c, d
    return R, translation


def transform_sample(self, sample_ret, target_relaxed, target):
    joints = self.sample_to_joints(sample_ret)
    target_gt_list_relax = [target_relaxed[0][:3], target_relaxed[0][3:]]
    target_gt_list = [target[0][:3], target[0][3:]]
    R, translation = calc_RT_between_goals(target_gt_list_relax, target_gt_list)
    joints_dst = apply_RT_on_joints(joints, R, translation)
    joints_dst = th.cat([joints_dst, joints_dst[:, :, :, -1:]], 3)
    return self.joints_to_sample_with_XZ_offset(joints_dst)


def _mic_target(diffusion):
    t = getattr(diffusion, "target_mic", None)
    return t if t is not None else diffusion.target_gt


def _objective_pick(x0_hat, diffusion=None, **kwargs):
    return f_loss(diffusion, x0_hat, None, 0, _mic_target(diffusion))


def _criterion_success(x0_hat, diffusion=None, thresh=SUCCESS_THRESH, **kwargs):
    err = f_eval(diffusion, x0_hat, None, _mic_target(diffusion), XZ_offset=False)
    val = float(err.mean().item()) if th.is_tensor(err) else float(err)
    return 0.0 if val < thresh else 1.0


def build_mic_constraints(shape, length: int, device):
    def mask_wrist(shape_, length_, device_):
        t0, tend = 0, max(0, length_ - 1)
        return latent_mask_from_joint_frames(
            shape_, length_, device_, joint_ids=[left_wrist], frames=[t0, tend]
        )

    return [
        Constraint(
            name="pick_ab",
            type=ConstraintType.OBJECTIVE,
            energy_fn=_objective_pick,
            mask_fn=mask_wrist,
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
            mask_fn=mask_wrist,
        ),
    ]
