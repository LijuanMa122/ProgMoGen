"""
GEO-1 MIC: left wrist on plane (objective) + skating / success (criterion).

Warm-start = full ProgMoGen goal_relaxed (multi-epoch soft plane + noise opt).
MIC optimizes against the relaxed plane; post_transform maps to the hard target.
"""

from __future__ import annotations

import numpy as np
import torch as th

from atomic_lib.math_utils import *
from atomic_lib.relax_geometry import construct_plane, calc_RT_from_two_planes, apply_RT_on_joints
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from mic.common_criteria import criterion_skating, feet_mask_fn
from mic.constraints import Constraint, ConstraintType, latent_mask_from_joint_frames

DEMO_NUM = 32
lr = 0.05
epoch_relax = 5
iterations = 20
joint_control = left_wrist

MIC_WARM_START_MODE = "goal_relaxed"
MIC_POST_TRANSFORM = True
MIC_WARM_START_ITERATIONS = None  # unused in goal_relaxed mode (uses epoch_relax × iterations)
MIC_GAMMA = 0.1
MIC_LAMBDA = 1.0
MIC_W_MAX = 10.0
SUCCESS_THRESH = 0.05  # metres to plane


def get_lr_schedule(self, i_k):
    if i_k in [0]:
        return 1e-2
    if i_k in [1, 2]:
        return 5e-3
    return 1e-3


def f_sample_random_plane(self, r_range=3, seed=0):
    rng = np.random.default_rng(seed)
    r = r_range * rng.random()
    theta = rng.random() * np.pi * 2
    x0 = r * np.cos(theta) + 1e-5
    z0 = r * np.sin(theta)
    A, B = 1, 0
    C = z0 / x0
    D = (-x0 * x0 - z0 * z0) / x0
    plane = th.FloatTensor([[A, B, C, D]])
    print("plane=", plane)
    return plane


def f_loss(self, sample, sample_0, it, plane_params):
    joints = self.sample_to_joints(sample)
    assert joints.shape[0] == 1
    joint_traj = get_joint(joints, joint_control).squeeze().permute(1, 0).contiguous()
    return loss_dist_to_plane(joint_traj, plane_params).mean()


def f_eval(self, sample, sample_0, plane_params, XZ_offset=False):
    if plane_params.shape[1] == 6:
        plane_target = construct_plane(plane_params[0, :3], plane_params[0, 3:])
        plane_params = plane_point_normal_form_to_params_4d(plane_target)
    if XZ_offset is False:
        joints = self.sample_to_joints(sample)
    else:
        joints = self.sample_to_joints_with_XZ_offset(sample)
    joints = get_joint(joints, joint_control)
    return dist_to_plane(joints, plane_params)


def update_goal(self, sample, target, target_relaxed, i_k):
    joint_traj = self.get_global_traj_for_joints(sample, joint_control)
    return fit_yPlane(joint_traj)


def transform_sample(self, sample_ret, target_relaxed, target):
    if True:
        target_relaxed = self.update_goal(sample_ret, None, None, None)
    plane_params = target_relaxed
    plane_pn = plane_params_4d_to_point_normal_form(plane_params)
    plane_relax = construct_plane(plane_pn[0, :3], plane_pn[0, 3:])
    target_list = plane_params_4d_to_point_normal_form(target)
    plane_target = construct_plane(target_list[0, :3], target_list[0, 3:])
    R, translation = calc_RT_from_two_planes(plane_relax, plane_target)
    joints_src = self.sample_to_joints(sample_ret)
    joints_dst = apply_RT_on_joints(joints_src, R, translation)
    joints_dst = th.cat([joints_dst, joints_dst[:, :, :, -1:]], 3)
    return self.joints_to_sample_with_XZ_offset(joints_dst)


def _mic_target(diffusion):
    """Prefer relaxed target during MIC; fall back to hard target_gt."""
    t = getattr(diffusion, "target_mic", None)
    return t if t is not None else diffusion.target_gt


def _objective_plane(x0_hat, diffusion=None, **kwargs):
    return f_loss(diffusion, x0_hat, None, 0, _mic_target(diffusion))


def _criterion_success(x0_hat, diffusion=None, thresh=SUCCESS_THRESH, **kwargs):
    dist = f_eval(diffusion, x0_hat, None, _mic_target(diffusion), XZ_offset=False)
    # dist: per-frame; success if mean distance small
    mean_d = float(dist.mean().item()) if th.is_tensor(dist) else float(dist)
    return 0.0 if mean_d < thresh else 1.0


def build_mic_constraints(shape, length: int, device):
    def mask_wrist(shape_, length_, device_):
        return latent_mask_from_joint_frames(
            shape_, length_, device_, joint_ids=[joint_control], frames=None
        )

    return [
        Constraint(
            name="plane_touch",
            type=ConstraintType.OBJECTIVE,
            energy_fn=_objective_plane,
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
