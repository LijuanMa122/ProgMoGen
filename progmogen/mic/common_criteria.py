"""Shared criterion energies used across MIC task configs."""

from __future__ import annotations


def criterion_skating(x0_hat, diffusion=None, **kwargs):
    from eval.metrics import calculate_skating_ratio

    joints = diffusion.sample_to_joints(x0_hat)
    L = int(diffusion.length)
    joints = joints[:, :, :, :L]
    ratio, _ = calculate_skating_ratio(joints)
    return float(ratio.mean())


def feet_mask_fn(shape, length, device):
    from atomic_lib.math_utils import left_foot, right_foot
    from .constraints import latent_mask_from_joint_frames

    return latent_mask_from_joint_frames(
        shape, length, device, joint_ids=[left_foot, right_foot], frames=None
    )
