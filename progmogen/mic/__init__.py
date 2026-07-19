"""
MIC: Motion-Inference-as-Control (Hui et al., arXiv:2607.01990)

Training-free plug-in on top of ProgMoGen / MDM DDIM sampling.
See project root MIC_REPRODUCTION_README.md for formula mapping.
"""

from .config import MICConfig, default_mic_config
from .constraints import Constraint, ConstraintType, build_constraints_from_task
from .sample_loop_mic import run_mic_sample_loop

__all__ = [
    "MICConfig",
    "default_mic_config",
    "Constraint",
    "ConstraintType",
    "build_constraints_from_task",
    "run_mic_sample_loop",
]
