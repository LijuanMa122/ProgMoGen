"""MIC hyperparameters (paper Implementation Details + tunable knobs)."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


@dataclass
class MICConfig:
    # ---- paper-specified ----
    M: int = 16                      # Eq.9 importance samples
    elite_ratio: float = 0.2         # CEM elite ratio
    eta: float = 0.0                 # DDIM deterministic
    guidance_scale: float = 1.0      # global scale on applied u_t

    # ---- warm-start (ProgMoGen / DNO) ----
    warm_start: bool = True
    warm_start_iterations: Optional[int] = None  # None → use task_config.iterations // 2
    warm_start_lr: Optional[float] = None        # None → use task_config.lr
    # "noise_opt": short Adam on noise_init (HSI)
    # "goal_relaxed": multi-epoch relax + noise opt (GEO/HOI); post-MIC transform_sample
    warm_start_mode: str = "noise_opt"
    post_transform: bool = False  # GEO/HOI: map relaxed goal → hard target via RT

    # ---- feedback regulator (Eq.11) ----
    gamma: float = 0.1
    W_max: float = 10.0
    W_init: float = 0.0
    ema_alpha: float = 0.95

    # ---- control allocator (Eq.12–13) ----
    lambda_: float = 1.0

    # ---- CEM extras ----
    cem_eps: float = 1e-4
    cem_momentum: float = 0.5

    # ---- ablations ----
    # none | no_regulation | no_allocation | no_coordination | objective_only | criterion_only
    ablation: str = "none"

    # ---- misc ----
    progress: bool = True
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_mic_config(**overrides) -> MICConfig:
    cfg = MICConfig()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown MICConfig field: {k}")
        setattr(cfg, k, v)
    return cfg
