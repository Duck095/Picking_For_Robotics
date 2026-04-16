from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class SimConfig:
    use_gui: bool = False
    physics_hz: int = 240
    substeps: int = 12
    max_steps: int = 220
    seed: int = 42


@dataclass
class WorkspaceConfig:
    x_range: Tuple[float, float] = (0.30, 0.75)
    y_range: Tuple[float, float] = (-0.10, 0.25)
    z_range: Tuple[float, float] = (0.05, 0.55)


@dataclass
class ActionConfig:
    clip_action: bool = True
    action_scale_xy: float = 0.024
    action_scale_z: float = 0.028
    action_scale_yaw: float = 0.0
    action_smoothing: float = 0.15

    near_dist: float = 0.14
    settle_dist: float = 0.09
    descend_z_gate: float = 0.055


@dataclass
class TargetConfig:
    use_hover_target: bool = True
    use_pregrasp_target: bool = False
    use_orientation_target: bool = False

    z_hover: float = 0.15
    pregrasp_z: float = 0.08

    success_dist: float = 0.08
    xy_align_threshold: float = 0.05

    xy_success_dist: float = 0.012
    z_success_dist: float = 0.015
    yaw_align_threshold: float = 0.18
    yaw_success_dist: float = 0.10
    yaw_offset: float = 0.0

    stable_steps_required: int = 3


@dataclass
class ResetConfig:
    start_near_target: bool = False

    near_target_xy_noise: float = 0.010
    near_target_z_noise: float = 0.010

    near_target_yaw_min_error: float = 0.50
    near_target_yaw_max_error: float = 1.20


@dataclass
class CurriculumLevelConfig:
    xy_noise: float = 0.010
    z_noise: float = 0.008
    yaw_min_error: float = 0.55
    yaw_max_error: float = 1.10
    action_scale_yaw_mult: float = 1.0
    yaw_progress_mult: float = 1.0
    timeout_penalty_mult: float = 1.0


@dataclass
class CurriculumConfig:
    enabled: bool = False
    force_level: str = ""
    easy_until_episode: int = 900
    medium_until_episode: int = 1800

    easy: CurriculumLevelConfig = field(
        default_factory=lambda: CurriculumLevelConfig(
            xy_noise=0.006,
            z_noise=0.006,
            yaw_min_error=0.30,
            yaw_max_error=0.80,
            action_scale_yaw_mult=0.92,
            yaw_progress_mult=0.90,
            timeout_penalty_mult=0.85,
        )
    )
    medium: CurriculumLevelConfig = field(
        default_factory=lambda: CurriculumLevelConfig(
            xy_noise=0.008,
            z_noise=0.008,
            yaw_min_error=0.45,
            yaw_max_error=1.00,
            action_scale_yaw_mult=1.00,
            yaw_progress_mult=1.00,
            timeout_penalty_mult=1.00,
        )
    )
    hard: CurriculumLevelConfig = field(
        default_factory=lambda: CurriculumLevelConfig(
            xy_noise=0.010,
            z_noise=0.010,
            yaw_min_error=0.55,
            yaw_max_error=1.20,
            action_scale_yaw_mult=1.08,
            yaw_progress_mult=1.06,
            timeout_penalty_mult=1.10,
        )
    )


@dataclass
class SpawnConfig:
    spawn_mode: str = "fixed"
    fixed_pose_xyz: Tuple[float, float, float] = (0.55, 0.05, 0.02)

    roi_x: Tuple[float, float] = (0.30, 0.75)
    roi_y: Tuple[float, float] = (-0.10, 0.25)
    object_z: float = 0.02

    random_yaw: bool = False
    yaw_range: Tuple[float, float] = (-3.14159, 3.14159)

    object_colors: Tuple[str, ...] = ("red", "green", "blue", "yellow")


@dataclass
class RewardConfig:
    w_dist_progress: float = 1.0
    w_xy_progress: float = 1.6
    w_z_progress: float = 1.2
    w_success: float = 3.0

    w_idle_penalty: float = 0.0015
    idle_eps: float = 8e-5
    w_wrong_descend_penalty: float = 0.008
    w_action_penalty: float = 0.0004
    w_workspace_violation: float = 0.05
    w_timeout_penalty: float = 0.02

    w_descend_bonus: float = 0.03
    w_xy_regress_descend_penalty: float = 0.01
    w_hang_high_penalty: float = 0.015
    w_no_descend_penalty: float = 0.008

    w_jitter_near: float = 0.004
    w_reverse_action_near: float = 0.008
    hang_steps_tolerance: int = 6

    w_stable_bonus: float = 0.04
    w_regress_near: float = 0.015

    w_yaw_progress: float = 0.0
    w_yaw_regress_near: float = 0.0
    w_yaw_action_penalty: float = 0.0008

    w_stagnation_penalty: float = 0.0
    stagnation_motion_eps: float = 0.0
    stagnation_progress_eps: float = 0.0
    w_approach_stagnation_boost: float = 0.0

    w_phase_lock_xy_penalty: float = 0.0
    w_phase_lock_z_penalty: float = 0.0


@dataclass
class NoiseConfig:
    enable_obs_noise: bool = False
    pos_noise_std: float = 0.002


@dataclass
class DebugConfig:
    draw_target: bool = True
    draw_ee_path: bool = True


@dataclass
class ReachEnvConfig:
    substage: str = "1A"

    sim: SimConfig = field(default_factory=SimConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    reset_cfg: ResetConfig = field(default_factory=ResetConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    include_joint_state: bool = True
    include_joint_velocity: bool = True
    include_prev_action: bool = True


def _apply_nested_overrides(obj, overrides: Dict):
    for key, value in overrides.items():
        current = getattr(obj, key)
        if isinstance(value, dict):
            _apply_nested_overrides(current, value)
        else:
            setattr(obj, key, value)


def get_substage_overrides(substage: str) -> Dict:
    if substage == "1A":
        return {
            "sim": {"max_steps": 220},
            "action": {
                "action_scale_xy": 0.024,
                "action_scale_z": 0.030,
                "action_scale_yaw": 0.00,
                "action_smoothing": 0.12,
                "near_dist": 0.15,
                "settle_dist": 0.09,
                "descend_z_gate": 0.055,
            },
            "target": {
                "use_hover_target": True,
                "use_pregrasp_target": False,
                "use_orientation_target": False,
                "z_hover": 0.15,
                "success_dist": 0.08,
                "xy_align_threshold": 0.05,
            },
            "spawn": {
                "spawn_mode": "fixed",
                "fixed_pose_xyz": (0.55, 0.05, 0.02),
                "random_yaw": False,
            },
            "reward": {
                "w_dist_progress": 1.0,
                "w_xy_progress": 1.7,
                "w_z_progress": 1.4,
                "w_success": 3.4,
                "w_idle_penalty": 0.0010,
                "w_wrong_descend_penalty": 0.004,
                "w_action_penalty": 0.0002,
                "w_timeout_penalty": 0.01,
                "w_descend_bonus": 0.05,
                "w_xy_regress_descend_penalty": 0.006,
                "w_hang_high_penalty": 0.012,
                "w_no_descend_penalty": 0.006,
                "w_jitter_near": 0.002,
                "w_reverse_action_near": 0.004,
                "hang_steps_tolerance": 8,
                "w_yaw_progress": 0.0,
                "w_yaw_regress_near": 0.0,
                "w_yaw_action_penalty": 0.0010,
            },
            "noise": {"enable_obs_noise": False},
        }

    if substage == "1B":
        return {
            "sim": {"max_steps": 210},
            "action": {
                "action_scale_xy": 0.022,
                "action_scale_z": 0.026,
                "action_scale_yaw": 0.01,
                "action_smoothing": 0.20,
                "near_dist": 0.13,
                "settle_dist": 0.08,
                "descend_z_gate": 0.050,
            },
            "target": {
                "use_hover_target": True,
                "use_pregrasp_target": False,
                "use_orientation_target": False,
                "z_hover": 0.10,
                "success_dist": 0.065,
                "xy_align_threshold": 0.035,
            },
            "spawn": {
                "spawn_mode": "random",
                "roi_x": (0.45, 0.65),
                "roi_y": (-0.05, 0.12),
                "object_z": 0.02,
                "random_yaw": True,
            },
            "reward": {
                "w_dist_progress": 1.0,
                "w_xy_progress": 1.45,
                "w_z_progress": 1.15,
                "w_success": 3.0,
                "w_idle_penalty": 0.0018,
                "w_wrong_descend_penalty": 0.012,
                "w_action_penalty": 0.0005,
                "w_timeout_penalty": 0.02,
                "w_descend_bonus": 0.035,
                "w_xy_regress_descend_penalty": 0.012,
                "w_hang_high_penalty": 0.018,
                "w_no_descend_penalty": 0.010,
                "w_jitter_near": 0.004,
                "w_reverse_action_near": 0.008,
                "hang_steps_tolerance": 6,
                "w_yaw_progress": 0.0,
                "w_yaw_regress_near": 0.0,
                "w_yaw_action_penalty": 0.0010,
            },
            "noise": {"enable_obs_noise": False},
        }

    if substage == "1C":
        return {
            "sim": {"max_steps": 190},
            "action": {
                "action_scale_xy": 0.016,
                "action_scale_z": 0.018,
                "action_scale_yaw": 0.02,
                "action_smoothing": 0.30,
                "near_dist": 0.11,
                "settle_dist": 0.070,
                "descend_z_gate": 0.045,
            },
            "target": {
                "use_hover_target": False,
                "use_pregrasp_target": True,
                "use_orientation_target": False,
                "pregrasp_z": 0.08,
                "xy_align_threshold": 0.020,
                "xy_success_dist": 0.015,
                "z_success_dist": 0.020,
                "stable_steps_required": 2,
            },
            "spawn": {
                "spawn_mode": "random",
                "roi_x": (0.35, 0.70),
                "roi_y": (-0.08, 0.18),
                "object_z": 0.02,
                "random_yaw": True,
            },
            "reward": {
                "w_xy_progress": 1.35,
                "w_z_progress": 1.05,
                "w_success": 3.0,
                "w_stable_bonus": 0.035,
                "w_jitter_near": 0.008,
                "w_regress_near": 0.010,
                "w_action_penalty": 0.0010,
                "w_timeout_penalty": 0.03,
                "w_yaw_progress": 0.0,
                "w_yaw_regress_near": 0.0,
                "w_yaw_action_penalty": 0.0010,
            },
            "noise": {"enable_obs_noise": False},
        }

    if substage == "1D":
        return {
            "sim": {"max_steps": 200},
            "action": {
                "action_scale_xy": 0.013,
                "action_scale_z": 0.015,
                "action_scale_yaw": 0.03,
                "action_smoothing": 0.36,
                "near_dist": 0.10,
                "settle_dist": 0.065,
                "descend_z_gate": 0.045,
            },
            "target": {
                "use_hover_target": False,
                "use_pregrasp_target": True,
                "use_orientation_target": False,
                "pregrasp_z": 0.08,
                "xy_align_threshold": 0.016,
                "xy_success_dist": 0.010,
                "z_success_dist": 0.012,
                "stable_steps_required": 4,
            },
            "spawn": {
                "spawn_mode": "random",
                "roi_x": (0.30, 0.75),
                "roi_y": (-0.10, 0.25),
                "object_z": 0.02,
                "random_yaw": True,
            },
            "reward": {
                "w_xy_progress": 1.45,
                "w_z_progress": 1.15,
                "w_success": 3.4,
                "w_stable_bonus": 0.05,
                "w_jitter_near": 0.010,
                "w_regress_near": 0.015,
                "w_action_penalty": 0.0010,
                "w_timeout_penalty": 0.03,
                "w_yaw_progress": 0.0,
                "w_yaw_regress_near": 0.0,
                "w_yaw_action_penalty": 0.0010,
            },
            "noise": {
                "enable_obs_noise": True,
                "pos_noise_std": 0.0015,
            },
        }

    if substage == "1E":
        return {
            "sim": {"max_steps": 180},
            "action": {
                "action_scale_xy": 0.015,
                "action_scale_z": 0.010,
                "action_scale_yaw": 0.085,
                "action_smoothing": 0.40,
                "near_dist": 0.085,
                "settle_dist": 0.050,
                "descend_z_gate": 0.040,
            },
            "target": {
                "use_hover_target": False,
                "use_pregrasp_target": True,
                "use_orientation_target": True,
                "pregrasp_z": 0.08,
                "xy_align_threshold": 0.020,
                "xy_success_dist": 0.012,
                "z_success_dist": 0.015,
                "yaw_align_threshold": 0.24,
                "yaw_success_dist": 0.16,
                "yaw_offset": 0.0,
                "stable_steps_required": 3,
            },
            "reset_cfg": {
                "start_near_target": False,
                "near_target_xy_noise": 0.008,
                "near_target_z_noise": 0.008,
                "near_target_yaw_min_error": 0.45,
                "near_target_yaw_max_error": 1.00,
            },
            "curriculum": {
                "enabled": True,
                "force_level": "",
                "easy_until_episode": 300,
                "medium_until_episode": 900,
                "easy": {
                    "xy_noise": 0.006,
                    "z_noise": 0.006,
                    "yaw_min_error": 0.30,
                    "yaw_max_error": 0.80,
                    "action_scale_yaw_mult": 0.92,
                    "yaw_progress_mult": 0.90,
                    "timeout_penalty_mult": 0.85,
                },
                "medium": {
                    "xy_noise": 0.008,
                    "z_noise": 0.008,
                    "yaw_min_error": 0.45,
                    "yaw_max_error": 1.00,
                    "action_scale_yaw_mult": 1.00,
                    "yaw_progress_mult": 1.00,
                    "timeout_penalty_mult": 1.00,
                },
                "hard": {
                    "xy_noise": 0.010,
                    "z_noise": 0.010,
                    "yaw_min_error": 0.55,
                    "yaw_max_error": 1.20,
                    "action_scale_yaw_mult": 1.08,
                    "yaw_progress_mult": 1.06,
                    "timeout_penalty_mult": 1.10,
                },
            },
            "spawn": {
                "spawn_mode": "random",
                "roi_x": (0.30, 0.75),
                "roi_y": (-0.10, 0.25),
                "object_z": 0.02,
                "random_yaw": True,
            },
            "reward": {
                "w_dist_progress": 0.60,
                "w_xy_progress": 0.95,
                "w_z_progress": 0.90,
                "w_yaw_progress": 1.25,
                "w_success": 4.5,
                "w_stable_bonus": 0.12,
                "w_idle_penalty": 0.0030,
                "idle_eps": 1.2e-4,
                "w_action_penalty": 0.0008,
                "w_yaw_action_penalty": 0.00045,
                "w_timeout_penalty": 0.03,
                "w_jitter_near": 0.010,
                "w_reverse_action_near": 0.012,
                "w_regress_near": 0.028,
                "w_yaw_regress_near": 0.014,
                "w_stagnation_penalty": 0.0035,
                "stagnation_motion_eps": 0.0009,
                "stagnation_progress_eps": 0.00035,
                "w_approach_stagnation_boost": 0.0020,
                "w_phase_lock_xy_penalty": 0.050,
                "w_phase_lock_z_penalty": 0.045,
            },
            "noise": {
                "enable_obs_noise": True,
                "pos_noise_std": 0.0010,
            },
        }

    if substage == "1F":
        return {
            "sim": {"max_steps": 220},
            "action": {
                "action_scale_xy": 0.010,
                "action_scale_z": 0.012,
                "action_scale_yaw": 0.12,
                "action_smoothing": 0.40,
                "near_dist": 0.10,
                "settle_dist": 0.065,
                "descend_z_gate": 0.045,
            },
            "target": {
                "use_hover_target": False,
                "use_pregrasp_target": True,
                "use_orientation_target": True,
                "pregrasp_z": 0.08,
                "xy_align_threshold": 0.015,
                "xy_success_dist": 0.008,
                "z_success_dist": 0.010,
                "yaw_align_threshold": 0.18,
                "yaw_success_dist": 0.12,
                "yaw_offset": 0.0,
                "stable_steps_required": 5,
            },
            "reset_cfg": {
                "start_near_target": False,
            },
            "spawn": {
                "spawn_mode": "random",
                "roi_x": (0.30, 0.75),
                "roi_y": (-0.10, 0.25),
                "object_z": 0.02,
                "random_yaw": True,
            },
            "reward": {
                "w_xy_progress": 1.50,
                "w_z_progress": 1.20,
                "w_yaw_progress": 1.00,
                "w_success": 4.4,
                "w_stable_bonus": 0.06,
                "w_jitter_near": 0.012,
                "w_regress_near": 0.018,
                "w_yaw_regress_near": 0.015,
                "w_action_penalty": 0.0010,
                "w_yaw_action_penalty": 0.0006,
                "w_timeout_penalty": 0.03,
            },
            "noise": {
                "enable_obs_noise": True,
                "pos_noise_std": 0.002,
            },
        }

    raise ValueError(f"Unknown substage: {substage}")


def build_reach_config(substage: str = "1A") -> ReachEnvConfig:
    cfg = ReachEnvConfig()
    cfg.substage = substage
    _apply_nested_overrides(cfg, get_substage_overrides(substage))
    return cfg