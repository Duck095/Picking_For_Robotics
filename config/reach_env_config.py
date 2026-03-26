from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class SimConfig:
    use_gui: bool = False
    physics_hz: int = 120
    substeps: int = 6
    max_steps: int = 220
    seed: int = 42


@dataclass
class WorkspaceConfig:
    x_range: Tuple[float, float] = (0.10, 0.85)
    y_range: Tuple[float, float] = (-0.60, 0.60)
    z_range: Tuple[float, float] = (0.05, 0.55)


@dataclass
class ActionConfig:
    clip_action: bool = True
    action_scale_xy: float = 0.024
    action_scale_z: float = 0.028

    # smoothing theo substage thực tế được env dùng linh hoạt
    action_smoothing: float = 0.15

    # các ngưỡng pha để env scale action theo phase
    near_dist: float = 0.14
    settle_dist: float = 0.09
    descend_z_gate: float = 0.055


@dataclass
class TargetConfig:
    use_hover_target: bool = True
    z_hover: float = 0.18
    success_dist: float = 0.08
    xy_align_threshold: float = 0.05


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
    # progress core
    w_dist_progress: float = 1.0
    w_xy_progress: float = 1.6
    w_z_progress: float = 1.2
    w_success: float = 3.0

    # penalties chung
    w_idle_penalty: float = 0.0015
    idle_eps: float = 8e-5
    w_wrong_descend_penalty: float = 0.008
    w_action_penalty: float = 0.0004
    w_workspace_violation: float = 0.05
    w_timeout_penalty: float = 0.02

    # shaping theo phase
    w_descend_bonus: float = 0.03
    w_xy_regress_descend_penalty: float = 0.01
    w_hang_high_penalty: float = 0.015
    w_no_descend_penalty: float = 0.008

    # anti-jitter / anti-stall
    w_jitter_near: float = 0.004
    w_reverse_action_near: float = 0.008
    hang_steps_tolerance: int = 6


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
            "sim": {
                "max_steps": 220,
            },
            "action": {
                "action_scale_xy": 0.024,
                "action_scale_z": 0.030,
                "action_smoothing": 0.12,
                "near_dist": 0.15,
                "settle_dist": 0.09,
                "descend_z_gate": 0.055,
            },
            "target": {
                "success_dist": 0.08,
                "xy_align_threshold": 0.05,
                "z_hover": 0.18,
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
            },
            "noise": {
                "enable_obs_noise": False,
            },
        }

    if substage == "1B":
        return {
            "sim": {
                "max_steps": 210,
            },
            "action": {
                "action_scale_xy": 0.022,
                "action_scale_z": 0.026,
                "action_smoothing": 0.20,
                "near_dist": 0.13,
                "settle_dist": 0.08,
                "descend_z_gate": 0.050,
            },
            "target": {
                "success_dist": 0.065,
                "xy_align_threshold": 0.035,
                "z_hover": 0.18,
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
            },
            "noise": {
                "enable_obs_noise": False,
            },
        }

    if substage == "1C":
        return {
            "sim": {
                "max_steps": 200,
            },
            "action": {
                "action_scale_xy": 0.018,
                "action_scale_z": 0.020,
                "action_smoothing": 0.28,
                "near_dist": 0.11,
                "settle_dist": 0.070,
                "descend_z_gate": 0.045,
            },
            "target": {
                "success_dist": 0.035,
                "xy_align_threshold": 0.0225,
                "z_hover": 0.18,
            },
            "spawn": {
                "spawn_mode": "random",
                "roi_x": (0.30, 0.75),
                "roi_y": (-0.10, 0.25),
                "object_z": 0.02,
                "random_yaw": True,
            },
            "reward": {
                "w_dist_progress": 0.9,
                "w_xy_progress": 1.25,
                "w_z_progress": 1.0,
                "w_success": 2.8,
                "w_idle_penalty": 0.0022,
                "w_wrong_descend_penalty": 0.022,
                "w_action_penalty": 0.0010,
                "w_timeout_penalty": 0.03,
                "w_descend_bonus": 0.02,
                "w_xy_regress_descend_penalty": 0.018,
                "w_hang_high_penalty": 0.022,
                "w_no_descend_penalty": 0.014,
                "w_jitter_near": 0.007,
                "w_reverse_action_near": 0.014,
                "hang_steps_tolerance": 4,
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
