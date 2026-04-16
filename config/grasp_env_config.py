from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class SimConfig:
    use_gui: bool = False
    physics_hz: int = 240
    substeps: int = 12
    max_steps: int = 140
    seed: int = 42


@dataclass
class WorkspaceConfig:
    x_range: Tuple[float, float] = (0.30, 0.75)
    y_range: Tuple[float, float] = (-0.10, 0.25)
    z_range: Tuple[float, float] = (0.01, 0.55)


@dataclass
class SpawnConfig:
    spawn_mode: str = "random"
    fixed_pose_xyz: Tuple[float, float, float] = (0.55, 0.05, 0.02)
    roi_x: Tuple[float, float] = (0.45, 0.72)
    roi_y: Tuple[float, float] = (-0.10, 0.23)
    object_z: float = 0.02
    random_yaw: bool = True
    yaw_range: Tuple[float, float] = (-3.14159, 3.14159)
    cube_half_extents: Tuple[float, float, float] = (0.02, 0.02, 0.02)
    object_mass: float = 0.05
    object_colors: Tuple[str, ...] = ("red", "green", "blue", "yellow")


@dataclass
class ActionConfig:
    clip_action: bool = True
    action_scale_xy: float = 0.008
    action_scale_z: float = 0.010
    action_scale_grip: float = 0.012
    action_smoothing: float = 0.08


@dataclass
class TargetConfig:
    pregrasp_z: float = 0.080
    grasp_z: float = 0.038
    lift_z: float = 0.120

    xy_phase_threshold: float = 0.010
    z_phase_threshold: float = 0.005
    yaw_phase_threshold: float = 0.12

    grasp_xy_success_dist: float = 0.012
    grasp_z_success_dist: float = 0.014
    close_bonus_width: float = 0.020

    stable_grasp_steps_required: int = 8
    stable_lift_steps_required: int = 6
    lift_success_delta_z: float = 0.045

    finger_xy_success_dist_2a: float = 0.015
    grip_success_width_2a: float = 0.020
    grasp_success_max_ee_z_margin_2a: float = 0.003

    finger_xy_success_dist: float = 0.015
    finger_z_success_dist: float = 0.045


@dataclass
class GripperConfig:
    open_width: float = 0.08
    close_width: float = 0.0
    initial_width: float = 0.08
    position_force: float = 80.0
    contact_force_threshold: float = 0.0
    close_bonus_width: float = 0.030


@dataclass
class RewardConfig:
    w_xy_progress: float = 1.8
    w_z_progress: float = 2.2
    w_grip_progress: float = 0.8
    w_contact: float = 0.5
    w_dual_contact_bonus: float = 1.5
    w_hold_bonus: float = 0.15
    w_lift_progress: float = 2.4
    w_success: float = 8.0

    w_action_penalty: float = 0.0005
    w_idle_penalty: float = 0.0015
    idle_eps: float = 8e-5
    w_workspace_violation: float = 0.05
    w_timeout_penalty: float = 0.03
    w_xy_regress_penalty: float = 0.02
    w_z_up_penalty: float = 0.04
    w_wrong_close_penalty: float = 0.03
    w_close_no_contact_penalty: float = 0.06
    w_drop_penalty: float = 3.0


@dataclass
class DebugConfig:
    draw_target: bool = True
    draw_ee_path: bool = True


@dataclass
class Stage2GraspConfig:
    substage: str = "2A"
    sim: SimConfig = field(default_factory=SimConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
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
    if substage == "2A":
        return {
            "sim": {"max_steps": 140},
            "action": {
                "action_scale_xy": 0.008,
                "action_scale_z": 0.010,
                "action_scale_grip": 0.012,
                "action_smoothing": 0.08,
            },
            "target": {
                "pregrasp_z": 0.080,
                "grasp_z": 0.021,
                "xy_phase_threshold": 0.010,
                "z_phase_threshold": 0.005,
                "grasp_xy_success_dist": 0.012,
                "grasp_z_success_dist": 0.014,
                "finger_xy_success_dist_2a": 0.015,
                "grip_success_width_2a": 0.020,
            },
        }

    if substage == "2B":
        return {
            "sim": {"max_steps": 170},
            "target": {
                "grasp_z": 0.038,
                "stable_grasp_steps_required": 10,
            },
        }

    if substage == "2C":
        return {
            "sim": {"max_steps": 240},
            "target": {
                "grasp_z": 0.021,
                "yaw_phase_threshold": 0.14,
                "lift_z": 0.120,
                "stable_grasp_steps_required": 8,
                "stable_lift_steps_required": 6,
                "lift_success_delta_z": 0.045,
            },
            "reward": {
                "w_lift_progress": 3.2,
                "w_success": 12.0,
                "w_drop_penalty": 4.0,
            },
        }

    raise ValueError(f"Unknown substage: {substage}")


def build_stage2_grasp_config(substage: str = "2A") -> Stage2GraspConfig:
    cfg = Stage2GraspConfig()
    cfg.substage = substage
    _apply_nested_overrides(cfg, get_substage_overrides(substage))
    return cfg
