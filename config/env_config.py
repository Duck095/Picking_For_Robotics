# config/env_config.py
from dataclasses import dataclass
from typing import Tuple


@dataclass
class EnvConfig:
    # ---------- Observation ----------
    img_size: int = 84
    frame_stack: int = 4

    # ---------- Physics ----------
    physics_hz: int = 240
    rl_hz: int = 20
    substeps: int = 12  # physics_hz // rl_hz

    # ---------- Episode ----------
    max_steps: int = 250

    # ---------- Action scale (delta EE per RL step) ----------
    action_scale_xy: float = 0.04
    action_scale_z: float = 0.04

    # ---------- Workspace clamp (TCP target) ----------
    x_range: Tuple[float, float] = (0.10, 0.85)
    y_range: Tuple[float, float] = (-0.60, 0.60)
    z_range: Tuple[float, float] = (0.03, 0.55)

    # ---------- Stage 1 (Reach) curriculum ----------
    # substage: "1A" (easy) or "1B" (hard)
    stage1_substage: str = "1A"

    # spawn ranges per substage
    stage1a_x: Tuple[float, float] = (0.53, 0.57)
    stage1a_y: Tuple[float, float] = (-0.03, 0.03)

    stage1b_x: Tuple[float, float] = (0.47, 0.63)
    stage1b_y: Tuple[float, float] = (-0.18, 0.18)

    obj_z: float = 0.02

    # success condition (curriculum)
    success_dist_1a: float = 0.10   # ✅ dễ hơn để bật success nhanh
    success_dist_1b: float = 0.06   # ✅ giữ chuẩn cho 1B

    # reward shaping
    dist_weight: float = 0.60       # ✅ kéo mạnh hơn về mục tiêu
    time_penalty: float = 0.01
    success_bonus: float = 3.0

    # optional: clip progress per step
    delta_clip: float = 0.04        # ✅ tăng nhẹ (từ 0.02 -> 0.04)