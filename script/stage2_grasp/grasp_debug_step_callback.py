from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class GraspDebugStepCallback(BaseCallback):
    def __init__(
        self,
        log_dir: str,
        file_name: str = "grasp_debug_step.log",
        print_freq: int = 5000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.file_path = self.log_dir / file_name
        self.print_freq = int(print_freq)
        self._fp = None
        self._last_print = 0

    def _on_training_start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.file_path, "a", encoding="utf-8")
        self._fp.write("\n" + "=" * 100 + "\n")
        self._fp.write(f"[START] {datetime.now()}\n")
        self._fp.write("=" * 100 + "\n")
        self._fp.flush()
        print(f"[GRASP_DEBUG_STEP] file={self.file_path}")

    def _extract_info(self) -> Optional[Dict[str, Any]]:
        infos = self.locals.get("infos", None)
        if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
            return infos[0]
        if isinstance(infos, dict):
            return infos
        return None

    def _extract_reward(self) -> float:
        rewards = self.locals.get("rewards", None)
        if isinstance(rewards, np.ndarray):
            rewards = rewards.reshape(-1)
        if isinstance(rewards, (list, tuple, np.ndarray)):
            return float(rewards[0]) if len(rewards) > 0 else 0.0
        return float(rewards) if rewards is not None else 0.0

    def _on_step(self) -> bool:
        info = self._extract_info()
        if info is None:
            return True

        reward = self._extract_reward()

        line = (
            f"[STEP] "
            f"t={self.num_timesteps:<8d} "
            f"ep={int(info.get('episode_idx', -1)):<5d} "
            f"step={int(info.get('step', -1)):<4d} "
            f"sub={str(info.get('substage', 'UNK')):<4s} "
            f"phase={str(info.get('phase', 'UNK')):<11s} "
            f"hold={int(info.get('stable_pose_steps', 0)):<3d} "
            f"lift_hold={int(info.get('lift_hold_steps', 0)):<3d} "
            f"home_hold={int(info.get('home_hold_steps', 0)):<3d} "
            f"r={reward:+.4f} "
            f"dist={float(info.get('dist', -1.0)):.4f} "
            f"xy={float(info.get('xy_dist', -1.0)):.4f} "
            f"z={float(info.get('z_dist', -1.0)):.4f} "
            f"yaw={float(info.get('yaw_error', 0.0)):.4f} "
            f"grip={float(info.get('grip_width', -1.0)):.4f} "
            f"contact=({int(bool(info.get('left_contact', False)))},{int(bool(info.get('right_contact', False)))}) "
            f"grasp={int(bool(info.get('grasp_established', False)))} "
            f"lift_dz={float(info.get('object_lift_delta', 0.0)):.4f} "
            f"home_err={float(info.get('ee_to_home_xyz_error', -1.0)):.4f} "
            f"success={int(bool(info.get('success', False)))} "
            f"truncated={int(bool(info.get('truncated', False)))}"
        )

        self._fp.write(line + "\n")
        self._fp.flush()

        if self.num_timesteps - self._last_print >= self.print_freq:
            print(line)
            self._last_print = self.num_timesteps

        return True

    def _on_training_end(self) -> None:
        if self._fp is not None:
            self._fp.write("=" * 100 + "\n")
            self._fp.write(f"[END] {datetime.now()} | total_timesteps={self.num_timesteps}\n")
            self._fp.write("=" * 100 + "\n")
            self._fp.close()
            self._fp = None