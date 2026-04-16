from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class GraspDebugSummaryCallback(BaseCallback):
    def __init__(
        self,
        log_dir: str,
        file_name: str = "grasp_debug_summary.log",
        window_size: int = 100,
        print_freq: int = 10000,
        verbose: int = 1,
    ):
        super().__init__(verbose)

        self.log_dir = Path(log_dir)
        self.file_path = self.log_dir / file_name

        self.window_size = int(window_size)
        self.print_freq = int(print_freq)

        self._fp = None
        self._last_print = 0

        self.success_window = deque(maxlen=self.window_size)
        self.reward_window = deque(maxlen=self.window_size)
        self.ep_len_window = deque(maxlen=self.window_size)

        self.dist_window = deque(maxlen=self.window_size)
        self.xy_dist_window = deque(maxlen=self.window_size)
        self.z_dist_window = deque(maxlen=self.window_size)
        self.yaw_error_window = deque(maxlen=self.window_size)

        self.grip_width_window = deque(maxlen=self.window_size)
        self.grasp_window = deque(maxlen=self.window_size)
        self.left_contact_window = deque(maxlen=self.window_size)
        self.right_contact_window = deque(maxlen=self.window_size)
        self.dual_contact_window = deque(maxlen=self.window_size)
        self.lift_delta_window = deque(maxlen=self.window_size)
        self.hold_steps_window = deque(maxlen=self.window_size)
        self.lift_hold_steps_window = deque(maxlen=self.window_size)

        self.current_episode_reward = 0.0
        self.current_episode_len = 0

    def _on_training_start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.file_path, "a", encoding="utf-8")
        self._fp.write("\n" + "=" * 100 + "\n")
        self._fp.write(f"[START] {datetime.now()}\n")
        self._fp.write("=" * 100 + "\n")
        self._fp.flush()

        if self.verbose > 0:
            print(f"[GRASP_DEBUG_SUMMARY] file={self.file_path}")

    def _extract_info(self) -> Optional[Dict[str, Any]]:
        infos = self.locals.get("infos", None)
        if infos is None:
            return None

        if isinstance(infos, (list, tuple)) and len(infos) > 0:
            if isinstance(infos[0], dict):
                return infos[0]

        if isinstance(infos, dict):
            return infos

        return None

    def _extract_reward(self) -> float:
        rewards = self.locals.get("rewards", None)
        if rewards is None:
            return 0.0

        if isinstance(rewards, (list, tuple)):
            return float(rewards[0]) if len(rewards) > 0 else 0.0

        if isinstance(rewards, np.ndarray):
            flat = rewards.reshape(-1)
            return float(flat[0]) if flat.size > 0 else 0.0

        return float(rewards)

    def _extract_done(self) -> bool:
        dones = self.locals.get("dones", None)
        if dones is None:
            return False

        if isinstance(dones, (list, tuple)):
            return bool(dones[0]) if len(dones) > 0 else False

        if isinstance(dones, np.ndarray):
            flat = dones.reshape(-1)
            return bool(flat[0]) if flat.size > 0 else False

        return bool(dones)

    @staticmethod
    def _safe_mean(buf) -> float:
        return float(np.mean(buf)) if len(buf) > 0 else 0.0

    def _on_step(self) -> bool:
        info = self._extract_info()
        reward = self._extract_reward()
        done = self._extract_done()

        if info is None:
            return True

        self.current_episode_reward += reward
        self.current_episode_len += 1

        if done:
            sub = str(info.get("substage", "UNK"))

            success = float(bool(info.get("success", False)))
            final_dist = float(info.get("dist", -1.0))
            final_xy = float(info.get("xy_dist", -1.0))
            final_z = float(info.get("z_dist", -1.0))
            final_yaw = float(info.get("yaw_error", 0.0))

            grip_width = float(info.get("grip_width", 0.0))
            left_contact = float(bool(info.get("left_contact", False)))
            right_contact = float(bool(info.get("right_contact", False)))
            dual_contact = float(bool(info.get("left_contact", False) and info.get("right_contact", False)))
            grasp_established = float(bool(info.get("grasp_established", False)))
            lift_delta = float(info.get("object_lift_delta", 0.0))
            hold_steps = float(info.get("stable_pose_steps", 0.0))
            lift_hold_steps = float(info.get("lift_hold_steps", 0.0))

            self.success_window.append(success)
            self.reward_window.append(float(self.current_episode_reward))
            self.ep_len_window.append(float(self.current_episode_len))

            self.dist_window.append(final_dist)
            self.xy_dist_window.append(final_xy)
            self.z_dist_window.append(final_z)
            self.yaw_error_window.append(final_yaw)

            self.grip_width_window.append(grip_width)
            self.left_contact_window.append(left_contact)
            self.right_contact_window.append(right_contact)
            self.dual_contact_window.append(dual_contact)
            self.grasp_window.append(grasp_established)
            self.lift_delta_window.append(lift_delta)
            self.hold_steps_window.append(hold_steps)
            self.lift_hold_steps_window.append(lift_hold_steps)

            line = (
                f"[SUMMARY] "
                f"t={self.num_timesteps:<8d} "
                f"sub={sub:<4s} "
                f"success_rate_{self.window_size}={self._safe_mean(self.success_window):.3f} "
                f"reward_mean_{self.window_size}={self._safe_mean(self.reward_window):.3f} "
                f"ep_len_mean_{self.window_size}={self._safe_mean(self.ep_len_window):.1f} "
                f"dist_mean_{self.window_size}={self._safe_mean(self.dist_window):.4f} "
                f"xy_mean_{self.window_size}={self._safe_mean(self.xy_dist_window):.4f} "
                f"z_mean_{self.window_size}={self._safe_mean(self.z_dist_window):.4f} "
                f"yaw_mean_{self.window_size}={self._safe_mean(self.yaw_error_window):.4f} "
                f"grip_mean_{self.window_size}={self._safe_mean(self.grip_width_window):.4f} "
                f"grasp_rate_{self.window_size}={self._safe_mean(self.grasp_window):.3f} "
                f"dual_contact_rate_{self.window_size}={self._safe_mean(self.dual_contact_window):.3f} "
                f"lift_dz_mean_{self.window_size}={self._safe_mean(self.lift_delta_window):.4f} "
                f"hold_mean_{self.window_size}={self._safe_mean(self.hold_steps_window):.2f} "
                f"lift_hold_mean_{self.window_size}={self._safe_mean(self.lift_hold_steps_window):.2f}"
            )

            self._fp.write(line + "\n")
            self._fp.flush()

            if self.verbose > 0 and self.num_timesteps - self._last_print >= self.print_freq:
                print(line)
                self._last_print = self.num_timesteps

            self.current_episode_reward = 0.0
            self.current_episode_len = 0

        return True

    def _on_training_end(self) -> None:
        if self._fp is not None:
            self._fp.write("=" * 100 + "\n")
            self._fp.write(f"[END] {datetime.now()} | total_timesteps={self.num_timesteps}\n")
            self._fp.write("=" * 100 + "\n")
            self._fp.flush()
            self._fp.close()
            self._fp = None