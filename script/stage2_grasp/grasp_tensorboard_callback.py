from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class GraspTensorboardCallback(BaseCallback):
    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = int(window_size)

        self.success_window = deque(maxlen=self.window_size)
        self.reward_window = deque(maxlen=self.window_size)
        self.ep_len_window = deque(maxlen=self.window_size)

        self.xy_dist_window = deque(maxlen=self.window_size)
        self.z_dist_window = deque(maxlen=self.window_size)
        self.yaw_error_window = deque(maxlen=self.window_size)
        self.grip_width_window = deque(maxlen=self.window_size)
        self.dual_contact_window = deque(maxlen=self.window_size)
        self.grasp_window = deque(maxlen=self.window_size)
        self.lift_delta_window = deque(maxlen=self.window_size)
        self.hold_steps_window = deque(maxlen=self.window_size)
        self.lift_hold_steps_window = deque(maxlen=self.window_size)
        self.home_hold_steps_window = deque(maxlen=self.window_size)
        self.home_error_window = deque(maxlen=self.window_size)

        self.current_episode_reward = 0.0
        self.current_episode_len = 0

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

    def _extract_done(self) -> bool:
        dones = self.locals.get("dones", None)
        if isinstance(dones, np.ndarray):
            dones = dones.reshape(-1)
        if isinstance(dones, (list, tuple, np.ndarray)):
            return bool(dones[0]) if len(dones) > 0 else False
        return bool(dones) if dones is not None else False

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

        self.xy_dist_window.append(float(info.get("xy_dist", 0.0)))
        self.z_dist_window.append(float(info.get("z_dist", 0.0)))
        self.yaw_error_window.append(float(info.get("yaw_error", 0.0)))
        self.grip_width_window.append(float(info.get("grip_width", 0.0)))
        self.dual_contact_window.append(
            float(bool(info.get("left_contact", False) and info.get("right_contact", False)))
        )
        self.grasp_window.append(float(bool(info.get("grasp_established", False))))
        self.lift_delta_window.append(float(info.get("object_lift_delta", 0.0)))
        self.hold_steps_window.append(float(info.get("stable_pose_steps", 0.0)))
        self.lift_hold_steps_window.append(float(info.get("lift_hold_steps", 0.0)))
        self.home_hold_steps_window.append(float(info.get("home_hold_steps", 0.0)))
        self.home_error_window.append(float(info.get("ee_to_home_xyz_error", 0.0)))

        if done:
            self.success_window.append(float(bool(info.get("success", False))))
            self.reward_window.append(float(self.current_episode_reward))
            self.ep_len_window.append(float(self.current_episode_len))
            self.current_episode_reward = 0.0
            self.current_episode_len = 0

        self.logger.record(f"episode/success_rate_{self.window_size}", self._safe_mean(self.success_window))
        self.logger.record(f"episode/reward_mean_{self.window_size}", self._safe_mean(self.reward_window))
        self.logger.record(f"episode/ep_len_mean_{self.window_size}", self._safe_mean(self.ep_len_window))

        self.logger.record(f"grasp/xy_dist_mean_{self.window_size}", self._safe_mean(self.xy_dist_window))
        self.logger.record(f"grasp/z_dist_mean_{self.window_size}", self._safe_mean(self.z_dist_window))
        self.logger.record(f"grasp/yaw_error_mean_{self.window_size}", self._safe_mean(self.yaw_error_window))
        self.logger.record(f"grasp/grip_width_mean_{self.window_size}", self._safe_mean(self.grip_width_window))
        self.logger.record(f"grasp/dual_contact_rate_{self.window_size}", self._safe_mean(self.dual_contact_window))
        self.logger.record(f"grasp/grasp_established_rate_{self.window_size}", self._safe_mean(self.grasp_window))
        self.logger.record(f"grasp/lift_delta_mean_{self.window_size}", self._safe_mean(self.lift_delta_window))
        self.logger.record(f"grasp/hold_steps_mean_{self.window_size}", self._safe_mean(self.hold_steps_window))
        self.logger.record(f"grasp/lift_hold_steps_mean_{self.window_size}", self._safe_mean(self.lift_hold_steps_window))
        self.logger.record(f"grasp/home_hold_steps_mean_{self.window_size}", self._safe_mean(self.home_hold_steps_window))
        self.logger.record(f"grasp/home_error_mean_{self.window_size}", self._safe_mean(self.home_error_window))

        return True