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

        self.dist_window = deque(maxlen=self.window_size)
        self.xy_dist_window = deque(maxlen=self.window_size)
        self.z_dist_window = deque(maxlen=self.window_size)
        self.yaw_error_window = deque(maxlen=self.window_size)

        self.grip_width_window = deque(maxlen=self.window_size)
        self.left_contact_window = deque(maxlen=self.window_size)
        self.right_contact_window = deque(maxlen=self.window_size)
        self.dual_contact_window = deque(maxlen=self.window_size)
        self.grasp_window = deque(maxlen=self.window_size)
        self.lift_delta_window = deque(maxlen=self.window_size)
        self.hold_steps_window = deque(maxlen=self.window_size)
        self.lift_hold_steps_window = deque(maxlen=self.window_size)

        self.current_episode_reward = 0.0
        self.current_episode_len = 0

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

        if "dist" in info:
            self.dist_window.append(float(info["dist"]))
        if "xy_dist" in info:
            self.xy_dist_window.append(float(info["xy_dist"]))
        if "z_dist" in info:
            self.z_dist_window.append(float(info["z_dist"]))
        if "yaw_error" in info:
            self.yaw_error_window.append(float(info["yaw_error"]))
        if "grip_width" in info:
            self.grip_width_window.append(float(info["grip_width"]))
        if "left_contact" in info:
            self.left_contact_window.append(float(bool(info["left_contact"])))
        if "right_contact" in info:
            self.right_contact_window.append(float(bool(info["right_contact"])))
        if "left_contact" in info and "right_contact" in info:
            self.dual_contact_window.append(
                float(bool(info["left_contact"] and info["right_contact"]))
            )
        if "grasp_established" in info:
            self.grasp_window.append(float(bool(info["grasp_established"])))
        if "object_lift_delta" in info:
            self.lift_delta_window.append(float(info["object_lift_delta"]))
        if "stable_pose_steps" in info:
            self.hold_steps_window.append(float(info["stable_pose_steps"]))
        if "lift_hold_steps" in info:
            self.lift_hold_steps_window.append(float(info["lift_hold_steps"]))

        if done:
            success = float(bool(info.get("success", False)))
            self.success_window.append(success)
            self.reward_window.append(float(self.current_episode_reward))
            self.ep_len_window.append(float(self.current_episode_len))

            self.current_episode_reward = 0.0
            self.current_episode_len = 0

        self.logger.record(
            f"episode/success_rate_{self.window_size}",
            self._safe_mean(self.success_window),
        )
        self.logger.record(
            f"episode/reward_mean_{self.window_size}",
            self._safe_mean(self.reward_window),
        )
        self.logger.record(
            f"episode/ep_len_mean_{self.window_size}",
            self._safe_mean(self.ep_len_window),
        )

        self.logger.record(
            f"grasp/dist_mean_{self.window_size}",
            self._safe_mean(self.dist_window),
        )
        self.logger.record(
            f"grasp/xy_dist_mean_{self.window_size}",
            self._safe_mean(self.xy_dist_window),
        )
        self.logger.record(
            f"grasp/z_dist_mean_{self.window_size}",
            self._safe_mean(self.z_dist_window),
        )
        self.logger.record(
            f"grasp/yaw_error_mean_{self.window_size}",
            self._safe_mean(self.yaw_error_window),
        )
        self.logger.record(
            f"grasp/grip_width_mean_{self.window_size}",
            self._safe_mean(self.grip_width_window),
        )
        self.logger.record(
            f"grasp/left_contact_rate_{self.window_size}",
            self._safe_mean(self.left_contact_window),
        )
        self.logger.record(
            f"grasp/right_contact_rate_{self.window_size}",
            self._safe_mean(self.right_contact_window),
        )
        self.logger.record(
            f"grasp/dual_contact_rate_{self.window_size}",
            self._safe_mean(self.dual_contact_window),
        )
        self.logger.record(
            f"grasp/grasp_established_rate_{self.window_size}",
            self._safe_mean(self.grasp_window),
        )
        self.logger.record(
            f"grasp/lift_delta_mean_{self.window_size}",
            self._safe_mean(self.lift_delta_window),
        )
        self.logger.record(
            f"grasp/hold_steps_mean_{self.window_size}",
            self._safe_mean(self.hold_steps_window),
        )
        self.logger.record(
            f"grasp/lift_hold_steps_mean_{self.window_size}",
            self._safe_mean(self.lift_hold_steps_window),
        )

        return True