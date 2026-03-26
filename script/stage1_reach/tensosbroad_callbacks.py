from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ReachTensorboardCallback(BaseCallback):
    """
    TensorBoard callback tối giản cho Stage 1 - Reach.

    Chỉ log những metric thật sự quan trọng để trả lời:
    1. Model có học không?
    2. Model có tới gần target hơn không?
    3. PPO có đang train ổn định không?

    --------------------------------------------------------
    Metric được giữ lại
    --------------------------------------------------------
    episode/
        - success_rate_100
        - reward_mean_100
        - ep_len_mean_100

    reach/
        - dist_mean_100
        - xy_dist_mean_100
        - z_dist_mean_100

    --------------------------------------------------------
    Không log
    --------------------------------------------------------
    - reward breakdown chi tiết
    - penalty breakdown
    - step-level metrics
    - flags kiểu is_1A/is_1B/is_1C
    - workspace_violated mỗi step
    """

    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)

        self.window_size = int(window_size)

        # rolling window buffers
        self.success_window = deque(maxlen=self.window_size)
        self.reward_window = deque(maxlen=self.window_size)
        self.ep_len_window = deque(maxlen=self.window_size)

        self.dist_window = deque(maxlen=self.window_size)
        self.xy_dist_window = deque(maxlen=self.window_size)
        self.z_dist_window = deque(maxlen=self.window_size)

        # current episode accumulators
        self.current_episode_reward = 0.0
        self.current_episode_len = 0

    def _extract_info(self) -> Optional[Dict[str, Any]]:
        """
        Lấy info từ SB3 locals.

        Với VecEnv, infos thường là list[dict].
        Ta lấy env đầu tiên để log đơn giản và ổn định.
        """
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

        # accumulate current episode
        self.current_episode_reward += reward
        self.current_episode_len += 1

        # only keep geometry metrics
        if "dist" in info:
            self.dist_window.append(float(info["dist"]))
        if "xy_dist" in info:
            self.xy_dist_window.append(float(info["xy_dist"]))
        if "z_dist" in info:
            self.z_dist_window.append(float(info["z_dist"]))

        # episode end -> update rolling episode metrics
        if done:
            success = float(bool(info.get("success", False)))

            self.success_window.append(success)
            self.reward_window.append(float(self.current_episode_reward))
            self.ep_len_window.append(float(self.current_episode_len))

            self.current_episode_reward = 0.0
            self.current_episode_len = 0

        # ----------------------------------------------------
        # Episode-level rolling metrics
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # Reach geometry rolling metrics
        # ----------------------------------------------------
        self.logger.record(
            f"reach/dist_mean_{self.window_size}",
            self._safe_mean(self.dist_window),
        )
        self.logger.record(
            f"reach/xy_dist_mean_{self.window_size}",
            self._safe_mean(self.xy_dist_window),
        )
        self.logger.record(
            f"reach/z_dist_mean_{self.window_size}",
            self._safe_mean(self.z_dist_window),
        )

        return True