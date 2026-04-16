from __future__ import annotations

from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ReachDebugSummaryCallback(BaseCallback):
    def __init__(
        self,
        log_dir: str,
        file_name: str = "debug_summary.log",
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

        # global rolling windows
        self.success_window = deque(maxlen=self.window_size)
        self.reward_window = deque(maxlen=self.window_size)
        self.ep_len_window = deque(maxlen=self.window_size)
        self.dist_window = deque(maxlen=self.window_size)
        self.xy_dist_window = deque(maxlen=self.window_size)
        self.z_dist_window = deque(maxlen=self.window_size)
        self.stable_pose_steps_window = deque(maxlen=self.window_size)
        self.yaw_error_window = deque(maxlen=self.window_size)

        # current episode metrics
        self.current_episode_reward = 0.0
        self.current_episode_len = 0

        # rolling windows by curriculum level
        self.level_stats: Dict[str, Dict[str, deque]] = {
            "easy": defaultdict(lambda: deque(maxlen=self.window_size)),
            "medium": defaultdict(lambda: deque(maxlen=self.window_size)),
            "hard": defaultdict(lambda: deque(maxlen=self.window_size)),
        }

    def _on_training_start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.file_path, "a", encoding="utf-8")
        self._fp.write("\n" + "=" * 100 + "\n")
        self._fp.write(f"[START] {datetime.now()}\n")
        self._fp.write("=" * 100 + "\n")
        self._fp.flush()

        if self.verbose > 0:
            print(f"[DEBUG_SUMMARY] file={self.file_path}")

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

        # accumulate current episode
        self.current_episode_reward += reward
        self.current_episode_len += 1

        if done:
            sub = str(info.get("substage", "UNK"))
            level = str(info.get("curriculum_level", "unknown"))

            success = float(bool(info.get("success", False)))
            final_dist = float(info.get("dist", -1.0))
            final_xy = float(info.get("xy_dist", -1.0))
            final_z = float(info.get("z_dist", -1.0))
            stable_pose_steps = float(info.get("stable_pose_steps", 0.0))
            final_yaw_error = float(info.get("yaw_error", 0.0))

            # global window
            self.success_window.append(success)
            self.reward_window.append(float(self.current_episode_reward))
            self.ep_len_window.append(float(self.current_episode_len))
            self.dist_window.append(final_dist)
            self.xy_dist_window.append(final_xy)
            self.z_dist_window.append(final_z)
            self.stable_pose_steps_window.append(stable_pose_steps)
            self.yaw_error_window.append(final_yaw_error)

            # level-specific window
            self.level_stats[level]["success"].append(success)
            self.level_stats[level]["reward"].append(float(self.current_episode_reward))
            self.level_stats[level]["ep_len"].append(float(self.current_episode_len))
            self.level_stats[level]["dist"].append(final_dist)
            self.level_stats[level]["xy"].append(final_xy)
            self.level_stats[level]["z"].append(final_z)
            self.level_stats[level]["stable"].append(stable_pose_steps)
            self.level_stats[level]["yaw"].append(final_yaw_error)

            # build global summary line
            line = (
                f"[SUMMARY] "
                f"t={self.num_timesteps:<8d} "
                f"sub={sub:<4s} "
                f"curr={level:<6} "
                f"success_rate_{self.window_size}={self._safe_mean(self.success_window):.3f} "
                f"reward_mean_{self.window_size}={self._safe_mean(self.reward_window):.3f} "
                f"ep_len_mean_{self.window_size}={self._safe_mean(self.ep_len_window):.1f} "
                f"dist_mean_{self.window_size}={self._safe_mean(self.dist_window):.4f} "
                f"xy_mean_{self.window_size}={self._safe_mean(self.xy_dist_window):.4f} "
                f"z_mean_{self.window_size}={self._safe_mean(self.z_dist_window):.4f} "
                f"stable_mean_{self.window_size}={self._safe_mean(self.stable_pose_steps_window):.2f} "
                f"yaw_mean_{self.window_size}={self._safe_mean(self.yaw_error_window):.4f}"
            )

            # print and log
            self._fp.write(line + "\n")
            self._fp.flush()

            if self.verbose > 0 and self.num_timesteps - self._last_print >= self.print_freq:
                print(line)
                self._last_print = self.num_timesteps

            # reset episode accumulators
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