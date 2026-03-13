# script/stage1_reach/debug_callbacks.py
import os
import csv
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback


class ReachDebugLoggerCallback(BaseCallback):
    """
    Ghi 2 file CSV:
    1) debug_csv_path   : mỗi step một dòng
    2) summary_csv_path : mỗi episode một dòng
    """

    def __init__(self, debug_csv_path: str, summary_csv_path: str, n_envs: int, verbose=0):
        super().__init__(verbose)
        self.debug_csv_path = debug_csv_path
        self.summary_csv_path = summary_csv_path
        self.n_envs = n_envs

        self.debug_header_written = False
        self.summary_header_written = False

        self.episode_counts = [0 for _ in range(n_envs)]
        self.episode_reward_sums = [0.0 for _ in range(n_envs)]
        self.episode_step_counts = [0 for _ in range(n_envs)]
        self.episode_min_dists = [float("inf") for _ in range(n_envs)]
        self.episode_start_dists = [None for _ in range(n_envs)]

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.debug_csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.summary_csv_path), exist_ok=True)

        self.debug_header_written = (
            os.path.exists(self.debug_csv_path)
            and os.path.getsize(self.debug_csv_path) > 0
        )
        self.summary_header_written = (
            os.path.exists(self.summary_csv_path)
            and os.path.getsize(self.summary_csv_path) > 0
        )

    def _safe_tuple3(self, v):
        if v is None:
            return (None, None, None)
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            return (v[0], v[1], v[2])
        return (None, None, None)

    def _write_debug_row(self, row: dict):
        with open(self.debug_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not self.debug_header_written:
                writer.writeheader()
                self.debug_header_written = True
            writer.writerow(row)

    def _write_summary_row(self, row: dict):
        with open(self.summary_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not self.summary_header_written:
                writer.writeheader()
                self.summary_header_written = True
            writer.writerow(row)

    def _reset_episode_stats(self, env_idx: int):
        self.episode_reward_sums[env_idx] = 0.0
        self.episode_step_counts[env_idx] = 0
        self.episode_min_dists[env_idx] = float("inf")
        self.episode_start_dists[env_idx] = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        for env_idx in range(len(infos)):
            info = infos[env_idx] if env_idx < len(infos) else {}
            reward = float(rewards[env_idx]) if env_idx < len(rewards) else None
            done = bool(dones[env_idx]) if env_idx < len(dones) else False

            ee_pos = self._safe_tuple3(info.get("ee_pos"))
            obj_pos = self._safe_tuple3(info.get("obj_pos"))
            target_ee_pos = self._safe_tuple3(info.get("target_ee_pos"))

            action = info.get("action", None)
            if isinstance(action, (list, tuple)) and len(action) >= 3:
                action_x, action_y, action_z = action[0], action[1], action[2]
            else:
                action_x, action_y, action_z = None, None, None

            dist = info.get("ee_obj_dist", None)
            success = bool(info.get("success", False))

            if self.episode_start_dists[env_idx] is None and dist is not None:
                self.episode_start_dists[env_idx] = dist

            if dist is not None:
                self.episode_min_dists[env_idx] = min(self.episode_min_dists[env_idx], dist)

            if reward is not None:
                self.episode_reward_sums[env_idx] += reward

            self.episode_step_counts[env_idx] += 1

            delta_x = None if (ee_pos[0] is None or obj_pos[0] is None) else (obj_pos[0] - ee_pos[0])
            delta_y = None if (ee_pos[1] is None or obj_pos[1] is None) else (obj_pos[1] - ee_pos[1])
            delta_z = None if (ee_pos[2] is None or obj_pos[2] is None) else (obj_pos[2] - ee_pos[2])

            debug_row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "global_timesteps": self.num_timesteps,
                "env_idx": env_idx + 1,  # hiển thị 1..8 cho dễ đọc
                "episode_idx_env": self.episode_counts[env_idx],
                "step_count": info.get("step_count", None),
                "stage1_substage": info.get("stage1_substage", ""),
                "reward": reward,
                "success": success,
                "done": done,
                "ee_obj_dist": dist,
                "success_dist": info.get("success_dist", None),
                "ee_x": ee_pos[0],
                "ee_y": ee_pos[1],
                "ee_z": ee_pos[2],
                "obj_x": obj_pos[0],
                "obj_y": obj_pos[1],
                "obj_z": obj_pos[2],
                "target_ee_x": target_ee_pos[0],
                "target_ee_y": target_ee_pos[1],
                "target_ee_z": target_ee_pos[2],
                "action_x": action_x,
                "action_y": action_y,
                "action_z": action_z,
                "delta_x": delta_x,
                "delta_y": delta_y,
                "delta_z": delta_z,
            }
            self._write_debug_row(debug_row)

            if done:
                ep_len = self.episode_step_counts[env_idx]
                ep_total_reward = self.episode_reward_sums[env_idx]
                ep_final_dist = dist
                ep_start_dist = self.episode_start_dists[env_idx]
                ep_min_dist = None if self.episode_min_dists[env_idx] == float("inf") else self.episode_min_dists[env_idx]
                ep_success = int(success)
                ep_mean_reward = ep_total_reward / ep_len if ep_len > 0 else None

                summary_row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "global_timesteps": self.num_timesteps,
                    "env_idx": env_idx + 1,
                    "episode_idx_env": self.episode_counts[env_idx],
                    "stage1_substage": info.get("stage1_substage", ""),
                    "episode_len": ep_len,
                    "episode_total_reward": ep_total_reward,
                    "episode_mean_reward": ep_mean_reward,
                    "episode_start_dist": ep_start_dist,
                    "episode_final_dist": ep_final_dist,
                    "episode_min_dist": ep_min_dist,
                    "episode_success": ep_success,
                    "final_ee_x": ee_pos[0],
                    "final_ee_y": ee_pos[1],
                    "final_ee_z": ee_pos[2],
                    "obj_x": obj_pos[0],
                    "obj_y": obj_pos[1],
                    "obj_z": obj_pos[2],
                    "terminated": info.get("terminated", None),
                    "truncated": info.get("truncated", None),
                }
                self._write_summary_row(summary_row)

                self.episode_counts[env_idx] += 1
                self._reset_episode_stats(env_idx)

        return True