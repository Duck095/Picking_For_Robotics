from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ReachDebugStepCallback(BaseCallback):
    """
    Debug step callback kiểu production.

    Mục tiêu:
    - Ghi log step-level gọn, dễ đọc
    - Không dump quá nhiều reward breakdown
    - Chỉ giữ các metric cần để debug motion ngắn hạn

    File output:
        debug_logs/stage1_<substage>_debug.log

    Mỗi dòng log:
        [STEP] t=... ep=... step=... sub=... r=... dist=... xy=... z=... success=...

    Lưu ý:
    - Step log vẫn hữu ích khi debug env/reward
    - Nhưng không nên log quá nhiều field gây nhiễu
    """

    def __init__(
        self,
        log_dir: str,
        file_name: str = "debug_step.log",
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

        if self.verbose > 0:
            print(f"[DEBUG_STEP] file={self.file_path}")

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

    def _on_step(self) -> bool:
        info = self._extract_info()
        if info is None:
            return True

        reward = self._extract_reward()

        ep = int(info.get("episode_idx", -1))
        ep_step = int(info.get("step", -1))
        sub = str(info.get("substage", "UNK"))

        dist = float(info.get("dist", -1.0))
        xy_dist = float(info.get("xy_dist", -1.0))
        z_dist = float(info.get("z_dist", -1.0))
        success = bool(info.get("success", False))
        truncated = bool(info.get("truncated", False))
        xy_aligned = bool(info.get("xy_aligned", False))

        line = (
            f"[STEP] "
            f"t={self.num_timesteps:<8d} "
            f"ep={ep:<5d} "
            f"step={ep_step:<4d} "
            f"sub={sub:<2s} "
            f"r={reward:+.4f} "
            f"dist={dist:.4f} "
            f"xy={xy_dist:.4f} "
            f"z={z_dist:.4f} "
            f"aligned={int(xy_aligned)} "
            f"success={int(success)} "
            f"truncated={int(truncated)}"
        )

        self._fp.write(line + "\n")
        self._fp.flush()

        # chỉ print định kỳ
        if self.num_timesteps - self._last_print >= self.print_freq:
            print(line)
            self._last_print = self.num_timesteps

        return True

    def _on_training_end(self) -> None:
        if self._fp is not None:
            self._fp.write("=" * 100 + "\n")
            self._fp.write(f"[END] {datetime.now()} | total_timesteps={self.num_timesteps}\n")
            self._fp.write("=" * 100 + "\n")
            self._fp.flush()
            self._fp.close()
            self._fp = None