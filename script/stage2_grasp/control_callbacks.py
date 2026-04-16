from __future__ import annotations

import multiprocessing
import signal
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\n[CONTROL] Received signal {signum}. Will save latest checkpoint and stop safely.")


if multiprocessing.current_process().name == "MainProcess":
    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)


class SaveLatestOnStopCallback(BaseCallback):
    def __init__(
        self,
        save_dir: str,
        latest_name: str = "latest",
        final_name: str = "final",
        save_final_on_training_end: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)

        self.save_dir = Path(save_dir)
        self.latest_name = latest_name
        self.final_name = final_name
        self.save_final_on_training_end = save_final_on_training_end

        self.latest_path = self.save_dir / f"{self.latest_name}.zip"
        self.final_path = self.save_dir / f"{self.final_name}.zip"

        self.stopped_by_signal = False

    def _init_callback(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose > 0:
            print(f"[CONTROL] save_dir={self.save_dir}")
            print(f"[CONTROL] latest_path={self.latest_path}")
            print(f"[CONTROL] final_path={self.final_path}")

    def _save_latest(self) -> None:
        if self.model is None:
            return

        self.model.save(str(self.latest_path.with_suffix("")))

        if self.verbose > 0:
            print(f"[CONTROL] Saved latest: {self.latest_path}")

    def _save_final(self) -> None:
        if self.model is None:
            return

        self.model.save(str(self.final_path.with_suffix("")))

        if self.verbose > 0:
            print(f"[CONTROL] Saved final: {self.final_path}")

    def _on_step(self) -> bool:
        global STOP_REQUESTED

        if STOP_REQUESTED:
            self.stopped_by_signal = True
            self._save_latest()
            STOP_REQUESTED = False
            return False

        return True

    def _on_training_end(self) -> None:
        if self.save_final_on_training_end and not self.stopped_by_signal:
            self._save_final()