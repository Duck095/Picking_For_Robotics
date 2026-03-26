from __future__ import annotations

import multiprocessing
import signal
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


# ============================================================
# GLOBAL STOP FLAG
# ------------------------------------------------------------
# Khi người dùng bấm Ctrl+C, signal handler sẽ set cờ này.
# Callback sẽ kiểm tra cờ ở _on_step().
#
# Lưu ý:
# - Chỉ main process mới nên đăng ký signal handler
# - Với SubprocVecEnv trên Windows, callback có thể không phải lúc nào
#   cũng kịp save latest trước khi pipe bị ngắt
# - Vì vậy file train vẫn cần except KeyboardInterrupt để save fallback
# ============================================================
STOP_REQUESTED = False


def request_stop(signum, frame):
    """
    Signal handler cho Ctrl+C / SIGTERM.

    Không dừng ngay tại đây.
    Chỉ set cờ STOP_REQUESTED để callback xử lý ở lần _on_step() tiếp theo.
    """
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\n[CONTROL] Received signal {signum}. Will save latest checkpoint and stop safely.")


# ------------------------------------------------------------
# Chỉ main process mới được đăng ký signal handler
# để tránh subprocess của SubprocVecEnv cũng nhận signal và in log lặp.
# ------------------------------------------------------------
if multiprocessing.current_process().name == "MainProcess":
    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)


class SaveLatestOnStopCallback(BaseCallback):
    """
    Callback điều khiển lưu model cho Stage 1.

    Vai trò:
    - Nếu nhận tín hiệu stop và callback còn chạy tới _on_step():
        -> save latest
        -> return False để dừng train sạch
    - Nếu train kết thúc bình thường:
        -> save final

    Không làm:
    - Không save latest định kỳ
      (checkpoint định kỳ do CheckpointCallback đảm nhiệm)

    Tham số
    --------
    save_dir:
        thư mục chứa model, ví dụ "models"
    latest_name:
        tên file latest không kèm .zip,
        ví dụ "stage1_reach_1A_latest"
    final_name:
        tên file final không kèm .zip,
        ví dụ "stage1_reach_1A"
    save_final_on_training_end:
        True -> train kết thúc bình thường sẽ save final
    verbose:
        mức log
    """

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

        # đánh dấu train có bị dừng bởi signal hay không
        self.stopped_by_signal = False

    def _init_callback(self) -> None:
        """
        Được gọi khi callback được khởi tạo trong learn().
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose > 0:
            print(f"[CONTROL] save_dir={self.save_dir}")
            print(f"[CONTROL] latest_path={self.latest_path}")
            print(f"[CONTROL] final_path={self.final_path}")

    def _save_latest(self) -> None:
        """
        Save latest model.
        """
        if self.model is None:
            return

        # model.save("models/stage1_reach_1A_latest")
        # sẽ tạo ra file .zip
        self.model.save(str(self.latest_path.with_suffix("")))

        if self.verbose > 0:
            print(f"[CONTROL] Saved latest: {self.latest_path}")

    def _save_final(self) -> None:
        """
        Save final model.
        """
        if self.model is None:
            return

        self.model.save(str(self.final_path.with_suffix("")))

        if self.verbose > 0:
            print(f"[CONTROL] Saved final: {self.final_path}")

    def _on_step(self) -> bool:
        """
        Gọi sau mỗi step rollout.

        Nếu STOP_REQUESTED=True:
        - save latest
        - reset cờ
        - return False để SB3 dừng train
        """
        global STOP_REQUESTED

        if STOP_REQUESTED:
            self.stopped_by_signal = True
            self._save_latest()
            STOP_REQUESTED = False
            return False

        return True

    def _on_training_end(self) -> None:
        """
        Được gọi khi train kết thúc.

        Nếu train kết thúc bình thường:
        - save final

        Nếu train bị stop bởi signal:
        - không save final
        """
        if self.save_final_on_training_end and not self.stopped_by_signal:
            self._save_final()