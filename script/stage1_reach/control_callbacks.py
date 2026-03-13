# script/stage1_reach/control_callbacks.py
import signal
from stable_baselines3.common.callbacks import BaseCallback


STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\n[SIGNAL] Received signal {signum}. Will save latest and stop safely...")


signal.signal(signal.SIGINT, request_stop)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, request_stop)


class StopTrainingGracefully(Exception):
    pass


class SaveLatestOnStopCallback(BaseCallback):
    """
    Chỉ lưu latest khi có tín hiệu dừng (Ctrl+C / SIGTERM).
    Không lưu định kỳ.
    """

    def __init__(self, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        global STOP_REQUESTED

        if STOP_REQUESTED:
            self.model.save(self.save_path)
            if self.verbose > 0:
                print(
                    f"[LATEST] Saved on stop: {self.save_path}.zip | "
                    f"timesteps={self.num_timesteps}"
                )
            raise StopTrainingGracefully()

        return True