# script/stage1_reach/tensosbroad_callbacks.py
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class ReachTensorboardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.success_buffer = []
        self.dist_buffer = []

    def _on_step(self):

        infos = self.locals["infos"]

        for info in infos:

            if "success" in info:
                self.success_buffer.append(info["success"])

            if "ee_obj_dist" in info:
                self.dist_buffer.append(info["ee_obj_dist"])

        if len(self.success_buffer) > 100:
            self.success_buffer.pop(0)

        if len(self.dist_buffer) > 100:
            self.dist_buffer.pop(0)

        if len(self.success_buffer) > 0:
            success_rate = np.mean(self.success_buffer)
            self.logger.record("reach/success_rate", success_rate)

        if len(self.dist_buffer) > 0:
            dist_mean = np.mean(self.dist_buffer)
            self.logger.record("reach/distance_mean", dist_mean)

        return True