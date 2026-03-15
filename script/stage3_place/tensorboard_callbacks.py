from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class PlaceTensorboardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.success_buffer = []
        self.ee_obj_dist_buffer = []
        self.obj_target_dist_buffer = []
        self.holding_buffer = []

    def _on_step(self):
        infos = self.locals["infos"]

        for info in infos:
            if "success" in info:
                self.success_buffer.append(float(info["success"]))

            if "ee_obj_dist" in info:
                self.ee_obj_dist_buffer.append(float(info["ee_obj_dist"]))

            if "obj_target_dist" in info:
                self.obj_target_dist_buffer.append(float(info["obj_target_dist"]))

            if "holding" in info:
                self.holding_buffer.append(float(info["holding"]))

        if len(self.success_buffer) > 100:
            self.success_buffer.pop(0)

        if len(self.ee_obj_dist_buffer) > 100:
            self.ee_obj_dist_buffer.pop(0)

        if len(self.obj_target_dist_buffer) > 100:
            self.obj_target_dist_buffer.pop(0)

        if len(self.holding_buffer) > 100:
            self.holding_buffer.pop(0)

        if len(self.success_buffer) > 0:
            self.logger.record("place/success_rate", np.mean(self.success_buffer))

        if len(self.ee_obj_dist_buffer) > 0:
            self.logger.record("place/ee_obj_distance_mean", np.mean(self.ee_obj_dist_buffer))

        if len(self.obj_target_dist_buffer) > 0:
            self.logger.record("place/obj_target_distance_mean", np.mean(self.obj_target_dist_buffer))

        if len(self.holding_buffer) > 0:
            self.logger.record("place/holding_rate", np.mean(self.holding_buffer))

        return True