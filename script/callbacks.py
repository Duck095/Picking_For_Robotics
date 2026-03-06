from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class ReachTensorboardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.success_buffer = []
        self.dist_buffer = []

    def _on_step(self):

        infos = self.locals.get("infos", [])

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
    
class PlaceTensorboardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.success_buffer = []
        self.holding_buffer = []
        self.release_buffer = []
        self.obj_target_dist_buffer = []
        self.obj_z_buffer = []

    def _on_step(self):

        infos = self.locals.get("infos", [])

        for info in infos:

            if "success" in info:
                self.success_buffer.append(info["success"])

            if "holding" in info:
                self.holding_buffer.append(info["holding"])

            # event thả vật (tùy env đặt key)
            if "released" in info:
                self.release_buffer.append(info["released"])
            elif "drop_event" in info:
                self.release_buffer.append(info["drop_event"])

            if "obj_target_dist" in info:
                self.obj_target_dist_buffer.append(info["obj_target_dist"])

            if "obj_z" in info:
                self.obj_z_buffer.append(info["obj_z"])

        if len(self.success_buffer) > 100:
            self.success_buffer.pop(0)

        if len(self.holding_buffer) > 100:
            self.holding_buffer.pop(0)

        if len(self.release_buffer) > 100:
            self.release_buffer.pop(0)

        if len(self.obj_target_dist_buffer) > 100:
            self.obj_target_dist_buffer.pop(0)

        if len(self.obj_z_buffer) > 100:
            self.obj_z_buffer.pop(0)

        if len(self.success_buffer) > 0:
            success_rate = np.mean(self.success_buffer)
            self.logger.record("place/success_rate", success_rate)

        if len(self.holding_buffer) > 0:
            holding_rate = np.mean(self.holding_buffer)
            self.logger.record("place/holding_rate", holding_rate)

        if len(self.release_buffer) > 0:
            release_rate = np.mean(self.release_buffer)
            self.logger.record("place/release_rate", release_rate)

        if len(self.obj_target_dist_buffer) > 0:
            dist_mean = np.mean(self.obj_target_dist_buffer)
            self.logger.record("place/obj_target_dist_mean", dist_mean)

        if len(self.obj_z_buffer) > 0:
            z_mean = np.mean(self.obj_z_buffer)
            self.logger.record("place/obj_z_mean", z_mean)

        return True