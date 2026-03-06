import math
import numpy as np
import pybullet as p


class RewardModulePlace:
    """
    Reward cho place/drop.
    Info keys:
      success, holding, released, drop_event, obj_target_dist, obj_z
    """

    def __init__(
        self,
        ee_link: int,
        target_pos,
        time_penalty=0.01,
        dist_weight=0.6,
        release_bonus=0.2,
        success_bonus=3.0,
        success_dist=0.12,
        table_z=0.06,
        delta_clip=0.03,
        z_release_max=None,
        high_release_penalty=1.0,
        physics_client_id=None,
    ):
        self.ee_link = int(ee_link)
        self.target_pos = np.array(target_pos, dtype=np.float32)

        self.time_penalty = float(time_penalty)
        self.dist_weight = float(dist_weight)
        self.release_bonus = float(release_bonus)
        self.success_bonus = float(success_bonus)

        self.success_dist = float(success_dist)
        self.table_z = float(table_z)
        self.delta_clip = float(delta_clip)

        self.z_release_max = None if z_release_max is None else float(z_release_max)
        self.high_release_penalty = float(high_release_penalty)

        self.physics_client_id = physics_client_id

        self._prev_obj_target_dist = None
        self._prev_holding = None
        self._success = False

    def reset(self):
        self._prev_obj_target_dist = None
        self._prev_holding = None
        self._success = False

    def set_target(self, target_pos):
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self._prev_obj_target_dist = None

    @staticmethod
    def _dist(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def compute(self, robot_id: int, obj_id: int, grip: float, holding: bool):
        info = {}

        obj_pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.physics_client_id)
        obj_pos = np.array(obj_pos, dtype=np.float32)

        obj_target_dist = self._dist(obj_pos, self.target_pos)

        # progress shaping: object -> target
        if self._prev_obj_target_dist is None:
            delta = 0.0
        else:
            delta = self._prev_obj_target_dist - obj_target_dist
        self._prev_obj_target_dist = obj_target_dist

        delta = float(np.clip(delta, -self.delta_clip, self.delta_clip))

        reward = self.dist_weight * delta
        reward -= self.time_penalty

        # released event: holding True -> False
        released = False
        if self._prev_holding is None:
            self._prev_holding = holding
        else:
            if self._prev_holding and (not holding):
                released = True

                # bonus chỉ khi thả gần target
                if obj_target_dist <= (self.success_dist * 1.5):
                    reward += self.release_bonus

                # 3C: phạt thả quá cao
                if self.z_release_max is not None and obj_pos[2] > self.z_release_max:
                    reward -= self.high_release_penalty

            self._prev_holding = holding

        # success condition
        terminated = False
        success = False

        release_height_ok = True
        if self.z_release_max is not None:
            release_height_ok = (obj_pos[2] <= self.z_release_max)

        if (not self._success) and released and (not holding) and release_height_ok:
            if (obj_target_dist <= self.success_dist) and (obj_pos[2] <= self.table_z):
                self._success = True
                reward += self.success_bonus
                terminated = True
                success = True

        info["success"] = bool(success)
        info["holding"] = bool(holding)
        info["released"] = bool(released)
        info["drop_event"] = bool(released)
        info["obj_target_dist"] = float(obj_target_dist)
        info["obj_z"] = float(obj_pos[2])

        return float(reward), bool(terminated), info