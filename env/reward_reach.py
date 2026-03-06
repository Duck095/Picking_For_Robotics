# env/reward_reach.py
import math
import pybullet as p


class ReachReward:
    def __init__(
        self,
        ee_link: int,
        success_dist: float,
        dist_weight: float,
        time_penalty: float,
        success_bonus: float,
        physics_client_id=None,
        delta_clip: float = 0.02,
    ):
        self.ee_link = int(ee_link)
        self.success_dist = float(success_dist)
        self.dist_weight = float(dist_weight)
        self.time_penalty = float(time_penalty)
        self.success_bonus = float(success_bonus)
        self.cid = physics_client_id
        self.delta_clip = float(delta_clip)

        self.prev_dist = None
        self.done_once = False

    def reset(self):
        self.prev_dist = None
        self.done_once = False

    def _ee_pos(self, robot_id: int):
        return p.getLinkState(robot_id, self.ee_link, physicsClientId=self.cid)[4]

    @staticmethod
    def _dist(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def compute(self, robot_id: int, obj_id: int):
        ee = self._ee_pos(robot_id)
        obj_pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.cid)
        d = self._dist(ee, obj_pos)

        r = 0.0

        r += -0.02 * d

        if self.prev_dist is None:
            self.prev_dist = d
        else:
            delta = self.prev_dist - d
            if delta > self.delta_clip:
                delta = self.delta_clip
            if delta < -self.delta_clip:
                delta = -self.delta_clip
            r += self.dist_weight * delta
            self.prev_dist = d

        r -= self.time_penalty

        terminated = False
        success = False
        if (not self.done_once) and (d < self.success_dist):
            self.done_once = True
            terminated = True
            success = True
            r += self.success_bonus

        info = {
            "success": success,
            "ee_obj_dist": float(d),
            "success_dist": float(self.success_dist),
            "obj_pos": [float(x) for x in obj_pos],
        }
        return float(r), bool(terminated), info