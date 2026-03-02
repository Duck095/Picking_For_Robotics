import math
import pybullet as p


class RewardModuleStage1:
    def __init__(
        self,
        ee_link: int,
        lift_height: float = 0.045,
        time_penalty: float = 0.01,
        dist_weight: float = 0.2,
        grasp_reward: float = 2.0,
        success_bonus: float = 2.0,
        success_dist: float = 0.12,  # EE gần vật
    ):
        self.ee_link = ee_link
        self.lift_height = lift_height
        self.time_penalty = time_penalty
        self.dist_weight = dist_weight
        self.grasp_reward = grasp_reward
        self.success_bonus = success_bonus
        self.success_dist = success_dist

        self._prev_dist = None
        self._success = False

    def reset(self):
        self._prev_dist = None
        self._success = False

    @staticmethod
    def _dist(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _ee_pos(self, robot_id: int):
        ls = p.getLinkState(robot_id, self.ee_link)
        return ls[4]

    def is_grasp_success(self, robot_id: int, obj_id: int, grip: float) -> bool:
        if float(grip) < 0.5:
            return False

        obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
        if obj_pos[2] < self.lift_height:
            return False

        ee = self._ee_pos(robot_id)
        d = self._dist(ee, obj_pos)
        return d < self.success_dist

    def compute(self, robot_id: int, obj_id: int, grip: float):
        info = {}

        ee = self._ee_pos(robot_id)
        obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
        d = self._dist(ee, obj_pos)

        reward = 0.0
        if self._prev_dist is None:
            self._prev_dist = d
        else:
            reward += self.dist_weight * (self._prev_dist - d)
            self._prev_dist = d

        reward -= self.time_penalty

        terminated = False
        if not self._success and self.is_grasp_success(robot_id, obj_id, grip):
            self._success = True
            reward += (self.grasp_reward + self.success_bonus)
            terminated = True
            info["success"] = True
        else:
            info["success"] = False

        info["ee_obj_dist"] = float(d)
        info["obj_z"] = float(obj_pos[2])
        return reward, terminated, info