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
        delta_clip: float = 0.02,   # ✅ clip progress để tránh farm reward
        physics_client_id=None,
    ):
        self.ee_link = ee_link
        self.lift_height = lift_height
        self.time_penalty = time_penalty
        self.dist_weight = dist_weight
        self.grasp_reward = grasp_reward
        self.success_bonus = success_bonus
        self.delta_clip = delta_clip
        self.physics_client_id = physics_client_id

        self._prev_dist = None
        self._success = False
        self._gave_grasp = False

    def reset(self):
        self._prev_dist = None
        self._success = False
        self._gave_grasp = False

    @staticmethod
    def _dist(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _ee_pos(self, robot_id: int):
        ls = p.getLinkState(robot_id, self.ee_link, physicsClientId=self.physics_client_id)
        return ls[4]

    def compute(self, robot_id: int, obj_id: int, grip: float, holding: bool):
        info = {}

        ee = self._ee_pos(robot_id)
        obj_pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.physics_client_id)
        d = self._dist(ee, obj_pos)

        # ✅ progress shaping (clipped)
        if self._prev_dist is None:
            delta = 0.0
        else:
            delta = self._prev_dist - d
        self._prev_dist = d

        if delta > self.delta_clip:
            delta = self.delta_clip
        elif delta < -self.delta_clip:
            delta = -self.delta_clip

        reward = self.dist_weight * delta
        reward -= self.time_penalty

        # ✅ thưởng khi lần đầu cầm thật (constraint attach)
        if holding and not self._gave_grasp:
            reward += self.grasp_reward
            self._gave_grasp = True

        terminated = False
        success = False

        # ✅ success khi đang cầm thật + nhấc lên
        if (not self._success) and holding and (obj_pos[2] >= self.lift_height):
            self._success = True
            reward += self.success_bonus
            terminated = True
            success = True

        info["success"] = bool(success)
        info["holding"] = bool(holding)
        info["ee_obj_dist"] = float(d)
        info["obj_z"] = float(obj_pos[2])
        return float(reward), bool(terminated), info