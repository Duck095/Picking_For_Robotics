import math
import numpy as np
import pybullet as p


class RewardModulePlace:
    """
    Reward cho place/drop.
    GIỮ KEY INFO CŨ:
      success, holding, released, drop_event, obj_target_dist, obj_z
    """

    def __init__(
        self,
        ee_link: int,
        target_pos,
        time_penalty=0.01,
        dist_weight=0.3,
        release_bonus=1.0,
        success_bonus=3.0,
        success_dist=0.12,
        table_z=0.06,
        delta_clip=0.02,
        z_release_max=None,          # 3C: giới hạn độ cao khi thả
        high_release_penalty=1.0,
        physics_client_id=None,
    ):
        self.ee_link = ee_link
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

        self._prev_obj_dist = None
        self._prev_holding = None
        self._success = False
        self._gave_grasp = False  # ✅ Thêm biến này để theo dõi đã cầm chưa

    def reset(self):
        self._prev_obj_dist = None
        self._prev_holding = None
        self._success = False
        self._gave_grasp = False  # ✅ Đặt lại mỗi lần reset

    def set_target(self, target_pos):
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self._prev_obj_dist = None

    @staticmethod
    def _dist(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _ee_pos(self, robot_id: int):
        """
        Hàm này lấy vị trí của end-effector (EE) từ robot ID và ee_link.
        """
        ls = p.getLinkState(robot_id, self.ee_link, physicsClientId=self.physics_client_id)
        return ls[4]  # Trả về vị trí của EE (thường là phần thứ 4 của ls)

    def compute(self, robot_id: int, obj_id: int, grip: float, holding: bool):
        info = {}

        ee = self._ee_pos(robot_id)  # Vị trí của end-effector
        obj_pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.physics_client_id)
        d = self._dist(ee, obj_pos)

        # ✅ progress shaping (clipped)
        if self._prev_obj_dist is None:
            delta = 0.0
        else:
            delta = self._prev_obj_dist - d
        self._prev_obj_dist = d

        if delta > self.delta_clip:
            delta = self.delta_clip
        elif delta < -self.delta_clip:
            delta = -self.delta_clip

        reward = self.dist_weight * delta
        reward -= self.time_penalty

        # ✅ thưởng khi lần đầu cầm thật (constraint attach)
        if holding and not self._gave_grasp:
            reward += self.release_bonus  # hoặc self.grasp_reward
            self._gave_grasp = True

        terminated = False
        success = False

        # ✅ success khi đang cầm thật + nhấc lên
        if (not self._success) and holding and (obj_pos[2] >= self.table_z):
            self._success = True
            reward += self.success_bonus
            terminated = True
            success = True

        # ✅ Chỉ thưởng release khi gần target (nhả đúng lúc)
        if holding and "released" in info:
            if d <= self.success_dist * 1.5:  # Thả gần target (dễ học hơn)
                reward += self.release_bonus

        info["success"] = bool(success)
        info["holding"] = bool(holding)
        info["ee_obj_dist"] = float(d)
        info["obj_z"] = float(obj_pos[2])
        return float(reward), bool(terminated), info