import math
import numpy as np
import pybullet as p


class RewardModulePlace:
    """
    Reward tốt hơn cho bài toán place:
    - khuyến khích giữ vật ổn định
    - đưa đúng tâm target
    - hạ thấp trước khi thả
    - thả nhẹ, gần bàn, gần target
    - success chỉ khi vật đã ổn định sau khi thả
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

        # internal states
        self._prev_obj_target_dist = None
        self._prev_xy_dist = None
        self._prev_height_above_table = None
        self._prev_holding = None
        self._success = False
        self._has_released = False

        # tuning
        self.xy_weight = 0.9
        self.z_weight = 0.35
        self.holding_bonus = 0.015
        self.near_target_hold_bonus = 0.04
        self.gentle_release_bonus = 0.5
        self.bad_release_penalty = 0.8
        self.velocity_penalty_weight = 0.08

        self.success_z_tol = 0.018
        self.success_vel_tol = 0.08
        self.good_release_vel = 0.10

        # độ cao mong muốn trước khi nhả: hơi trên mặt bàn
        self.release_height_target = self.table_z + 0.020

    def reset(self):
        self._prev_obj_target_dist = None
        self._prev_xy_dist = None
        self._prev_height_above_table = None
        self._prev_holding = None
        self._success = False
        self._has_released = False

    def set_target(self, target_pos):
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self._prev_obj_target_dist = None
        self._prev_xy_dist = None

    @staticmethod
    def _dist(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _xy_dist(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.sqrt(dx * dx + dy * dy)

    def compute(self, robot_id: int, obj_id: int, grip: float, holding: bool):
        info = {}

        obj_pos, _ = p.getBasePositionAndOrientation(
            obj_id, physicsClientId=self.physics_client_id
        )
        obj_pos = np.array(obj_pos, dtype=np.float32)

        lin_vel, ang_vel = p.getBaseVelocity(
            obj_id, physicsClientId=self.physics_client_id
        )
        lin_vel = np.array(lin_vel, dtype=np.float32)
        ang_vel = np.array(ang_vel, dtype=np.float32)

        obj_speed = float(np.linalg.norm(lin_vel))
        obj_ang_speed = float(np.linalg.norm(ang_vel))

        obj_target_dist = self._dist(obj_pos, self.target_pos)
        xy_dist = self._xy_dist(obj_pos, self.target_pos)
        height_above_table = max(0.0, float(obj_pos[2] - self.table_z))

        reward = 0.0

        # -------------------------------------------------
        # 1) progress shaping theo object -> target
        # -------------------------------------------------
        if self._prev_obj_target_dist is None:
            total_delta = 0.0
        else:
            total_delta = self._prev_obj_target_dist - obj_target_dist
        self._prev_obj_target_dist = obj_target_dist

        if self._prev_xy_dist is None:
            xy_delta = 0.0
        else:
            xy_delta = self._prev_xy_dist - xy_dist
        self._prev_xy_dist = xy_dist

        # chiều cao hướng tới release_height_target
        z_err = abs(height_above_table - (self.release_height_target - self.table_z))
        if self._prev_height_above_table is None:
            z_progress = 0.0
        else:
            prev_z_err = abs(
                self._prev_height_above_table - (self.release_height_target - self.table_z)
            )
            z_progress = prev_z_err - z_err
        self._prev_height_above_table = height_above_table

        total_delta = float(np.clip(total_delta, -self.delta_clip, self.delta_clip))
        xy_delta = float(np.clip(xy_delta, -self.delta_clip, self.delta_clip))
        z_progress = float(np.clip(z_progress, -self.delta_clip, self.delta_clip))

        reward += self.dist_weight * total_delta
        reward += self.xy_weight * xy_delta

        # -------------------------------------------------
        # 2) shaping riêng khi đang cầm vật
        # -------------------------------------------------
        if holding:
            reward += self.holding_bonus

            # ưu tiên đưa vật đúng tâm target trước
            reward += self.z_weight * z_progress

            if xy_dist < self.success_dist * 1.2:
                reward += self.near_target_hold_bonus

            # phạt nhẹ nếu rung/lắc mạnh khi đang cầm
            reward -= self.velocity_penalty_weight * min(obj_speed, 1.0) * 0.2

        else:
            # sau khi thả, phạt nhẹ nếu vật còn bay nhanh
            reward -= self.velocity_penalty_weight * min(obj_speed, 1.0)

        # time penalty luôn có
        reward -= self.time_penalty

        # -------------------------------------------------
        # 3) detect released event
        # -------------------------------------------------
        released = False
        if self._prev_holding is None:
            self._prev_holding = holding
        else:
            if self._prev_holding and (not holding):
                released = True
                self._has_released = True

                release_height_ok = True
                if self.z_release_max is not None:
                    release_height_ok = (obj_pos[2] <= self.z_release_max)

                near_xy = xy_dist <= self.success_dist
                low_enough = obj_pos[2] <= (self.table_z + 0.03)
                gentle_enough = obj_speed <= self.good_release_vel

                # thưởng release đẹp
                if near_xy and low_enough and gentle_enough and release_height_ok:
                    reward += self.release_bonus + self.gentle_release_bonus
                # release chấp nhận được
                elif near_xy and release_height_ok:
                    reward += self.release_bonus
                else:
                    reward -= self.bad_release_penalty

                # phạt thả quá cao
                if self.z_release_max is not None and obj_pos[2] > self.z_release_max:
                    reward -= self.high_release_penalty

            self._prev_holding = holding

        # -------------------------------------------------
        # 4) success condition chặt hơn
        # -------------------------------------------------
        terminated = False
        success = False

        release_height_ok = True
        if self.z_release_max is not None:
            release_height_ok = (obj_pos[2] <= self.z_release_max)

        on_table = abs(float(obj_pos[2]) - self.table_z) <= self.success_z_tol
        near_target = xy_dist <= self.success_dist
        settled = obj_speed <= self.success_vel_tol and obj_ang_speed <= 0.25

        if (
            (not self._success)
            and self._has_released
            and (not holding)
            and release_height_ok
            and near_target
            and on_table
            and settled
        ):
            self._success = True
            reward += self.success_bonus
            terminated = True
            success = True

        # -------------------------------------------------
        # 5) info
        # -------------------------------------------------
        info["success"] = bool(success)
        info["holding"] = bool(holding)
        info["released"] = bool(released)
        info["drop_event"] = bool(released)
        info["obj_target_dist"] = float(obj_target_dist)
        info["obj_xy_dist"] = float(xy_dist)
        info["obj_z"] = float(obj_pos[2])
        info["obj_speed"] = float(obj_speed)
        info["obj_ang_speed"] = float(obj_ang_speed)
        info["has_released"] = bool(self._has_released)
        info["on_table"] = bool(on_table)
        info["settled"] = bool(settled)

        return float(reward), bool(terminated), info