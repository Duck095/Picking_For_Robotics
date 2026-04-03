import math
import numpy as np
import pybullet as p


class RewardModulePlace:
    def __init__(
        self,
        ee_link: int,
        target_pos,
        substage="3A",
        time_penalty=0.01,
        dist_weight=1.0,
        xy_weight=2.0,
        z_weight=0.4,
        holding_bonus=0.004,
        near_target_hold_bonus=0.0,
        release_bonus=0.2,
        gentle_release_bonus=0.1,
        success_bonus=8.0,
        success_dist=0.09,
        table_z=0.02,
        delta_clip=0.03,
        z_release_max=None,
        high_release_penalty=1.0,
        velocity_penalty_weight=0.08,
        failed_place_penalty=2.0,
        early_release_penalty=3.0,
        far_release_penalty=1.0,
        release_near_factor=1.0,
        release_height_tol=0.03,
        success_z_tol=0.025,
        success_vel_tol=0.10,
        success_ang_vel_tol=0.30,
        good_release_vel=0.10,
        ready_release_grace_steps=12,
        overhold_penalty=0.02,
        terminate_on_bad_release=True,
        physics_client_id=None,
    ):
        self.ee_link = int(ee_link)
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self.substage = str(substage).upper()

        self.time_penalty = float(time_penalty)
        self.dist_weight = float(dist_weight)
        self.xy_weight = float(xy_weight)
        self.z_weight = float(z_weight)
        self.holding_bonus = float(holding_bonus)
        self.near_target_hold_bonus = float(near_target_hold_bonus)
        self.release_bonus = float(release_bonus)
        self.gentle_release_bonus = float(gentle_release_bonus)
        self.success_bonus = float(success_bonus)
        self.success_dist = float(success_dist)
        self.table_z = float(table_z)
        self.delta_clip = float(delta_clip)
        self.z_release_max = None if z_release_max is None else float(z_release_max)
        self.high_release_penalty = float(high_release_penalty)
        self.velocity_penalty_weight = float(velocity_penalty_weight)
        self.failed_place_penalty = float(failed_place_penalty)
        self.early_release_penalty = float(early_release_penalty)
        self.far_release_penalty = float(far_release_penalty)
        self.release_near_factor = float(release_near_factor)
        self.release_height_tol = float(release_height_tol)
        self.success_z_tol = float(success_z_tol)
        self.success_vel_tol = float(success_vel_tol)
        self.success_ang_vel_tol = float(success_ang_vel_tol)
        self.good_release_vel = float(good_release_vel)
        self.ready_release_grace_steps = int(ready_release_grace_steps)
        self.overhold_penalty = float(overhold_penalty)
        self.terminate_on_bad_release = bool(terminate_on_bad_release)
        self.physics_client_id = physics_client_id

        self._apply_substage_profile()
        self.release_height_target = self.table_z + min(self.release_height_tol, 0.02)
        self.reset()

    def _apply_substage_profile(self):
        if self.substage == "3A":
            self.release_bonus = max(self.release_bonus, 0.25)
            self.success_bonus = max(self.success_bonus, 8.0)
            self.ready_release_grace_steps = max(self.ready_release_grace_steps, 14)
            self.overhold_penalty = max(self.overhold_penalty, 0.015)
            return

        if self.substage == "3B":
            self.success_bonus = max(self.success_bonus, 10.0)
            self.ready_release_grace_steps = min(self.ready_release_grace_steps, 10)
            self.overhold_penalty = max(self.overhold_penalty, 0.02)
            self.release_height_tol = min(self.release_height_tol, 0.025)
            self.success_z_tol = min(self.success_z_tol, 0.022)
            return

        if self.substage == "3C":
            self.success_bonus = max(self.success_bonus, 12.0)
            self.high_release_penalty = max(self.high_release_penalty, 2.0)
            self.early_release_penalty = max(self.early_release_penalty, 4.0)
            self.far_release_penalty = max(self.far_release_penalty, 2.0)
            self.release_height_tol = min(self.release_height_tol, 0.02)
            self.success_vel_tol = min(self.success_vel_tol, 0.08)
            self.success_ang_vel_tol = min(self.success_ang_vel_tol, 0.20)
            self.good_release_vel = min(self.good_release_vel, 0.08)
            self.ready_release_grace_steps = min(self.ready_release_grace_steps, 8)
            self.overhold_penalty = max(self.overhold_penalty, 0.03)

    def reset(self):
        self._prev_obj_target_dist = None
        self._prev_xy_dist = None
        self._prev_height_above_table = None
        self._prev_holding = None
        self._success = False
        self._has_released = False
        self._ready_release_steps = 0

    def set_target(self, target_pos):
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self._prev_obj_target_dist = None
        self._prev_xy_dist = None
        self._prev_height_above_table = None
        self._ready_release_steps = 0

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
        del robot_id, grip

        info = {}

        obj_pos, _ = p.getBasePositionAndOrientation(
            obj_id, physicsClientId=self.physics_client_id
        )
        obj_pos = np.array(obj_pos, dtype=np.float32)

        lin_vel, ang_vel = p.getBaseVelocity(obj_id, physicsClientId=self.physics_client_id)
        lin_vel = np.array(lin_vel, dtype=np.float32)
        ang_vel = np.array(ang_vel, dtype=np.float32)

        obj_speed = float(np.linalg.norm(lin_vel))
        obj_ang_speed = float(np.linalg.norm(ang_vel))

        obj_target_dist = self._dist(obj_pos, self.target_pos)
        xy_dist = self._xy_dist(obj_pos, self.target_pos)
        height_above_table = max(0.0, float(obj_pos[2] - self.table_z))

        reward = -self.time_penalty
        terminated = False
        bad_release = False

        total_delta = (
            0.0
            if self._prev_obj_target_dist is None
            else self._prev_obj_target_dist - obj_target_dist
        )
        self._prev_obj_target_dist = obj_target_dist

        xy_delta = 0.0 if self._prev_xy_dist is None else self._prev_xy_dist - xy_dist
        self._prev_xy_dist = xy_dist

        z_err = abs(height_above_table - (self.release_height_target - self.table_z))
        if self._prev_height_above_table is None:
            z_progress = 0.0
        else:
            prev_z_err = abs(
                self._prev_height_above_table - (self.release_height_target - self.table_z)
            )
            z_progress = prev_z_err - z_err
        self._prev_height_above_table = height_above_table

        total_delta = float(max(-self.delta_clip, min(total_delta, self.delta_clip)))
        xy_delta = float(max(-self.delta_clip, min(xy_delta, self.delta_clip)))
        z_progress = float(max(-self.delta_clip, min(z_progress, self.delta_clip)))

        release_xy_limit = self.success_dist * self.release_near_factor
        release_height_ok = height_above_table <= self.release_height_tol
        release_xy_ok = xy_dist <= release_xy_limit
        ready_to_release = release_xy_ok and release_height_ok

        phase = "carry"
        if holding:
            reward += self.holding_bonus
            reward += self.dist_weight * total_delta
            reward += self.xy_weight * xy_delta

            if release_xy_ok:
                reward += self.z_weight * z_progress
                reward += self.near_target_hold_bonus

            if ready_to_release:
                phase = "release_ready"
                self._ready_release_steps += 1
                if self._ready_release_steps > self.ready_release_grace_steps:
                    extra_steps = self._ready_release_steps - self.ready_release_grace_steps
                    reward -= self.overhold_penalty * extra_steps
            else:
                self._ready_release_steps = 0
        else:
            phase = "post_release"
            self._ready_release_steps = 0
            reward -= self.velocity_penalty_weight * min(obj_speed, 1.0)
            if self._has_released:
                reward -= 0.02

        released = False
        prev_holding = True if self._prev_holding is None else self._prev_holding
        if prev_holding and (not holding):
            released = True
            phase = "release_event"

            if not self._has_released:
                if ready_to_release:
                    reward += self.release_bonus
                    if obj_speed <= self.good_release_vel:
                        reward += self.gentle_release_bonus
                else:
                    bad_release = True
                    reward -= self.early_release_penalty
                    if xy_dist > self.success_dist * 2.0:
                        reward -= self.far_release_penalty

                if self.z_release_max is not None and obj_pos[2] > self.z_release_max:
                    reward -= self.high_release_penalty
                    bad_release = True

            self._has_released = True

        self._prev_holding = holding

        on_table = abs(float(obj_pos[2]) - self.table_z) <= self.success_z_tol
        near_target = xy_dist <= self.success_dist
        settled = (
            obj_speed <= self.success_vel_tol
            and obj_ang_speed <= self.success_ang_vel_tol
        )

        success = False
        if bad_release and self.terminate_on_bad_release:
            terminated = True

        if (not self._success) and self._has_released and (not holding) and settled:
            terminated = True
            phase = "settled"
            if near_target and on_table:
                self._success = True
                reward += self.success_bonus
                success = True
            else:
                reward -= self.failed_place_penalty

        if (not holding) and obj_pos[2] < (self.table_z - 0.03):
            reward -= self.failed_place_penalty
            terminated = True

        info["success"] = bool(success)
        info["holding"] = bool(holding)
        info["released"] = bool(released)
        info["drop_event"] = bool(released)
        info["bad_release"] = bool(bad_release)
        info["phase"] = phase
        info["obj_target_dist"] = float(obj_target_dist)
        info["obj_xy_dist"] = float(xy_dist)
        info["obj_z"] = float(obj_pos[2])
        info["obj_speed"] = float(obj_speed)
        info["obj_ang_speed"] = float(obj_ang_speed)
        info["has_released"] = bool(self._has_released)
        info["on_table"] = bool(on_table)
        info["settled"] = bool(settled)
        info["success_dist"] = float(self.success_dist)
        info["release_xy_limit"] = float(release_xy_limit)
        info["release_height_ok"] = bool(release_height_ok)
        info["ready_release_steps"] = int(self._ready_release_steps)
        info["obj_pos"] = obj_pos.tolist()

        return float(reward), bool(terminated), info
