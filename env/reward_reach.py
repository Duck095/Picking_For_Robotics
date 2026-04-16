from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from config.reach_env_config import ReachEnvConfig


class RewardReach:
    def __init__(self, cfg: ReachEnvConfig):
        self.cfg = cfg
        self._aligned_hang_steps = 0

    @staticmethod
    def _norm(vec: np.ndarray) -> float:
        return float(np.linalg.norm(vec))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def reset(self) -> None:
        self._aligned_hang_steps = 0

    def _is_basic_reach_stage(self) -> bool:
        return self.cfg.substage in ("1A", "1B")

    def _is_precision_stage(self) -> bool:
        return self.cfg.substage in ("1C", "1D")

    def _is_yaw_refine_stage(self) -> bool:
        return self.cfg.substage == "1E"

    def _is_orientation_full_stage(self) -> bool:
        return self.cfg.substage == "1F"

    def _phase_from_metrics_basic(self, curr_xy_dist: float, curr_z_dist: float, curr_dist: float) -> str:
        tcfg = self.cfg.target
        acfg = self.cfg.action

        if curr_dist <= tcfg.success_dist * 1.25:
            return "settle"
        if curr_xy_dist <= tcfg.xy_align_threshold and curr_z_dist > acfg.descend_z_gate:
            return "descend"
        if curr_xy_dist <= tcfg.xy_align_threshold:
            return "near_aligned"
        if curr_dist <= acfg.near_dist:
            return "align"
        return "far"

    def _phase_from_metrics_precision(self, curr_xy_dist: float, curr_z_dist: float, curr_dist: float) -> str:
        tcfg = self.cfg.target
        acfg = self.cfg.action

        if curr_xy_dist <= tcfg.xy_success_dist * 1.5 and curr_z_dist <= tcfg.z_success_dist * 1.5:
            return "stable_hold"
        if curr_xy_dist <= tcfg.xy_align_threshold:
            return "z_settle"
        if curr_dist <= acfg.near_dist:
            return "xy_lock"
        return "approach"

    def _phase_from_metrics_orientation(
        self,
        curr_xy_dist: float,
        curr_z_dist: float,
        curr_dist: float,
        curr_yaw_error: float,
    ) -> str:
        tcfg = self.cfg.target
        acfg = self.cfg.action

        if (
            curr_xy_dist <= tcfg.xy_success_dist * 1.5
            and curr_z_dist <= tcfg.z_success_dist * 1.5
            and curr_yaw_error <= tcfg.yaw_success_dist * 1.5
        ):
            return "stable_hold"

        if curr_xy_dist <= tcfg.xy_align_threshold and curr_z_dist <= tcfg.z_success_dist * 1.8:
            return "yaw_align"

        if curr_xy_dist <= tcfg.xy_align_threshold:
            return "z_settle"

        if curr_dist <= acfg.near_dist:
            return "xy_lock"

        return "approach"

    def compute(
        self,
        prev_ee_pos: np.ndarray,
        curr_ee_pos: np.ndarray,
        target_pos: np.ndarray,
        action: np.ndarray,
        success: bool,
        truncated: bool,
        xy_aligned: bool,
        workspace_violated: bool,
        prev_action: np.ndarray | None = None,
        prev_ee_yaw: float = 0.0,
        curr_ee_yaw: float = 0.0,
        target_yaw: float = 0.0,
        phase: str | None = None,
        xy_locked: bool = False,
        z_locked: bool = False,
        curriculum_level: str = "hard",
    ) -> Tuple[float, Dict[str, float]]:
        rcfg = self.cfg.reward

        prev_delta = target_pos - prev_ee_pos
        curr_delta = target_pos - curr_ee_pos

        prev_dist = self._norm(prev_delta)
        curr_dist = self._norm(curr_delta)

        prev_xy_dist = self._norm(prev_delta[:2])
        curr_xy_dist = self._norm(curr_delta[:2])

        prev_z_dist = abs(float(prev_delta[2]))
        curr_z_dist = abs(float(curr_delta[2]))

        prev_yaw_error = abs(self._wrap_angle(target_yaw - prev_ee_yaw))
        curr_yaw_error = abs(self._wrap_angle(target_yaw - curr_ee_yaw))

        ee_motion = self._norm(curr_ee_pos - prev_ee_pos)
        action_mag = float(np.linalg.norm(action))
        xyz_action_mag = float(np.linalg.norm(action[:3]))
        yaw_action_mag = abs(float(action[3]))

        if self._is_yaw_refine_stage():
            phase = phase or self._phase_from_metrics_orientation(
                curr_xy_dist=curr_xy_dist,
                curr_z_dist=curr_z_dist,
                curr_dist=curr_dist,
                curr_yaw_error=curr_yaw_error,
            )

            reward = 0.0
            r_dist_progress = 0.20 * rcfg.w_dist_progress * (prev_dist - curr_dist)

            if phase == "approach":
                xy_weight = 1.10 * rcfg.w_xy_progress
                z_weight = 0.40 * rcfg.w_z_progress
                yaw_weight = 0.00

            elif phase == "xy_lock":
                xy_weight = 1.25 * rcfg.w_xy_progress
                z_weight = 0.30 * rcfg.w_z_progress
                yaw_weight = 0.00

            elif phase == "z_settle":
                xy_weight = 0.30 * rcfg.w_xy_progress
                z_weight = 1.25 * rcfg.w_z_progress
                yaw_weight = 0.00

            elif phase == "yaw_align":
                xy_weight = 0.10 * rcfg.w_xy_progress
                z_weight = 0.10 * rcfg.w_z_progress
                yaw_weight = 1.35 * rcfg.w_yaw_progress

            else:  # stable_hold
                xy_weight = 0.05 * rcfg.w_xy_progress
                z_weight = 0.05 * rcfg.w_z_progress
                yaw_weight = 0.25 * rcfg.w_yaw_progress

            r_xy_progress = xy_weight * (prev_xy_dist - curr_xy_dist)
            r_z_progress = z_weight * (prev_z_dist - curr_z_dist)
            r_yaw_progress = yaw_weight * (prev_yaw_error - curr_yaw_error)
            reward += r_dist_progress + r_xy_progress + r_z_progress + r_yaw_progress

            near_pose = (
                curr_xy_dist < self.cfg.target.xy_success_dist * 1.35
                and curr_z_dist < self.cfg.target.z_success_dist * 1.35
                and curr_yaw_error < self.cfg.target.yaw_success_dist * 1.6
            )

            r_pose_maintain = 0.0
            if phase in ("yaw_align", "stable_hold"):
                xy_not_worse = curr_xy_dist <= prev_xy_dist + 5e-4
                z_not_worse = curr_z_dist <= prev_z_dist + 5e-4
                yaw_better = curr_yaw_error < prev_yaw_error - 1e-4
                if xy_not_worse and z_not_worse and yaw_better:
                    r_pose_maintain = 0.20 * rcfg.w_stable_bonus
                    reward += r_pose_maintain

            r_success = rcfg.w_success if success else 0.0
            reward += r_success

            r_stable_bonus = 0.0
            if near_pose and action_mag < 0.06 and ee_motion < 0.0020:
                r_stable_bonus = rcfg.w_stable_bonus
                reward += r_stable_bonus

            p_idle = rcfg.w_idle_penalty if ee_motion < rcfg.idle_eps else 0.0
            reward -= p_idle

            dist_progress_abs = abs(prev_dist - curr_dist)
            xy_progress_abs = abs(prev_xy_dist - curr_xy_dist)
            z_progress_abs = abs(prev_z_dist - curr_z_dist)
            yaw_progress_abs = abs(prev_yaw_error - curr_yaw_error)

            tiny_motion = ee_motion < rcfg.stagnation_motion_eps
            tiny_progress = (
                dist_progress_abs < rcfg.stagnation_progress_eps
                and xy_progress_abs < rcfg.stagnation_progress_eps
                and z_progress_abs < rcfg.stagnation_progress_eps
                and yaw_progress_abs < (1.5 * rcfg.stagnation_progress_eps)
            )

            p_stagnation = 0.0
            if tiny_motion and tiny_progress and not success:
                p_stagnation = rcfg.w_stagnation_penalty
                if phase in ("approach", "xy_lock"):
                    p_stagnation += rcfg.w_approach_stagnation_boost
            reward -= p_stagnation

            p_action = rcfg.w_action_penalty * xyz_action_mag
            p_yaw_action = rcfg.w_yaw_action_penalty * yaw_action_mag
            reward -= (p_action + p_yaw_action)

            p_workspace = rcfg.w_workspace_violation if workspace_violated else 0.0
            reward -= p_workspace

            p_regress_near = 0.0
            if phase in ("yaw_align", "stable_hold"):
                if curr_xy_dist > prev_xy_dist + 1e-5:
                    p_regress_near += rcfg.w_regress_near
                if curr_z_dist > prev_z_dist + 1e-5:
                    p_regress_near += rcfg.w_regress_near
                if curr_yaw_error > prev_yaw_error + 1e-5:
                    p_regress_near += rcfg.w_yaw_regress_near
            reward -= p_regress_near

            p_phase_lock = 0.0
            if xy_locked and curr_xy_dist > max(self.cfg.target.xy_align_threshold * 1.10, prev_xy_dist + 8e-4):
                p_phase_lock += rcfg.w_phase_lock_xy_penalty
            if z_locked and curr_z_dist > max(self.cfg.target.z_success_dist * 1.30, prev_z_dist + 8e-4):
                p_phase_lock += rcfg.w_phase_lock_z_penalty
            reward -= p_phase_lock

            p_jitter = 0.0
            if near_pose or phase == "stable_hold":
                p_jitter += rcfg.w_jitter_near * action_mag
                if prev_action is not None:
                    prev_mag = float(np.linalg.norm(prev_action))
                    curr_mag = float(np.linalg.norm(action))
                    if prev_mag > 1e-6 and curr_mag > 1e-6:
                        cosine = float(np.dot(prev_action, action) / (prev_mag * curr_mag + 1e-8))
                        if cosine < 0.0:
                            p_jitter += rcfg.w_reverse_action_near * abs(cosine)
            reward -= p_jitter

            p_timeout = rcfg.w_timeout_penalty if truncated and (not success) else 0.0
            reward -= p_timeout

            curriculum_index = {"easy": 0.0, "medium": 1.0, "hard": 2.0}.get(curriculum_level, 2.0)
            reward_info = {
                "reward_total": float(reward),
                "r_dist_progress": float(r_dist_progress),
                "r_xy_progress": float(r_xy_progress),
                "r_z_progress": float(r_z_progress),
                "r_yaw_progress": float(r_yaw_progress),
                "r_pose_maintain": float(r_pose_maintain),
                "r_success": float(r_success),
                "r_stable_bonus": float(r_stable_bonus),
                "p_idle": float(p_idle),
                "p_stagnation": float(p_stagnation),
                "p_action": float(p_action),
                "p_yaw_action": float(p_yaw_action),
                "p_workspace": float(p_workspace),
                "p_regress_near": float(p_regress_near),
                "p_phase_lock": float(p_phase_lock),
                "p_jitter": float(p_jitter),
                "p_timeout": float(p_timeout),
                "phase": phase,
                "yaw_error": float(curr_yaw_error),
                "aligned_hang_steps": 0.0,
                "curriculum_index": float(curriculum_index),
            }
            return float(reward), reward_info

        if self._is_precision_stage():
            phase = self._phase_from_metrics_precision(curr_xy_dist, curr_z_dist, curr_dist)

            reward = 0.0
            r_dist_progress = 0.60 * rcfg.w_dist_progress * (prev_dist - curr_dist)

            if phase == "approach":
                xy_weight = 1.00 * rcfg.w_xy_progress
                z_weight = 0.60 * rcfg.w_z_progress
            elif phase == "xy_lock":
                xy_weight = 1.25 * rcfg.w_xy_progress
                z_weight = 0.55 * rcfg.w_z_progress
            elif phase == "z_settle":
                xy_weight = 0.65 * rcfg.w_xy_progress
                z_weight = 1.20 * rcfg.w_z_progress
            else:
                xy_weight = 0.35 * rcfg.w_xy_progress
                z_weight = 0.40 * rcfg.w_z_progress

            r_xy_progress = xy_weight * (prev_xy_dist - curr_xy_dist)
            r_z_progress = z_weight * (prev_z_dist - curr_z_dist)
            reward += r_dist_progress + r_xy_progress + r_z_progress

            r_success = rcfg.w_success if success else 0.0
            reward += r_success

            near_pose = (
                curr_xy_dist < self.cfg.target.xy_success_dist * 1.5
                and curr_z_dist < self.cfg.target.z_success_dist * 1.5
            )

            r_stable_bonus = 0.0
            if near_pose and action_mag < 0.12 and ee_motion < 0.003:
                r_stable_bonus = rcfg.w_stable_bonus
                reward += r_stable_bonus

            p_idle = rcfg.w_idle_penalty if ee_motion < rcfg.idle_eps else 0.0
            reward -= p_idle

            p_action = rcfg.w_action_penalty * xyz_action_mag
            p_yaw_action = rcfg.w_yaw_action_penalty * yaw_action_mag
            reward -= (p_action + p_yaw_action)

            p_workspace = rcfg.w_workspace_violation if workspace_violated else 0.0
            reward -= p_workspace

            p_regress_near = 0.0
            if near_pose:
                if curr_xy_dist > prev_xy_dist + 1e-5:
                    p_regress_near += rcfg.w_regress_near
                if curr_z_dist > prev_z_dist + 1e-5:
                    p_regress_near += 0.8 * rcfg.w_regress_near
            reward -= p_regress_near

            p_jitter = 0.0
            if near_pose:
                p_jitter += rcfg.w_jitter_near * action_mag

                if prev_action is not None:
                    prev_mag = float(np.linalg.norm(prev_action))
                    curr_mag = float(np.linalg.norm(action))
                    if prev_mag > 1e-6 and curr_mag > 1e-6:
                        cosine = float(np.dot(prev_action, action) / (prev_mag * curr_mag + 1e-8))
                        if cosine < 0.0:
                            p_jitter += rcfg.w_reverse_action_near * abs(cosine)

                if curr_dist > prev_dist + 1e-5:
                    p_jitter += 1.25 * rcfg.w_jitter_near

            reward -= p_jitter

            p_timeout = rcfg.w_timeout_penalty if truncated and (not success) else 0.0
            reward -= p_timeout

            reward_info = {
                "reward_total": float(reward),
                "r_dist_progress": float(r_dist_progress),
                "r_xy_progress": float(r_xy_progress),
                "r_z_progress": float(r_z_progress),
                "r_success": float(r_success),
                "r_stable_bonus": float(r_stable_bonus),
                "p_idle": float(p_idle),
                "p_action": float(p_action),
                "p_yaw_action": float(p_yaw_action),
                "p_workspace": float(p_workspace),
                "p_regress_near": float(p_regress_near),
                "p_jitter": float(p_jitter),
                "p_timeout": float(p_timeout),
                "phase": phase,
                "yaw_error": float(curr_yaw_error),
                "aligned_hang_steps": 0.0,
            }
            return float(reward), reward_info

        phase = self._phase_from_metrics_basic(curr_xy_dist, curr_z_dist, curr_dist)

        reward = 0.0
        r_dist_progress = rcfg.w_dist_progress * (prev_dist - curr_dist)
        reward += r_dist_progress

        if phase in ("far", "align"):
            xy_weight = rcfg.w_xy_progress
            z_weight = 0.15 * rcfg.w_z_progress if self.cfg.substage == "1A" else 0.0
        elif phase == "descend":
            xy_weight = 0.70 * rcfg.w_xy_progress
            z_weight = rcfg.w_z_progress
        else:
            xy_weight = 0.55 * rcfg.w_xy_progress
            z_weight = 0.60 * rcfg.w_z_progress

        r_xy_progress = xy_weight * (prev_xy_dist - curr_xy_dist)
        r_z_progress = z_weight * (prev_z_dist - curr_z_dist)
        reward += r_xy_progress + r_z_progress

        r_descend_bonus = 0.0
        if phase == "descend" and (prev_z_dist - curr_z_dist) > 2e-4:
            r_descend_bonus = rcfg.w_descend_bonus
            reward += r_descend_bonus

        r_success = rcfg.w_success if success else 0.0
        reward += r_success

        p_idle = rcfg.w_idle_penalty if ee_motion < rcfg.idle_eps else 0.0
        reward -= p_idle

        p_wrong_descend = 0.0
        if (not xy_aligned) and float(action[2]) < 0.0:
            p_wrong_descend = rcfg.w_wrong_descend_penalty
            reward -= p_wrong_descend

        p_action = rcfg.w_action_penalty * xyz_action_mag
        p_yaw_action = rcfg.w_yaw_action_penalty * yaw_action_mag
        reward -= (p_action + p_yaw_action)

        p_workspace = rcfg.w_workspace_violation if workspace_violated else 0.0
        reward -= p_workspace

        p_xy_regress_descend = 0.0
        if phase == "descend" and curr_xy_dist > prev_xy_dist + 1e-5:
            p_xy_regress_descend = (
                rcfg.w_xy_regress_descend_penalty
                * min(1.0, (curr_xy_dist - prev_xy_dist) * 100.0)
            )
            reward -= p_xy_regress_descend

        p_hang_high = 0.0
        p_no_descend = 0.0
        if phase == "descend":
            z_progress = prev_z_dist - curr_z_dist
            if z_progress < 2e-4:
                self._aligned_hang_steps += 1
            else:
                self._aligned_hang_steps = 0

            if self._aligned_hang_steps >= rcfg.hang_steps_tolerance:
                severity = min(1.0, curr_z_dist / max(self.cfg.action.descend_z_gate, 1e-6))
                p_hang_high = rcfg.w_hang_high_penalty * severity
                reward -= p_hang_high

            if float(action[2]) >= -0.05:
                p_no_descend = rcfg.w_no_descend_penalty
                reward -= p_no_descend
        else:
            self._aligned_hang_steps = 0

        p_jitter = 0.0
        near_target = curr_dist < max(self.cfg.target.success_dist * 1.8, self.cfg.action.settle_dist)
        if near_target:
            p_jitter += rcfg.w_jitter_near * action_mag
            if prev_action is not None:
                prev_mag = float(np.linalg.norm(prev_action))
                curr_mag = float(np.linalg.norm(action))
                if prev_mag > 1e-6 and curr_mag > 1e-6:
                    cosine = float(np.dot(prev_action, action) / (prev_mag * curr_mag + 1e-8))
                    if cosine < 0.0:
                        p_jitter += rcfg.w_reverse_action_near * abs(cosine)
            if curr_dist > prev_dist + 1e-5:
                p_jitter += rcfg.w_jitter_near * 1.5
        reward -= p_jitter

        p_timeout = rcfg.w_timeout_penalty if truncated and (not success) else 0.0
        reward -= p_timeout

        curriculum_index = {"easy": 0.0, "medium": 1.0, "hard": 2.0}.get(curriculum_level, 2.0)
        reward_info = {
            "reward_total": float(reward),
            "r_dist_progress": float(r_dist_progress),
            "r_xy_progress": float(r_xy_progress),
            "r_z_progress": float(r_z_progress),
            "r_descend_bonus": float(r_descend_bonus),
            "r_success": float(r_success),
            "p_idle": float(p_idle),
            "p_wrong_descend": float(p_wrong_descend),
            "p_action": float(p_action),
            "p_yaw_action": float(p_yaw_action),
            "p_workspace": float(p_workspace),
            "p_xy_regress_descend": float(p_xy_regress_descend),
            "p_hang_high": float(p_hang_high),
            "p_no_descend": float(p_no_descend),
            "p_jitter": float(p_jitter),
            "p_timeout": float(p_timeout),
            "phase": phase,
            "yaw_error": float(curr_yaw_error),
            "aligned_hang_steps": float(self._aligned_hang_steps),
            "curriculum_index": float(curriculum_index),
        }
        return float(reward), reward_info