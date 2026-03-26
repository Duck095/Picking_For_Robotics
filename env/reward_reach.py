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

    def reset(self) -> None:
        self._aligned_hang_steps = 0

    def _phase_from_metrics(self, curr_xy_dist: float, curr_z_dist: float, curr_dist: float) -> str:
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
    ) -> Tuple[float, Dict[str, float]]:
        rcfg = self.cfg.reward
        substage = self.cfg.substage

        prev_delta = target_pos - prev_ee_pos
        curr_delta = target_pos - curr_ee_pos

        prev_dist = self._norm(prev_delta)
        curr_dist = self._norm(curr_delta)
        prev_xy_dist = self._norm(prev_delta[:2])
        curr_xy_dist = self._norm(curr_delta[:2])
        prev_z_dist = abs(float(prev_delta[2]))
        curr_z_dist = abs(float(curr_delta[2]))
        ee_motion = self._norm(curr_ee_pos - prev_ee_pos)

        phase = self._phase_from_metrics(curr_xy_dist, curr_z_dist, curr_dist)
        reward = 0.0

        r_dist_progress = rcfg.w_dist_progress * (prev_dist - curr_dist)
        reward += r_dist_progress

        if phase in ("far", "align"):
            xy_weight = rcfg.w_xy_progress
            z_weight = 0.15 * rcfg.w_z_progress if substage == "1A" else 0.0
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

        p_action = rcfg.w_action_penalty * float(np.linalg.norm(action))
        reward -= p_action

        p_workspace = rcfg.w_workspace_violation if workspace_violated else 0.0
        reward -= p_workspace

        p_xy_regress_descend = 0.0
        if phase == "descend" and curr_xy_dist > prev_xy_dist + 1e-5:
            p_xy_regress_descend = rcfg.w_xy_regress_descend_penalty * min(1.0, (curr_xy_dist - prev_xy_dist) * 100.0)
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
            action_mag = float(np.linalg.norm(action))
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
            "p_workspace": float(p_workspace),
            "p_xy_regress_descend": float(p_xy_regress_descend),
            "p_hang_high": float(p_hang_high),
            "p_no_descend": float(p_no_descend),
            "p_jitter": float(p_jitter),
            "p_timeout": float(p_timeout),
            "phase": phase,
            "aligned_hang_steps": float(self._aligned_hang_steps),
        }
        return float(reward), reward_info
