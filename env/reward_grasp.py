from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from config.grasp_env_config import Stage2GraspConfig


class RewardGrasp:
    def __init__(self, cfg: Stage2GraspConfig):
        self.cfg = cfg

    @staticmethod
    def _norm(vec: np.ndarray) -> float:
        return float(np.linalg.norm(vec))

    def compute(
        self,
        prev_ee_pos: np.ndarray,
        curr_ee_pos: np.ndarray,
        target_xy: np.ndarray,
        grasp_z: float,
        lift_z: float,
        home_pos: np.ndarray,
        action: np.ndarray,
        prev_action: np.ndarray,
        success: bool,
        truncated: bool,
        workspace_violated: bool,
        phase: str,
        grip_width: float,
        prev_grip_width: float,
        left_contact: bool,
        right_contact: bool,
        grasp_established: bool,
        hold_counter: int,
        lift_progress: float,
        object_lift_delta: float,
        object_dropped: bool,
    ) -> Tuple[float, Dict[str, float]]:
        rcfg = self.cfg.reward
        prev_xy = self._norm(prev_ee_pos[:2] - target_xy)
        curr_xy = self._norm(curr_ee_pos[:2] - target_xy)
        prev_z = abs(float(prev_ee_pos[2] - grasp_z))
        curr_z = abs(float(curr_ee_pos[2] - grasp_z))
        prev_lift = abs(float(prev_ee_pos[2] - lift_z))
        curr_lift = abs(float(curr_ee_pos[2] - lift_z))
        prev_home = self._norm(prev_ee_pos - home_pos)
        curr_home = self._norm(curr_ee_pos - home_pos)
        ee_motion = self._norm(curr_ee_pos - prev_ee_pos)
        action_mag = float(np.linalg.norm(action))
        z_up = float(curr_ee_pos[2] - prev_ee_pos[2])

        reward = 0.0
        r_xy = r_z = r_grip = r_contact = r_grasp = r_hold = r_lift = r_home = r_success = 0.0
        if phase == "xy_align":
            r_xy = rcfg.w_xy_progress * (prev_xy - curr_xy)
        elif phase == "descend":
            r_xy = 0.7 * rcfg.w_xy_progress * (prev_xy - curr_xy)
            r_z = rcfg.w_z_progress * (prev_z - curr_z)
        elif phase == "close":
            r_xy = 0.4 * rcfg.w_xy_progress * (prev_xy - curr_xy)
            r_z = 0.6 * rcfg.w_z_progress * (prev_z - curr_z)
            r_grip = rcfg.w_grip_progress * max(0.0, prev_grip_width - grip_width)
        elif phase == "hold":
            if grasp_established:
                r_hold = rcfg.w_hold_bonus * (1.0 + 0.15 * float(hold_counter))
        elif phase == "lift":
            r_lift = rcfg.w_lift_progress * max(0.0, prev_lift - curr_lift)
            r_lift += 0.8 * rcfg.w_lift_progress * max(0.0, lift_progress)
            r_lift += 0.4 * max(0.0, object_lift_delta)
        elif phase == "return_home":
            r_home = rcfg.w_home_progress * max(0.0, prev_home - curr_home)

        if phase != "return_home":
            if left_contact:
                r_contact += rcfg.w_contact_touch
            if right_contact:
                r_contact += rcfg.w_contact_touch
            if left_contact and right_contact:
                if grasp_established:
                    r_contact += rcfg.w_contact_stable + rcfg.w_dual_contact_bonus
                else:
                    r_contact += 0.05
            if grasp_established:
                r_grasp += rcfg.w_true_grasp_bonus
        if success:
            r_success = rcfg.w_success

        reward += r_xy + r_z + r_grip + r_contact + r_grasp + r_hold + r_lift + r_home + r_success

        p_action = rcfg.w_action_penalty * action_mag
        p_idle = rcfg.w_idle_penalty if ee_motion < rcfg.idle_eps else 0.0
        p_workspace = rcfg.w_workspace_violation if workspace_violated else 0.0
        p_xy_regress = rcfg.w_xy_regress_penalty if curr_xy > prev_xy + 1e-5 else 0.0
        p_z_up = 0.0
        p_wrong_close = 0.0
        p_close_no_contact = 0.0
        p_false_contact = 0.0
        p_home_stall = 0.0
        p_drop = rcfg.w_drop_penalty if object_dropped else 0.0
        p_timeout = rcfg.w_timeout_penalty if truncated and (not success) else 0.0

        if phase == "descend" and z_up > 1e-5:
            p_z_up = rcfg.w_z_up_penalty
        if phase == "close" and z_up > 1e-5:
            p_z_up = max(p_z_up, 1.5 * rcfg.w_z_up_penalty)
        if phase in ("xy_align", "descend") and action[3] < -0.05:
            p_wrong_close = rcfg.w_wrong_close_penalty
        if phase == "close" and grip_width <= 0.03 and not (left_contact or right_contact):
            p_close_no_contact = rcfg.w_close_no_contact_penalty
        if (left_contact and right_contact) and (not grasp_established):
            p_false_contact = rcfg.w_false_contact_penalty
        if phase == "lift" and z_up <= 0.0:
            p_z_up += 0.5 * rcfg.w_z_up_penalty
        if phase == "return_home" and curr_home >= prev_home - 1e-4:
            p_home_stall = 0.20
        if phase == "return_home" and not grasp_established:
            p_drop += 0.75 * rcfg.w_drop_penalty

        reward -= p_action + p_idle + p_workspace + p_xy_regress + p_z_up + p_wrong_close + p_close_no_contact + p_false_contact + p_home_stall + p_drop + p_timeout

        info = {
            "reward_total": float(reward),
            "r_xy_progress": float(r_xy),
            "r_z_progress": float(r_z),
            "r_grip_progress": float(r_grip),
            "r_contact": float(r_contact),
            "r_true_grasp": float(r_grasp),
            "r_hold_bonus": float(r_hold),
            "r_lift_progress": float(r_lift),
            "r_home_progress": float(r_home),
            "r_success": float(r_success),
            "p_action": float(p_action),
            "p_idle": float(p_idle),
            "p_workspace": float(p_workspace),
            "p_xy_regress": float(p_xy_regress),
            "p_z_up": float(p_z_up),
            "p_wrong_close": float(p_wrong_close),
            "p_close_no_contact": float(p_close_no_contact),
            "p_false_contact": float(p_false_contact),
            "p_home_stall": float(p_home_stall),
            "p_drop": float(p_drop),
            "p_timeout": float(p_timeout),
        }
        return float(reward), info
