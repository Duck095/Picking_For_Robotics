from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from env.camera import Camera
from env.panda_controller import PandaController
from env.reward_grasp import RewardGrasp
from config.grasp_env_config import Stage2GraspConfig, build_stage2_grasp_config


class GraspEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: Optional[Stage2GraspConfig] = None):
        super().__init__()
        self.cfg: Stage2GraspConfig = cfg if cfg is not None else build_stage2_grasp_config("2A")

        self.client_id: Optional[int] = None
        self.robot: Optional[PandaController] = None
        self.camera = None
        self.plane_id: Optional[int] = None
        self.table_id: Optional[int] = None
        self.object_id: Optional[int] = None

        self.step_count = 0
        self.episode_count = 0

        self.object_pos = np.zeros(3, dtype=np.float32)
        self.object_yaw = 0.0
        self.pregrasp_pos = np.zeros(3, dtype=np.float32)
        self.grasp_pos = np.zeros(3, dtype=np.float32)
        self.lift_pos = np.zeros(3, dtype=np.float32)
        self.target_yaw = 0.0

        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_ee_pos = np.zeros(3, dtype=np.float32)
        self.prev_grip_width = float(self.cfg.gripper.initial_width)
        self.current_grip_width = float(self.cfg.gripper.initial_width)

        self.phase = "xy_align"
        self.grasp_hold_counter = 0
        self.lift_hold_counter = 0
        self.object_initial_z = 0.0
        self.object_lift_delta = 0.0

        self.left_finger_link_index: Optional[int] = None
        self.right_finger_link_index: Optional[int] = None

        self.rewarder = RewardGrasp(self.cfg)

        self._connect()
        self._build_static_world()
        self._build_robot()
        self._cache_finger_links()
        if Camera is not None:
            self.camera = Camera()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        obs_dim = self._get_obs_dim()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _connect(self) -> None:
        self.client_id = p.connect(p.GUI if self.cfg.sim.use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(1.0 / float(self.cfg.sim.physics_hz))

    def _build_static_world(self) -> None:
        self.plane_id = p.loadURDF("plane.urdf")
        table_half = [0.45, 0.60, 0.02]
        table_pos = [0.55, 0.0, -0.02]

        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half)
        vis_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=table_half,
            rgbaColor=[0.75, 0.75, 0.75, 1.0],
        )
        self.table_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=table_pos,
        )

    def _build_robot(self) -> None:
        self.robot = PandaController(
            client_id=self.client_id,
            use_gui=self.cfg.sim.use_gui,
            workspace=self.cfg.workspace,
        )

    def _cache_finger_links(self) -> None:
        assert self.robot is not None and self.robot.robot_id is not None
        num_joints = p.getNumJoints(self.robot.robot_id)
        for joint_idx in range(num_joints):
            info = p.getJointInfo(self.robot.robot_id, joint_idx)
            name = info[1].decode("utf-8")
            if name == "panda_finger_joint1":
                self.left_finger_link_index = joint_idx
            elif name == "panda_finger_joint2":
                self.right_finger_link_index = joint_idx

        if self.left_finger_link_index is None or self.right_finger_link_index is None:
            raise RuntimeError("Không tìm được finger link indices của Panda.")

    def _remove_object(self) -> None:
        if self.object_id is not None:
            try:
                p.removeBody(self.object_id)
            except Exception:
                pass
            self.object_id = None

    def _sample_object_pose(self) -> Tuple[np.ndarray, float]:
        s = self.cfg.spawn
        if s.spawn_mode == "fixed":
            pos = np.array(s.fixed_pose_xyz, dtype=np.float32)
        else:
            pos = np.array([
                self.np_random.uniform(s.roi_x[0], s.roi_x[1]),
                self.np_random.uniform(s.roi_y[0], s.roi_y[1]),
                s.object_z,
            ], dtype=np.float32)

        yaw = self.np_random.uniform(s.yaw_range[0], s.yaw_range[1]) if s.random_yaw else 0.0
        return pos, float(yaw)

    def _spawn_object(self) -> None:
        self._remove_object()
        self.object_pos, self.object_yaw = self._sample_object_pose()

        color_name = self.np_random.choice(self.cfg.spawn.object_colors)
        color_map = {
            "red": [1.0, 0.2, 0.2, 1.0],
            "green": [0.2, 1.0, 0.2, 1.0],
            "blue": [0.2, 0.4, 1.0, 1.0],
            "yellow": [1.0, 1.0, 0.2, 1.0],
        }
        rgba = color_map.get(color_name, [1.0, 0.2, 0.2, 1.0])

        half_extents = list(self.cfg.spawn.cube_half_extents)
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=rgba,
        )

        orn = p.getQuaternionFromEuler([0.0, 0.0, self.object_yaw])
        self.object_id = p.createMultiBody(
            baseMass=self.cfg.spawn.object_mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=self.object_pos.tolist(),
            baseOrientation=orn,
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def _settle_object(self, steps: int = 24) -> None:
        for _ in range(int(steps)):
            p.stepSimulation()

    def _refresh_object_state(self) -> Tuple[np.ndarray, float]:
        object_pos, object_yaw = self._get_object_pose()
        self.object_pos = object_pos.astype(np.float32)
        self.object_yaw = float(object_yaw)
        return self.object_pos, self.object_yaw

    def _compute_targets(self) -> None:
        object_pos, object_yaw = self._refresh_object_state()

        self.target_yaw = self._wrap_angle(object_yaw)
        self.pregrasp_pos = np.array(
            [object_pos[0], object_pos[1], self.cfg.target.pregrasp_z],
            dtype=np.float32,
        )
        self.grasp_pos = np.array(
            [object_pos[0], object_pos[1], self.cfg.target.grasp_z],
            dtype=np.float32,
        )
        self.lift_pos = np.array(
            [object_pos[0], object_pos[1], self.cfg.target.lift_z],
            dtype=np.float32,
        )

    def _refresh_targets_for_step(self) -> None:
        object_pos, object_yaw = self._refresh_object_state()

        self.pregrasp_pos[:2] = object_pos[:2]
        self.grasp_pos[:2] = object_pos[:2]
        self.lift_pos[:2] = object_pos[:2]
        self.pregrasp_pos[2] = self.cfg.target.pregrasp_z
        self.grasp_pos[2] = self.cfg.target.grasp_z
        self.lift_pos[2] = self.cfg.target.lift_z

        if self.phase in ("xy_align", "descend", "close"):
            self.target_yaw = self._wrap_angle(object_yaw)

    def _move_robot_to_pregrasp(self) -> None:
        assert self.robot is not None
        self.robot.reset_home(open_gripper=True)
        self.current_grip_width = float(self.cfg.gripper.initial_width)
        self.robot.set_gripper_opening(
            width=self.current_grip_width,
            force=self.cfg.gripper.position_force,
        )
        self.robot.move_ee_to_with_yaw(
            target_pos=self.pregrasp_pos,
            target_yaw=self.target_yaw,
            substeps=60,
        )
        self.robot.step_simulation(20)

    def _get_object_pose(self) -> Tuple[np.ndarray, float]:
        assert self.object_id is not None
        pos, orn = p.getBasePositionAndOrientation(self.object_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        return np.array(pos, dtype=np.float32), float(yaw)

    def _get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray, float]:
        assert self.robot is not None
        ee_pos, ee_orn = self.robot.get_ee_pose()
        ee_yaw = self.robot.get_ee_yaw()
        return ee_pos.astype(np.float32), ee_orn.astype(np.float32), float(ee_yaw)

    def _get_finger_midpoint(self) -> np.ndarray:
        assert self.robot is not None
        assert self.left_finger_link_index is not None
        assert self.right_finger_link_index is not None

        left_state = p.getLinkState(
            self.robot.robot_id,
            self.left_finger_link_index,
            computeForwardKinematics=True,
        )
        right_state = p.getLinkState(
            self.robot.robot_id,
            self.right_finger_link_index,
            computeForwardKinematics=True,
        )

        left_pos = np.array(left_state[4], dtype=np.float32)
        right_pos = np.array(right_state[4], dtype=np.float32)
        return (left_pos + right_pos) * 0.5

    def _get_target_pose_for_phase(self) -> np.ndarray:
        if self.phase == "xy_align":
            return self.pregrasp_pos
        if self.phase in ("descend", "close", "hold"):
            return self.grasp_pos
        return self.lift_pos

    def _get_obs_dim(self) -> int:
        dim = 3 + 3 + 3 + 3 + 3 + 1 + 1 + 1 + 1 + 1 + 2 + 3
        if self.cfg.include_joint_state:
            dim += 7
        if self.cfg.include_joint_velocity:
            dim += 7
        if self.cfg.include_prev_action:
            dim += 4
        return dim

    def _check_workspace_violated(self) -> bool:
        assert self.robot is not None
        ee_pos, _, _ = self._get_ee_pose()
        return not self.robot.is_inside_workspace(ee_pos)

    def _get_contacts(self) -> Tuple[bool, bool]:
        assert self.robot is not None and self.object_id is not None
        left_contact = False
        right_contact = False

        contacts = p.getContactPoints(bodyA=self.robot.robot_id, bodyB=self.object_id)
        for c in contacts:
            link_idx = c[3]
            normal_force = float(c[9])

            if link_idx == self.left_finger_link_index and normal_force >= self.cfg.gripper.contact_force_threshold:
                left_contact = True
            elif link_idx == self.right_finger_link_index and normal_force >= self.cfg.gripper.contact_force_threshold:
                right_contact = True

        return left_contact, right_contact

    def _is_xy_aligned(self, ee_yaw: float) -> bool:
        finger_mid = self._get_finger_midpoint()
        xy_error = float(np.linalg.norm(finger_mid[:2] - self.object_pos[:2]))
        yaw_error = abs(self._wrap_angle(self.target_yaw - ee_yaw))
        return (
            xy_error <= self.cfg.target.xy_phase_threshold
            and yaw_error <= self.cfg.target.yaw_phase_threshold
        )

    def _is_descend_ready(self, ee_yaw: float) -> bool:
        ee_pos, _, _ = self._get_ee_pose()
        finger_mid = self._get_finger_midpoint()

        xy_error = float(np.linalg.norm(finger_mid[:2] - self.object_pos[:2]))
        z_error = abs(float(ee_pos[2] - self.grasp_pos[2]))
        yaw_error = abs(self._wrap_angle(self.target_yaw - ee_yaw))
        return (
            xy_error <= self.cfg.target.xy_phase_threshold
            and z_error <= self.cfg.target.z_phase_threshold
            and yaw_error <= self.cfg.target.yaw_phase_threshold
        )

    def _is_grasp_established(self) -> bool:
        left_contact, right_contact = self._get_contacts()
        object_pos, _ = self._get_object_pose()
        finger_mid = self._get_finger_midpoint()

        xy_error = float(np.linalg.norm(finger_mid[:2] - object_pos[:2]))
        width_ok = self.current_grip_width <= self.cfg.target.grip_success_width_2a

        return (
            left_contact
            and right_contact
            and xy_error <= self.cfg.target.finger_xy_success_dist_2a
            and width_ok
        )

    def _update_phase(self, ee_yaw: float) -> None:
        grasp_established = self._is_grasp_established()

        if self.phase == "xy_align":
            if self._is_xy_aligned(ee_yaw):
                self.phase = "descend"

        elif self.phase == "descend":
            if self._is_descend_ready(ee_yaw):
                self.phase = "close"

        elif self.phase == "close":
            if grasp_established:
                if self.cfg.substage == "2A":
                    return
                self.phase = "hold"
                self.grasp_hold_counter = 0

        elif self.phase == "hold":
            if grasp_established:
                self.grasp_hold_counter += 1
            else:
                self.grasp_hold_counter = 0

            if self.grasp_hold_counter >= self.cfg.target.stable_grasp_steps_required:
                if self.cfg.substage == "2B":
                    return
                self.phase = "lift"
                self.lift_hold_counter = 0

        elif self.phase == "lift":
            object_pos, _ = self._get_object_pose()
            lift_ok = object_pos[2] - self.object_initial_z >= self.cfg.target.lift_success_delta_z
            if grasp_established and lift_ok:
                self.lift_hold_counter += 1
            else:
                self.lift_hold_counter = 0

    def _check_success(self) -> bool:
        grasp_established = self._is_grasp_established()
        object_pos, _ = self._get_object_pose()
        lift_delta = float(object_pos[2] - self.object_initial_z)

        if self.cfg.substage == "2A":
            return grasp_established

        if self.cfg.substage == "2B":
            return grasp_established and self.grasp_hold_counter >= self.cfg.target.stable_grasp_steps_required

        return (
            grasp_established
            and lift_delta >= self.cfg.target.lift_success_delta_z
            and self.lift_hold_counter >= self.cfg.target.stable_lift_steps_required
        )

    def _check_truncated(self) -> bool:
        return self.step_count >= int(self.cfg.sim.max_steps)

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        if self.cfg.action.clip_action:
            action = np.clip(action, -1.0, 1.0)

        scaled = np.array([
            float(action[0]) * float(self.cfg.action.action_scale_xy),
            float(action[1]) * float(self.cfg.action.action_scale_xy),
            float(action[2]) * float(self.cfg.action.action_scale_z),
            float(action[3]) * float(self.cfg.action.action_scale_grip),
        ], dtype=np.float32)

        if self.phase == "xy_align":
            scaled[0] *= 0.35
            scaled[1] *= 0.35
            scaled[2] = 0.0
            scaled[3] = 0.0

        elif self.phase == "descend":
            scaled[0] *= 0.15
            scaled[1] *= 0.15
            scaled[2] = min(scaled[2], -0.002)
            scaled[3] = 0.0

        elif self.phase == "close":
            scaled[0] *= 0.05
            scaled[1] *= 0.05
            scaled[2] = -0.0015
            scaled[3] = -max(0.012, abs(scaled[3]))

        elif self.phase == "hold":
            scaled[0] *= 0.04
            scaled[1] *= 0.04
            scaled[2] = 0.0
            scaled[3] = -max(0.010, abs(scaled[3]))

        elif self.phase == "lift":
            scaled[0] *= 0.08
            scaled[1] *= 0.08
            min_lift_dz = 0.003 if self.cfg.substage == "2C" else 0.002
            scaled[2] = max(scaled[2], min_lift_dz)
            scaled[3] = -max(0.010, abs(scaled[3]))

        return scaled

    def _apply_action(self, action: np.ndarray) -> None:
        assert self.robot is not None
        alpha = float(np.clip(self.cfg.action.action_smoothing, 0.0, 0.95))
        smoothed = ((1.0 - alpha) * action + alpha * self.prev_action).astype(np.float32)

        dx, dy, dz, dgrip = self._scale_action(smoothed)

        ee_pos, _, _ = self._get_ee_pose()

        if self.phase == "xy_align":
            # Anti-drift:
            # Bám quanh pregrasp_pos thay vì cộng dồn từ ee_pos để tránh quay yaw làm XY trôi xa dần.
            target_pos = self.pregrasp_pos.copy()
            target_pos[0] += dx
            target_pos[1] += dy
        elif self.phase == "descend":
            # Khi đã descend, giữ XY bám quanh grasp_pos thay vì để sai số tích lũy.
            target_pos = self.grasp_pos.copy()
            target_pos[0] += dx
            target_pos[1] += dy
            target_pos[2] = ee_pos[2] + dz
        else:
            target_pos = ee_pos + np.array([dx, dy, dz], dtype=np.float32)

        target_pos = self.robot.clip_to_workspace(target_pos)

        self.robot.move_ee_to_with_yaw(
            target_pos=target_pos,
            target_yaw=self.target_yaw,
            substeps=self.cfg.sim.substeps,
        )

        self.prev_grip_width = float(self.current_grip_width)
        self.current_grip_width = float(np.clip(
            self.current_grip_width + dgrip,
            self.cfg.gripper.close_width,
            self.cfg.gripper.open_width,
        ))
        self.robot.set_gripper_opening(
            width=self.current_grip_width,
            force=self.cfg.gripper.position_force,
        )
        self.robot.step_simulation(2)

    def _get_obs(self) -> np.ndarray:
        assert self.robot is not None

        ee_pos, _, ee_yaw = self._get_ee_pose()
        object_pos, _ = self._get_object_pose()
        target_pos = self._get_target_pose_for_phase()
        finger_mid = self._get_finger_midpoint()

        delta = target_pos - ee_pos
        object_delta = object_pos - ee_pos
        xy_dist = float(np.linalg.norm(finger_mid[:2] - object_pos[:2]))
        z_dist = (
            abs(float(ee_pos[2] - self.grasp_pos[2]))
            if self.phase != "lift"
            else abs(float(ee_pos[2] - self.lift_pos[2]))
        )
        yaw_error = self._wrap_angle(self.target_yaw - ee_yaw)
        grip_norm = self.current_grip_width / max(self.cfg.gripper.open_width, 1e-6)
        object_lift_delta = float(object_pos[2] - self.object_initial_z)
        left_contact, right_contact = self._get_contacts()

        phase_onehot = np.array([
            1.0 if self.phase == "xy_align" else 0.0,
            1.0 if self.phase == "descend" else 0.0,
            1.0 if self.phase == "close" else 0.0,
        ], dtype=np.float32)

        obs_parts = [
            ee_pos.astype(np.float32),
            object_pos.astype(np.float32),
            target_pos.astype(np.float32),
            delta.astype(np.float32),
            object_delta.astype(np.float32),
            np.array([xy_dist], dtype=np.float32),
            np.array([z_dist], dtype=np.float32),
            np.array([yaw_error], dtype=np.float32),
            np.array([grip_norm], dtype=np.float32),
            np.array([object_lift_delta], dtype=np.float32),
            np.array([1.0 if left_contact else 0.0, 1.0 if right_contact else 0.0], dtype=np.float32),
            phase_onehot,
        ]

        if self.cfg.include_joint_state:
            obs_parts.append(self.robot.get_arm_joint_positions().astype(np.float32))
        if self.cfg.include_joint_velocity:
            obs_parts.append(self.robot.get_arm_joint_velocities().astype(np.float32))
        if self.cfg.include_prev_action:
            obs_parts.append(self.prev_action.astype(np.float32))

        return np.concatenate(obs_parts, axis=0).astype(np.float32)

    def _debug_draw(self) -> None:
        if not self.cfg.sim.use_gui or self.robot is None:
            return

        ee_pos, _, _ = self._get_ee_pose()
        target_pos = self._get_target_pose_for_phase()

        if self.cfg.debug.draw_target and hasattr(self.robot, "draw_target_pose"):
            self.robot.draw_target_pose(
                target_pos=target_pos,
                target_yaw=self.target_yaw,
                axis_len=0.05,
                life_time=0.05,
            )
            p.addUserDebugText(
                f"phase={self.phase} hold={self.grasp_hold_counter}",
                target_pos.tolist(),
                [1, 0, 0],
                lifeTime=0.05,
            )

        if self.cfg.debug.draw_ee_path:
            p.addUserDebugLine(
                ee_pos.tolist(),
                target_pos.tolist(),
                [0, 1, 0],
                lineWidth=2.0,
                lifeTime=0.05,
            )

    def _build_info(self, success: bool, truncated: bool, ee_yaw: float) -> Dict[str, Any]:
        ee_pos, _, _ = self._get_ee_pose()
        object_pos, _ = self._get_object_pose()
        target_pos = self._get_target_pose_for_phase()
        finger_mid = self._get_finger_midpoint()

        delta = target_pos - ee_pos
        xy_dist = float(np.linalg.norm(finger_mid[:2] - object_pos[:2]))
        z_dist = abs(float(ee_pos[2] - self.grasp_pos[2]))
        yaw_error = abs(self._wrap_angle(self.target_yaw - ee_yaw))

        left_contact, right_contact = self._get_contacts()
        grasp_established = self._is_grasp_established()
        lift_delta = float(object_pos[2] - self.object_initial_z)

        ee_z_world = float(ee_pos[2])
        grasp_z_world = float(self.grasp_pos[2])
        cube_center_z_world = float(object_pos[2])
        finger_mid_z_world = float(finger_mid[2])

        finger_xy_error = float(np.linalg.norm(finger_mid[:2] - object_pos[:2]))
        finger_z_error = abs(float(finger_mid[2] - object_pos[2]))
        ee_xy_error = float(np.linalg.norm(ee_pos[:2] - object_pos[:2]))
        ee_z_error = abs(float(ee_pos[2] - self.grasp_pos[2]))

        return {
            "episode_idx": self.episode_count,
            "step": self.step_count,
            "object_pos": object_pos.copy(),
            "object_yaw": float(self.object_yaw),
            "target_pos": target_pos.copy(),
            "target_yaw": float(self.target_yaw),
            "dist": float(np.linalg.norm(delta)),
            "xy_dist": xy_dist,
            "z_dist": z_dist,
            "yaw_error": yaw_error,
            "success": bool(success),
            "truncated": bool(truncated),
            "phase": self.phase,
            "substage": self.cfg.substage,
            "stable_pose_steps": self.grasp_hold_counter,
            "lift_hold_steps": self.lift_hold_counter,
            "grip_width": float(self.current_grip_width),
            "left_contact": bool(left_contact),
            "right_contact": bool(right_contact),
            "grasp_established": bool(grasp_established),
            "object_lift_delta": float(lift_delta),
            "xy_aligned": bool(self._is_xy_aligned(ee_yaw)),
            "pregrasp_z": float(self.pregrasp_pos[2]),
            "grasp_z": float(self.grasp_pos[2]),
            "ee_z_world": ee_z_world,
            "grasp_z_world": grasp_z_world,
            "cube_center_z_world": cube_center_z_world,
            "finger_mid_z_world": finger_mid_z_world,
            "ee_to_grasp_z_error": ee_z_error,
            "ee_to_cube_xy_error": ee_xy_error,
            "finger_mid_to_cube_xy_error": finger_xy_error,
            "finger_mid_to_cube_z_error": finger_z_error,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        assert self.robot is not None

        self.step_count = 0
        self.episode_count += 1
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_ee_pos = np.zeros(3, dtype=np.float32)
        self.prev_grip_width = float(self.cfg.gripper.initial_width)
        self.current_grip_width = float(self.cfg.gripper.initial_width)
        self.phase = "xy_align"
        self.grasp_hold_counter = 0
        self.lift_hold_counter = 0

        self._spawn_object()
        self._settle_object(steps=32)
        self._compute_targets()
        self._move_robot_to_pregrasp()

        self._settle_object(steps=8)
        self._compute_targets()

        object_pos, _ = self._get_object_pose()
        self.object_initial_z = float(object_pos[2])
        self.object_lift_delta = 0.0

        ee_pos, _, ee_yaw = self._get_ee_pose()
        self.prev_ee_pos = ee_pos.astype(np.float32)

        obs = self._get_obs()
        info = self._build_info(success=False, truncated=False, ee_yaw=ee_yaw)
        return obs, info

    def step(self, action: np.ndarray):
        assert self.robot is not None

        action = np.asarray(action, dtype=np.float32).reshape(4,)
        prev_ee_pos, _, _ = self._get_ee_pose()
        self.prev_ee_pos = prev_ee_pos.astype(np.float32)
        prev_object_pos, _ = self._get_object_pose()

        self._refresh_targets_for_step()
        self._apply_action(action)
        self.step_count += 1

        curr_ee_pos, _, curr_ee_yaw = self._get_ee_pose()
        self._refresh_targets_for_step()
        self._update_phase(curr_ee_yaw)

        curr_object_pos, _ = self._get_object_pose()
        self.object_lift_delta = float(curr_object_pos[2] - self.object_initial_z)
        object_dropped = self.phase in ("hold", "lift") and self.object_lift_delta < -0.002

        success = self._check_success()
        truncated = self._check_truncated()
        workspace_violated = self._check_workspace_violated()
        left_contact, right_contact = self._get_contacts()
        grasp_established = self._is_grasp_established()

        prev_lift_delta = float(prev_object_pos[2] - self.object_initial_z)
        lift_progress = self.object_lift_delta - prev_lift_delta

        reward, reward_info = self.rewarder.compute(
            prev_ee_pos=self.prev_ee_pos,
            curr_ee_pos=curr_ee_pos.astype(np.float32),
            target_xy=self.object_pos[:2].astype(np.float32),
            grasp_z=float(self.grasp_pos[2]),
            lift_z=float(self.lift_pos[2]),
            action=action,
            prev_action=self.prev_action.copy(),
            success=success,
            truncated=truncated,
            workspace_violated=workspace_violated,
            phase=self.phase,
            grip_width=self.current_grip_width,
            prev_grip_width=self.prev_grip_width,
            left_contact=left_contact,
            right_contact=right_contact,
            grasp_established=grasp_established,
            hold_counter=max(self.grasp_hold_counter, self.lift_hold_counter),
            lift_progress=float(lift_progress),
            object_lift_delta=self.object_lift_delta,
            object_dropped=object_dropped,
        )

        self.prev_action = action.copy()
        obs = self._get_obs()
        terminated = bool(success)
        info = self._build_info(success=success, truncated=truncated, ee_yaw=curr_ee_yaw)
        info.update(reward_info)

        self._debug_draw()
        return obs, float(reward), terminated, bool(truncated), info

    def close(self):
        try:
            if self.object_id is not None:
                p.removeBody(self.object_id)
                self.object_id = None
        except Exception:
            pass

        try:
            p.disconnect()
        except Exception:
            pass
