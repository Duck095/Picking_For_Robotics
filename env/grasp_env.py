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
    PHASE_NAMES = ["xy_align", "descend", "close", "hold", "lift", "return_home"]

    def __init__(self, cfg: Optional[Stage2GraspConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else build_stage2_grasp_config("2A")

        self.client_id = None
        self.robot: Optional[PandaController] = None
        self.camera = None
        self.object_id = None
        self.plane_id = None
        self.table_id = None

        self.step_count = 0
        self.episode_count = 0

        self.phase = "xy_align"
        self.grasp_hold_counter = 0
        self.lift_hold_counter = 0
        self.home_hold_counter = 0
        self.object_initial_z = 0.0
        self.object_lift_delta = 0.0
        self.lift_lost_counter = 0

        self.object_pos = np.zeros(3, dtype=np.float32)
        self.object_yaw = 0.0
        self.pregrasp_pos = np.zeros(3, dtype=np.float32)
        self.grasp_pos = np.zeros(3, dtype=np.float32)
        self.lift_pos = np.zeros(3, dtype=np.float32)
        self.home_pos = np.zeros(3, dtype=np.float32)
        self.target_yaw = 0.0

        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_ee_pos = np.zeros(3, dtype=np.float32)
        self.prev_grip_width = float(self.cfg.gripper.initial_width)
        self.current_grip_width = float(self.cfg.gripper.initial_width)
        self.gripper_locked = False
        self.locked_grip_width = float(self.cfg.gripper.initial_width)

        self.left_finger_link_index = None
        self.right_finger_link_index = None

        self.rewarder = RewardGrasp(self.cfg)

        self._connect()
        self._build_static_world()
        self._build_robot()
        self._cache_finger_links()
        if Camera is not None:
            self.camera = Camera()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._get_obs_dim(),),
            dtype=np.float32,
        )

    def _phase_onehot(self):
        vec = np.zeros(len(self.PHASE_NAMES), dtype=np.float32)
        vec[self.PHASE_NAMES.index(self.phase)] = 1.0
        return vec

    def _connect(self):
        self.client_id = p.connect(p.GUI if self.cfg.sim.use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(1.0 / float(self.cfg.sim.physics_hz))

    def _build_static_world(self):
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

    def _build_robot(self):
        self.robot = PandaController(
            client_id=self.client_id,
            use_gui=self.cfg.sim.use_gui,
            workspace=self.cfg.workspace,
        )

    def _cache_finger_links(self):
        num_joints = p.getNumJoints(self.robot.robot_id)
        for joint_idx in range(num_joints):
            name = p.getJointInfo(self.robot.robot_id, joint_idx)[1].decode("utf-8")
            if name == "panda_finger_joint1":
                self.left_finger_link_index = joint_idx
            elif name == "panda_finger_joint2":
                self.right_finger_link_index = joint_idx

        if self.left_finger_link_index is None or self.right_finger_link_index is None:
            raise RuntimeError("Không tìm được finger links.")

        for link_idx in [self.left_finger_link_index, self.right_finger_link_index]:
            p.changeDynamics(
                self.robot.robot_id,
                link_idx,
                lateralFriction=2.0,
                spinningFriction=0.01,
                rollingFriction=0.01,
                restitution=0.0,
                physicsClientId=self.client_id,
            )

    def _remove_object(self):
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
            pos = np.array(
                [
                    self.np_random.uniform(s.roi_x[0], s.roi_x[1]),
                    self.np_random.uniform(s.roi_y[0], s.roi_y[1]),
                    s.object_z,
                ],
                dtype=np.float32,
            )
        yaw = self.np_random.uniform(s.yaw_range[0], s.yaw_range[1]) if s.random_yaw else 0.0
        return pos, float(yaw)

    def _spawn_object(self):
        self._remove_object()
        self.object_pos, self.object_yaw = self._sample_object_pose()

        color_name = self.np_random.choice(self.cfg.spawn.object_colors)
        cmap = {
            "red": [1.0, 0.2, 0.2, 1.0],
            "green": [0.2, 1.0, 0.2, 1.0],
            "blue": [0.2, 0.4, 1.0, 1.0],
            "yellow": [1.0, 1.0, 0.2, 1.0],
        }
        rgba = cmap.get(color_name, [1.0, 0.2, 0.2, 1.0])

        he = list(self.cfg.spawn.cube_half_extents)
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=he)
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=rgba)
        orn = p.getQuaternionFromEuler([0.0, 0.0, self.object_yaw])

        self.object_id = p.createMultiBody(
            baseMass=self.cfg.spawn.object_mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=self.object_pos.tolist(),
            baseOrientation=orn,
        )

        p.changeDynamics(
            self.object_id,
            -1,
            lateralFriction=1.2,
            spinningFriction=0.002,
            rollingFriction=0.002,
            restitution=0.0,
            physicsClientId=self.client_id,
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def _settle_object(self, steps=24):
        for _ in range(int(steps)):
            p.stepSimulation()

    def _get_object_pose(self):
        pos, orn = p.getBasePositionAndOrientation(self.object_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        return np.array(pos, dtype=np.float32), float(yaw)

    def _refresh_object_state(self):
        object_pos, object_yaw = self._get_object_pose()
        self.object_pos = object_pos.astype(np.float32)
        self.object_yaw = float(object_yaw)
        return self.object_pos, self.object_yaw

    def _get_ee_pose(self):
        ee_pos, ee_orn = self.robot.get_ee_pose()
        ee_yaw = self.robot.get_ee_yaw()
        return ee_pos.astype(np.float32), ee_orn.astype(np.float32), float(ee_yaw)

    def _get_finger_midpoint(self):
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
        return (
            np.array(left_state[4], dtype=np.float32)
            + np.array(right_state[4], dtype=np.float32)
        ) * 0.5

    def _apply_locked_gripper_width(self, width: float):
        width = float(np.clip(width, self.cfg.gripper.close_width, self.cfg.gripper.open_width))
        half = width * 0.5
        joint_ids = self.robot.gripper_joint_indices if hasattr(self.robot, "gripper_joint_indices") else [9, 10]
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot.robot_id,
            jointIndices=joint_ids,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[half, half],
            forces=[self.cfg.gripper.position_force, self.cfg.gripper.position_force],
        )

    def _update_gripper_lock(self):
        if self.gripper_locked:
            return
        if self._is_grasp_established():
            self.gripper_locked = True
            self.locked_grip_width = float(self.current_grip_width)

    def _compute_targets(self):
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
        self.home_pos = np.array(
            [self.cfg.target.home_x, self.cfg.target.home_y, self.cfg.target.home_z],
            dtype=np.float32,
        )

    def _refresh_targets_for_step(self):
        object_pos, object_yaw = self._refresh_object_state()

        self.pregrasp_pos[:2] = object_pos[:2]
        self.grasp_pos[:2] = object_pos[:2]
        self.lift_pos[:2] = object_pos[:2]

        self.pregrasp_pos[2] = self.cfg.target.pregrasp_z
        self.grasp_pos[2] = self.cfg.target.grasp_z
        self.lift_pos[2] = self.cfg.target.lift_z
        self.home_pos = np.array(
            [self.cfg.target.home_x, self.cfg.target.home_y, self.cfg.target.home_z],
            dtype=np.float32,
        )

        if self.phase in ("xy_align", "descend", "close"):
            self.target_yaw = self._wrap_angle(object_yaw)
        elif self.phase == "return_home":
            self.target_yaw = float(self.cfg.target.home_yaw)

    def _move_robot_to_pregrasp(self):
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

    def _get_target_pose_for_phase(self):
        if self.phase == "xy_align":
            return self.pregrasp_pos
        if self.phase in ("descend", "close", "hold"):
            return self.grasp_pos
        if self.phase == "lift":
            return self.lift_pos
        return self.home_pos

    def _get_obs_dim(self):
        dim = 3 + 3 + 3 + 3 + 3 + 1 + 1 + 1 + 1 + 1 + 2 + len(self.PHASE_NAMES)
        if self.cfg.include_joint_state:
            dim += 7
        if self.cfg.include_joint_velocity:
            dim += 7
        if self.cfg.include_prev_action:
            dim += 4
        return dim

    def _get_contacts(self):
        left_contact = right_contact = False
        for c in p.getContactPoints(bodyA=self.robot.robot_id, bodyB=self.object_id):
            link_idx = c[3]
            normal_force = float(c[9])
            if (
                link_idx == self.left_finger_link_index
                and normal_force >= self.cfg.gripper.contact_force_threshold
            ):
                left_contact = True
            elif (
                link_idx == self.right_finger_link_index
                and normal_force >= self.cfg.gripper.contact_force_threshold
            ):
                right_contact = True
        return left_contact, right_contact

    def _check_workspace_violated(self):
        ee_pos, _, _ = self._get_ee_pose()
        return not self.robot.is_inside_workspace(ee_pos)

    def _is_xy_aligned(self, ee_yaw: float):
        finger_mid = self._get_finger_midpoint()
        xy_error = float(np.linalg.norm(finger_mid[:2] - self.object_pos[:2]))
        yaw_error = abs(self._wrap_angle(self.target_yaw - ee_yaw))
        return (
            xy_error <= self.cfg.target.xy_phase_threshold
            and yaw_error <= self.cfg.target.yaw_phase_threshold
        )

    def _is_descend_ready(self, ee_yaw: float):
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

    def _is_grasp_established(self):
        left_contact, right_contact = self._get_contacts()
        ee_pos, _, _ = self._get_ee_pose()
        object_pos, _ = self._get_object_pose()
        finger_mid = self._get_finger_midpoint()

        xy_error = float(np.linalg.norm(finger_mid[:2] - object_pos[:2]))
        width_ok = self.current_grip_width <= self.cfg.target.grip_success_width

        # Quan trọng:
        # - Ở close/hold: còn kiểm tra ee_z gần grasp_z
        # - Ở lift/return_home: KHÔNG dùng z_margin theo grasp_z nữa
        #   vì nếu lift lên thì ee_z tất nhiên sẽ rời khỏi grasp_z
        if self.phase in ("lift", "return_home"):
            z_margin_ok = True
        else:
            z_limit = self.cfg.target.grasp_success_max_ee_z_margin
            z_margin_ok = abs(float(ee_pos[2] - self.grasp_pos[2])) <= z_limit

        return (
            left_contact
            and right_contact
            and xy_error <= self.cfg.target.finger_xy_success_dist
            and width_ok
            and z_margin_ok
        )

    def _is_home_ready(self, ee_yaw: float):
        ee_pos, _, _ = self._get_ee_pose()
        xy_error = float(np.linalg.norm(ee_pos[:2] - self.home_pos[:2]))
        z_error = abs(float(ee_pos[2] - self.home_pos[2]))
        yaw_error = abs(self._wrap_angle(self.cfg.target.home_yaw - ee_yaw))
        return (
            xy_error <= self.cfg.target.home_xy_threshold
            and z_error <= self.cfg.target.home_z_threshold
            and yaw_error <= self.cfg.target.home_yaw_threshold
        )

    def _update_phase(self, ee_yaw: float):
        grasp_established = self._is_grasp_established()

        if self.phase == "xy_align":
            if self._is_xy_aligned(ee_yaw):
                self.phase = "descend"

        elif self.phase == "descend":
            if self._is_descend_ready(ee_yaw):
                self.phase = "close"

        elif self.phase == "close":
            ready_for_hold = grasp_established

            # Cho 2B / 2C / 2D một điều kiện mềm hơn để thoát khỏi close-stuck
            if self.cfg.substage in ("2B", "2C", "2D") and self._is_soft_hold_ready():
                ready_for_hold = True

            if ready_for_hold:
                if self.cfg.substage == "2A":
                    return
                self.phase = "hold"
                self.grasp_hold_counter = 0

        elif self.phase == "hold":
            self.grasp_hold_counter = (
                self.grasp_hold_counter + 1 if grasp_established else 0
            )
            if self.grasp_hold_counter >= self.cfg.target.stable_grasp_steps_required:
                if self.cfg.substage == "2B":
                    return
                self.phase = "lift"
                self.lift_hold_counter = 0

        elif self.phase == "lift":
            object_pos, _ = self._get_object_pose()
            lift_ok = object_pos[2] - self.object_initial_z >= self.cfg.target.lift_success_delta_z

            if grasp_established:
                self.lift_lost_counter = 0

                if lift_ok:
                    self.lift_hold_counter += 1
                else:
                    self.lift_hold_counter = 0
            else:
                self.lift_hold_counter = 0
                self.lift_lost_counter += 1

            if self.cfg.substage == "2C":
                return

            if self.lift_hold_counter >= self.cfg.target.stable_lift_steps_required:
                self.phase = "return_home"
                self.home_hold_counter = 0

        elif self.phase == "return_home":
            self.home_hold_counter = (
                self.home_hold_counter + 1
                if (grasp_established and self._is_home_ready(ee_yaw))
                else 0
            )

    def _check_success(self):
        grasp_established = self._is_grasp_established()
        object_pos, _ = self._get_object_pose()
        lift_delta = float(object_pos[2] - self.object_initial_z)

        if self.cfg.substage == "2A":
            return grasp_established

        if self.cfg.substage == "2B":
            return (
                grasp_established
                and self.grasp_hold_counter >= self.cfg.target.stable_grasp_steps_required
            )

        if self.cfg.substage == "2C":
            return (
                grasp_established
                and lift_delta >= self.cfg.target.lift_success_delta_z
                and self.lift_hold_counter >= self.cfg.target.stable_lift_steps_required
            )

        _, _, ee_yaw = self._get_ee_pose()
        return (
            grasp_established
            and lift_delta >= self.cfg.target.lift_success_delta_z
            and self._is_home_ready(ee_yaw)
            and self.home_hold_counter >= self.cfg.target.stable_home_steps_required
        )

    def _check_truncated(self):
        return self.step_count >= int(self.cfg.sim.max_steps)

    def _scale_action(self, action: np.ndarray):
        if self.cfg.action.clip_action:
            action = np.clip(action, -1.0, 1.0)

        scaled = np.array(
            [
                float(action[0]) * float(self.cfg.action.action_scale_xy),
                float(action[1]) * float(self.cfg.action.action_scale_xy),
                float(action[2]) * float(self.cfg.action.action_scale_z),
                float(action[3]) * float(self.cfg.action.action_scale_grip),
            ],
            dtype=np.float32,
        )

        if self.phase == "xy_align":
            scaled[0] *= 0.35
            scaled[1] *= 0.35
            scaled[2] = 0.0
            scaled[3] = 0.0

        elif self.phase == "descend":
            scaled[0] *= 0.15
            scaled[1] *= 0.15
            scaled[2] = min(scaled[2], -0.004)
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
            scaled[3] = 0.0 if self.gripper_locked else -max(0.010, abs(scaled[3]))

        elif self.phase == "lift":
            scaled[0] *= 0.08
            scaled[1] *= 0.08

            if self._is_grasp_established():
                scaled[2] = max(scaled[2], 0.003)
            else:
                scaled[2] = 0.0

            scaled[3] = 0.0 if self.gripper_locked else -max(0.010, abs(scaled[3]))

        elif self.phase == "return_home":
            scaled[0] *= 0.03
            scaled[1] *= 0.03
            scaled[2] *= 0.25
            scaled[3] = -max(0.010, abs(scaled[3]))

        return scaled

    def _apply_action(self, action: np.ndarray):
        alpha = float(np.clip(self.cfg.action.action_smoothing, 0.0, 0.95))
        smoothed = ((1.0 - alpha) * action + alpha * self.prev_action).astype(np.float32)

        dx, dy, dz, dgrip = self._scale_action(smoothed)
        ee_pos, _, _ = self._get_ee_pose()

        if self.phase == "xy_align":
            target_pos = self.pregrasp_pos.copy()
            target_pos[0] += dx
            target_pos[1] += dy

        elif self.phase == "descend":
            target_pos = self.grasp_pos.copy()
            target_pos[0] += dx
            target_pos[1] += dy
            target_pos[2] = ee_pos[2] + dz

        elif self.phase == "return_home":
            # Đi từ từ từ vị trí hiện tại về home, không snap thẳng vào home_pos
            home_delta = self.home_pos - ee_pos

            target_pos = ee_pos.copy()
            target_pos[0] += np.clip(home_delta[0], -abs(dx), abs(dx))
            target_pos[1] += np.clip(home_delta[1], -abs(dy), abs(dy))
            target_pos[2] += np.clip(home_delta[2], -abs(dz), abs(dz))

        else:
            target_pos = ee_pos + np.array([dx, dy, dz], dtype=np.float32)

        target_pos = self.robot.clip_to_workspace(target_pos)
        
        if self.phase == "return_home":
            _, _, ee_yaw_now = self._get_ee_pose()
            yaw_delta = self._wrap_angle(self.cfg.target.home_yaw - ee_yaw_now)
            target_yaw = ee_yaw_now + np.clip(yaw_delta, -0.03, 0.03)
        else:
            target_yaw = self.target_yaw

        self.robot.move_ee_to_with_yaw(
            target_pos=target_pos,
            target_yaw=target_yaw,
            substeps=self.cfg.sim.substeps,
        )

        self.prev_grip_width = float(self.current_grip_width)
        if self.gripper_locked:
            self.current_grip_width = float(self.locked_grip_width)
        else:
            self.current_grip_width = float(
                np.clip(
                    self.current_grip_width + dgrip,
                    self.cfg.gripper.close_width,
                    self.cfg.gripper.open_width,
                )
            )
        self._apply_locked_gripper_width(self.current_grip_width)
        self.robot.step_simulation(2)

    def _get_obs(self):
        ee_pos, _, ee_yaw = self._get_ee_pose()
        object_pos, _ = self._get_object_pose()
        target_pos = self._get_target_pose_for_phase()
        finger_mid = self._get_finger_midpoint()

        delta = target_pos - ee_pos
        object_delta = object_pos - ee_pos
        xy_dist = float(np.linalg.norm(finger_mid[:2] - object_pos[:2]))

        z_ref = (
            self.home_pos[2]
            if self.phase == "return_home"
            else (self.lift_pos[2] if self.phase == "lift" else self.grasp_pos[2])
        )
        z_dist = abs(float(ee_pos[2] - z_ref))

        yaw_error = self._wrap_angle(
            (self.cfg.target.home_yaw if self.phase == "return_home" else self.target_yaw)
            - ee_yaw
        )

        grip_norm = self.current_grip_width / max(self.cfg.gripper.open_width, 1e-6)
        object_lift_delta = float(object_pos[2] - self.object_initial_z)
        left_contact, right_contact = self._get_contacts()

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
            np.array(
                [1.0 if left_contact else 0.0, 1.0 if right_contact else 0.0],
                dtype=np.float32,
            ),
            self._phase_onehot(),
        ]

        if self.cfg.include_joint_state:
            obs_parts.append(self.robot.get_arm_joint_positions().astype(np.float32))
        if self.cfg.include_joint_velocity:
            obs_parts.append(self.robot.get_arm_joint_velocities().astype(np.float32))
        if self.cfg.include_prev_action:
            obs_parts.append(self.prev_action.astype(np.float32))

        return np.concatenate(obs_parts, axis=0).astype(np.float32)
    
    def _is_soft_hold_ready(self) -> bool:
        left_contact, right_contact = self._get_contacts()
        ee_pos, _, _ = self._get_ee_pose()
        object_pos, _ = self._get_object_pose()
        finger_mid = self._get_finger_midpoint()

        xy_error = float(np.linalg.norm(finger_mid[:2] - object_pos[:2]))
        z_error = abs(float(ee_pos[2] - self.grasp_pos[2]))
        width_ok = self.current_grip_width <= self.cfg.target.grip_success_width

        return (
            left_contact
            and right_contact
            and width_ok
            and xy_error <= max(self.cfg.target.finger_xy_success_dist, 0.004)
            and z_error <= 0.012
        )

    def _build_info(self, success: bool, truncated: bool, ee_yaw: float) -> Dict[str, Any]:
        ee_pos, _, _ = self._get_ee_pose()
        object_pos, _ = self._get_object_pose()
        target_pos = self._get_target_pose_for_phase()
        finger_mid = self._get_finger_midpoint()
        left_contact, right_contact = self._get_contacts()

        xy_dist = float(np.linalg.norm(finger_mid[:2] - object_pos[:2]))
        z_ref = (
            self.home_pos[2]
            if self.phase == "return_home"
            else (self.lift_pos[2] if self.phase == "lift" else self.grasp_pos[2])
        )
        z_dist = abs(float(ee_pos[2] - z_ref))
        yaw_error = abs(
            self._wrap_angle(
                (self.cfg.target.home_yaw if self.phase == "return_home" else self.target_yaw)
                - ee_yaw
            )
        )
        dist = float(np.linalg.norm(target_pos - ee_pos))

        return {
            "episode_idx": self.episode_count,
            "step": self.step_count,
            "substage": self.cfg.substage,
            "phase": self.phase,
            "success": bool(success),
            "truncated": bool(truncated),
            "object_pos": object_pos.copy(),
            "target_pos": target_pos.copy(),
            "home_pos": self.home_pos.copy(),
            "dist": dist,
            "xy_dist": xy_dist,
            "z_dist": z_dist,
            "yaw_error": yaw_error,
            "grasp_established": bool(self._is_grasp_established()),
            "object_lift_delta": float(object_pos[2] - self.object_initial_z),
            "grip_width": float(self.current_grip_width),
            "left_contact": bool(left_contact),
            "right_contact": bool(right_contact),
            "stable_pose_steps": self.grasp_hold_counter,
            "lift_hold_steps": self.lift_hold_counter,
            "home_hold_steps": self.home_hold_counter,
            "pregrasp_z": float(self.pregrasp_pos[2]),
            "grasp_z": float(self.grasp_pos[2]),
            "lift_z": float(self.lift_pos[2]),
            "ee_z_world": float(ee_pos[2]),
            "ee_to_grasp_z_error": abs(float(ee_pos[2] - self.grasp_pos[2])),
            "ee_to_home_xyz_error": float(np.linalg.norm(ee_pos - self.home_pos)),
            "ee_to_cube_xy_error": float(np.linalg.norm(ee_pos[:2] - object_pos[:2])),
            "finger_mid_to_cube_xy_error": xy_dist,
            "finger_mid_to_cube_z_error": abs(float(finger_mid[2] - object_pos[2])),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.step_count = 0
        self.episode_count += 1
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_ee_pos = np.zeros(3, dtype=np.float32)
        self.prev_grip_width = float(self.cfg.gripper.initial_width)
        self.current_grip_width = float(self.cfg.gripper.initial_width)
        self.gripper_locked = False
        self.locked_grip_width = float(self.cfg.gripper.initial_width)
        self.phase = "xy_align"
        self.grasp_hold_counter = 0
        self.lift_hold_counter = 0
        self.lift_lost_counter = 0
        self.home_hold_counter = 0

        self._spawn_object()
        self._settle_object(32)
        self._compute_targets()
        self._move_robot_to_pregrasp()
        self._settle_object(8)
        self._compute_targets()

        object_pos, _ = self._get_object_pose()
        self.object_initial_z = float(object_pos[2])
        self.object_lift_delta = 0.0

        ee_pos, _, ee_yaw = self._get_ee_pose()
        self.prev_ee_pos = ee_pos.astype(np.float32)

        return self._get_obs(), self._build_info(False, False, ee_yaw)

    def step(self, action: np.ndarray):
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
        self._update_gripper_lock()

        curr_object_pos, _ = self._get_object_pose()
        self.object_lift_delta = float(curr_object_pos[2] - self.object_initial_z)

        object_dropped = (
            self.phase in ("hold", "lift", "return_home")
            and self.object_lift_delta < -0.002
        )

        success = self._check_success()
        truncated = self._check_truncated()
        workspace_violated = self._check_workspace_violated()
        left_contact, right_contact = self._get_contacts()
        grasp_established = self._is_grasp_established()
        failed_lift_abort = (self.phase == "lift" and self.lift_lost_counter >= 3)

        prev_lift_delta = float(prev_object_pos[2] - self.object_initial_z)
        lift_progress = self.object_lift_delta - prev_lift_delta

        reward, reward_info = self.rewarder.compute(
            prev_ee_pos=self.prev_ee_pos,
            curr_ee_pos=curr_ee_pos.astype(np.float32),
            target_xy=self.object_pos[:2].astype(np.float32),
            grasp_z=float(self.grasp_pos[2]),
            lift_z=float(self.lift_pos[2]),
            home_pos=self.home_pos.astype(np.float32),
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
            hold_counter=max(
                self.grasp_hold_counter,
                self.lift_hold_counter,
                self.home_hold_counter,
            ),
            lift_progress=float(lift_progress),
            object_lift_delta=self.object_lift_delta,
            object_dropped=object_dropped,
        )

        if failed_lift_abort:
            reward -= 4.0

        self.prev_action = action.copy()
        info = self._build_info(success, truncated, curr_ee_yaw)
        info.update(reward_info)

        terminated = bool(success or failed_lift_abort)
        info["failed_lift_abort"] = bool(failed_lift_abort)

        return self._get_obs(), float(reward), terminated, bool(truncated), info

    def close(self):
        try:
            if self.object_id is not None:
                p.removeBody(self.object_id)
        except Exception:
            pass

        try:
            p.disconnect()
        except Exception:
            pass