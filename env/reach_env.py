from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from config.reach_env_config import ReachEnvConfig, build_reach_config
from env.panda_controller import PandaController
from env.reward_reach import RewardReach


class ReachEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: Optional[ReachEnvConfig] = None):
        super().__init__()
        self.cfg: ReachEnvConfig = cfg if cfg is not None else build_reach_config("1A")

        self.client_id: Optional[int] = None
        self.robot: Optional[PandaController] = None
        self.plane_id: Optional[int] = None
        self.table_id: Optional[int] = None
        self.object_id: Optional[int] = None

        self.step_count = 0
        self.episode_count = 0

        self.object_pos = np.zeros(3, dtype=np.float32)
        self.object_yaw = 0.0

        self.target_pos = np.zeros(3, dtype=np.float32)
        self.target_yaw = 0.0

        self.prev_ee_pos = np.zeros(3, dtype=np.float32)
        self.prev_ee_yaw = 0.0
        self.prev_action = np.zeros(4, dtype=np.float32)

        self.last_phase = "far"
        self.stable_pose_steps = 0
        self.phase_state = "far"
        self.xy_locked = False
        self.z_locked = False
        self.curriculum_level = "hard"

        self._base_1e_reset_xy_noise = float(self.cfg.reset_cfg.near_target_xy_noise)
        self._base_1e_reset_z_noise = float(self.cfg.reset_cfg.near_target_z_noise)
        self._base_1e_yaw_min_error = float(self.cfg.reset_cfg.near_target_yaw_min_error)
        self._base_1e_yaw_max_error = float(self.cfg.reset_cfg.near_target_yaw_max_error)
        self._base_1e_action_scale_yaw = float(self.cfg.action.action_scale_yaw)
        self._base_1e_reward_yaw_progress = float(self.cfg.reward.w_yaw_progress)
        self._base_1e_timeout_penalty = float(self.cfg.reward.w_timeout_penalty)

        self.rewarder = RewardReach(self.cfg)

        self._connect()
        self._build_static_world()
        self._build_robot()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        obs_dim = self._get_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _is_basic_reach_stage(self) -> bool:
        return self.cfg.substage in ("1A", "1B")

    def _is_precision_stage(self) -> bool:
        return self.cfg.substage in ("1C", "1D")

    def _is_yaw_refine_stage(self) -> bool:
        return self.cfg.substage == "1E"

    def _is_orientation_stage(self) -> bool:
        return self.cfg.substage in ("1E", "1F")

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

    def _remove_object(self) -> None:
        if self.object_id is not None:
            try:
                p.removeBody(self.object_id)
            except Exception:
                pass
            self.object_id = None

    def _sample_object_pose(self) -> Tuple[np.ndarray, float]:
        spawn_cfg = self.cfg.spawn

        if spawn_cfg.spawn_mode == "fixed":
            pos = np.array(spawn_cfg.fixed_pose_xyz, dtype=np.float32)
        elif spawn_cfg.spawn_mode == "random":
            pos = np.array(
                [
                    self.np_random.uniform(spawn_cfg.roi_x[0], spawn_cfg.roi_x[1]),
                    self.np_random.uniform(spawn_cfg.roi_y[0], spawn_cfg.roi_y[1]),
                    spawn_cfg.object_z,
                ],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"spawn_mode không hợp lệ: {spawn_cfg.spawn_mode}")

        yaw = (
            self.np_random.uniform(spawn_cfg.yaw_range[0], spawn_cfg.yaw_range[1])
            if spawn_cfg.random_yaw
            else 0.0
        )

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

        half_extents = [0.02, 0.02, 0.02]
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=rgba,
        )
        orn = p.getQuaternionFromEuler([0.0, 0.0, self.object_yaw])

        self.object_id = p.createMultiBody(
            baseMass=0.2,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=self.object_pos.tolist(),
            baseOrientation=orn,
        )

    def _wrap_angle(self, angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def _get_ee_yaw(self) -> float:
        assert self.robot is not None
        return self.robot.get_ee_yaw()

    def _compute_target_pos(self) -> np.ndarray:
        if self.cfg.target.use_pregrasp_target:
            return np.array(
                [self.object_pos[0], self.object_pos[1], self.cfg.target.pregrasp_z],
                dtype=np.float32,
            )

        if self.cfg.target.use_hover_target:
            return np.array(
                [self.object_pos[0], self.object_pos[1], self.cfg.target.z_hover],
                dtype=np.float32,
            )

        return self.object_pos.copy()

    def _compute_target_yaw(self) -> float:
        if self.cfg.target.use_orientation_target:
            return self._wrap_angle(self.object_yaw + float(self.cfg.target.yaw_offset))
        return 0.0

    def _move_robot_near_target_for_yaw_refine(self) -> None:
        assert self.robot is not None

        rcfg = self.cfg.reset_cfg

        x_noise = self.np_random.uniform(-rcfg.near_target_xy_noise, rcfg.near_target_xy_noise)
        y_noise = self.np_random.uniform(-rcfg.near_target_xy_noise, rcfg.near_target_xy_noise)
        z_noise = self.np_random.uniform(-rcfg.near_target_z_noise, rcfg.near_target_z_noise)

        start_pos = np.array(
            [
                self.target_pos[0] + x_noise,
                self.target_pos[1] + y_noise,
                self.target_pos[2] + z_noise,
            ],
            dtype=np.float32,
        )

        sign = -1.0 if self.np_random.random() < 0.5 else 1.0
        yaw_mag = self.np_random.uniform(
            rcfg.near_target_yaw_min_error,
            rcfg.near_target_yaw_max_error,
        )
        start_yaw = self._wrap_angle(self.target_yaw + sign * yaw_mag)

        self.robot.move_ee_to_with_yaw(
            target_pos=start_pos,
            target_yaw=start_yaw,
            substeps=24,
        )

    def _get_obs_dim(self) -> int:
        dim = 12
        dim += 4

        if self.cfg.include_joint_state:
            dim += 7
        if self.cfg.include_joint_velocity:
            dim += 7
        if self.cfg.include_prev_action:
            dim += 4

        return dim

    def _apply_obs_noise(self, arr: np.ndarray) -> np.ndarray:
        if not self.cfg.noise.enable_obs_noise:
            return arr

        noise = self.np_random.normal(
            0.0,
            self.cfg.noise.pos_noise_std,
            size=arr.shape,
        ).astype(np.float32)
        return arr + noise

    def _get_obs(self) -> np.ndarray:
        assert self.robot is not None

        ee_pos, _ = self.robot.get_ee_pose()
        joint_q = self.robot.get_arm_joint_positions()
        joint_dq = self.robot.get_arm_joint_velocities()

        delta = self.target_pos - ee_pos
        dist = np.linalg.norm(delta)
        xy_dist = np.linalg.norm(delta[:2])
        z_dist = abs(float(delta[2]))

        ee_yaw = self._get_ee_yaw()
        yaw_error = self._wrap_angle(self.target_yaw - ee_yaw)

        obs_parts = [
            ee_pos.astype(np.float32),
            self.target_pos.astype(np.float32),
            delta.astype(np.float32),
            np.array([dist], dtype=np.float32),
            np.array([xy_dist], dtype=np.float32),
            np.array([z_dist], dtype=np.float32),
            np.array([ee_yaw], dtype=np.float32),
            np.array([self.target_yaw], dtype=np.float32),
            np.array([yaw_error], dtype=np.float32),
            np.array([abs(yaw_error)], dtype=np.float32),
        ]

        if self.cfg.include_joint_state:
            obs_parts.append(joint_q.astype(np.float32))
        if self.cfg.include_joint_velocity:
            obs_parts.append(joint_dq.astype(np.float32))
        if self.cfg.include_prev_action:
            obs_parts.append(self.prev_action.astype(np.float32))

        obs = np.concatenate(obs_parts, axis=0).astype(np.float32)
        return self._apply_obs_noise(obs)

    def _compute_dist_metrics(self) -> Tuple[float, float, float]:
        assert self.robot is not None
        ee_pos, _ = self.robot.get_ee_pose()
        delta = self.target_pos - ee_pos

        dist = float(np.linalg.norm(delta))
        xy_dist = float(np.linalg.norm(delta[:2]))
        z_dist = abs(float(delta[2]))
        return dist, xy_dist, z_dist

    def _reset_phase_memory(self) -> None:
        self.phase_state = "far"
        self.xy_locked = False
        self.z_locked = False

    def _resolve_1e_curriculum_level(self) -> str:
        if not self._is_yaw_refine_stage() or not self.cfg.curriculum.enabled:
            return "hard"

        forced = str(self.cfg.curriculum.force_level).strip().lower()
        if forced in ("easy", "medium", "hard"):
            return forced

        if self.episode_count < self.cfg.curriculum.easy_until_episode:
            return "easy"
        if self.episode_count < self.cfg.curriculum.medium_until_episode:
            return "medium"
        return "hard"

    def _apply_1e_curriculum(self) -> None:
        if not self._is_yaw_refine_stage():
            self.curriculum_level = "hard"
            return

        level = self._resolve_1e_curriculum_level()
        self.curriculum_level = level
        level_cfg = getattr(self.cfg.curriculum, level)

        self.cfg.reset_cfg.near_target_xy_noise = float(level_cfg.xy_noise)
        self.cfg.reset_cfg.near_target_z_noise = float(level_cfg.z_noise)
        self.cfg.reset_cfg.near_target_yaw_min_error = float(level_cfg.yaw_min_error)
        self.cfg.reset_cfg.near_target_yaw_max_error = float(level_cfg.yaw_max_error)

        self.cfg.action.action_scale_yaw = self._base_1e_action_scale_yaw * float(level_cfg.action_scale_yaw_mult)
        self.cfg.reward.w_yaw_progress = self._base_1e_reward_yaw_progress * float(level_cfg.yaw_progress_mult)
        self.cfg.reward.w_timeout_penalty = self._base_1e_timeout_penalty * float(level_cfg.timeout_penalty_mult)

    def _update_orientation_phase_state(
        self,
        dist: float,
        xy_dist: float,
        z_dist: float,
        yaw_error: float,
    ) -> str:
        if not self._is_orientation_stage():
            return self.phase_state

        tcfg = self.cfg.target
        acfg = self.cfg.action

        # Phase 1: phải khóa XY trước
        if not self.xy_locked:
            if xy_dist <= tcfg.xy_align_threshold:
                self.xy_locked = True
            else:
                self.phase_state = "xy_lock" if dist <= acfg.near_dist else "approach"
                self.z_locked = False
                return self.phase_state

        # Phase 2: sau khi XY ổn thì mới settle Z
        if self.xy_locked and not self.z_locked:
            if z_dist <= tcfg.z_success_dist * 1.35:
                self.z_locked = True
            else:
                self.phase_state = "z_settle"
                return self.phase_state

        # Phase 3: XY + Z đã ổn thì mới align yaw
        if self.xy_locked and self.z_locked:
            pose_good_for_hold = (
                xy_dist <= tcfg.xy_success_dist * 1.15
                and z_dist <= tcfg.z_success_dist * 1.15
                and yaw_error <= tcfg.yaw_success_dist * 1.25
            )
            if pose_good_for_hold:
                self.phase_state = "stable_hold"
            else:
                self.phase_state = "yaw_align"

        return self.phase_state

    def _check_success(self) -> bool:
        dist, xy_dist, z_dist = self._compute_dist_metrics()

        if self._is_orientation_stage():
            ee_yaw = self._get_ee_yaw()
            yaw_error = abs(self._wrap_angle(self.target_yaw - ee_yaw))

            is_pose_good = (
                xy_dist < float(self.cfg.target.xy_success_dist)
                and z_dist < float(self.cfg.target.z_success_dist)
                and yaw_error < float(self.cfg.target.yaw_success_dist)
            )

            if is_pose_good:
                self.stable_pose_steps += 1
            else:
                self.stable_pose_steps = 0

            return self.stable_pose_steps >= int(self.cfg.target.stable_steps_required)

        if self._is_precision_stage():
            is_pose_good = (
                xy_dist < float(self.cfg.target.xy_success_dist)
                and z_dist < float(self.cfg.target.z_success_dist)
            )

            if is_pose_good:
                self.stable_pose_steps += 1
            else:
                self.stable_pose_steps = 0

            return self.stable_pose_steps >= int(self.cfg.target.stable_steps_required)

        return dist < float(self.cfg.target.success_dist)

    def _check_truncated(self) -> bool:
        return self.step_count >= int(self.cfg.sim.max_steps)

    def _check_workspace_violated(self) -> bool:
        assert self.robot is not None
        ee_pos, _ = self.robot.get_ee_pose()
        return not self.robot.is_inside_workspace(ee_pos)

    def _is_xy_aligned(self) -> bool:
        _, xy_dist, _ = self._compute_dist_metrics()
        return xy_dist <= float(self.cfg.target.xy_align_threshold)

    def _get_phase(self, dist: float, xy_dist: float, z_dist: float) -> str:
        acfg = self.cfg.action
        tcfg = self.cfg.target

        if self._is_orientation_stage():
            ee_yaw = self._get_ee_yaw()
            yaw_error = abs(self._wrap_angle(self.target_yaw - ee_yaw))
            return self._update_orientation_phase_state(dist, xy_dist, z_dist, yaw_error)

        if self._is_precision_stage():
            if (
                xy_dist <= tcfg.xy_success_dist * 1.5
                and z_dist <= tcfg.z_success_dist * 1.5
            ):
                return "stable_hold"

            if xy_dist <= tcfg.xy_align_threshold:
                return "z_settle"

            if dist <= acfg.near_dist:
                return "xy_lock"

            return "approach"

        if dist <= tcfg.success_dist * 1.25:
            return "settle"
        if xy_dist <= tcfg.xy_align_threshold and z_dist > acfg.descend_z_gate:
            return "descend"
        if xy_dist <= tcfg.xy_align_threshold:
            return "near_aligned"
        if dist <= acfg.near_dist:
            return "align"
        return "far"

    def _scale_action(self, action: np.ndarray) -> Tuple[float, float, float, float, str]:
        if self.cfg.action.clip_action:
            action = np.clip(action, -1.0, 1.0)

        dist, xy_dist, z_dist = self._compute_dist_metrics()
        phase = self._get_phase(dist, xy_dist, z_dist)

        substage = self.cfg.substage
        xy_scale = float(self.cfg.action.action_scale_xy)
        z_scale = float(self.cfg.action.action_scale_z)
        yaw_scale = float(self.cfg.action.action_scale_yaw)

        if substage == "1A":
            if phase == "align":
                xy_scale *= 0.90
                z_scale *= 0.90
            elif phase == "descend":
                xy_scale *= 0.55
                z_scale *= 1.10
            elif phase == "near_aligned":
                xy_scale *= 0.50
                z_scale *= 0.85
            elif phase == "settle":
                xy_scale *= 0.40
                z_scale *= 0.60

        elif substage == "1B":
            if phase == "align":
                xy_scale *= 0.80
                z_scale *= 0.85
            elif phase == "descend":
                xy_scale *= 0.50
                z_scale *= 0.95
            elif phase == "near_aligned":
                xy_scale *= 0.42
                z_scale *= 0.72
            elif phase == "settle":
                xy_scale *= 0.32
                z_scale *= 0.50

        elif substage in ("1C", "1D"):
            if phase == "xy_lock":
                xy_scale *= 0.70
                z_scale *= 0.75
                yaw_scale *= 0.60
            elif phase == "z_settle":
                xy_scale *= 0.42
                z_scale *= 0.55
                yaw_scale *= 0.55
            elif phase == "stable_hold":
                xy_scale *= 0.20
                z_scale *= 0.25
                yaw_scale *= 0.35
            else:
                xy_scale *= 0.90
                z_scale *= 0.95
                yaw_scale *= 0.45

        elif substage == "1E":
            if phase == "approach":
                xy_scale *= 1.00
                z_scale *= 0.70
                yaw_scale *= 0.00
            elif phase == "xy_lock":
                xy_scale *= 0.72
                z_scale *= 0.45
                yaw_scale *= 0.00
            elif phase == "z_settle":
                xy_scale *= 0.18
                z_scale *= 0.70
                yaw_scale *= 0.00
            elif phase == "yaw_align":
                xy_scale *= 0.06
                z_scale *= 0.06
                yaw_scale *= 1.00
            elif phase == "stable_hold":
                xy_scale *= 0.03
                z_scale *= 0.03
                yaw_scale *= 0.05
            else:
                xy_scale *= 0.70
                z_scale *= 0.70
                yaw_scale *= 0.00

        elif substage == "1F":
            if phase == "xy_lock":
                xy_scale *= 0.70
                z_scale *= 0.75
                yaw_scale *= 0.55
            elif phase == "z_settle":
                xy_scale *= 0.40
                z_scale *= 0.50
                yaw_scale *= 0.60
            elif phase == "yaw_align":
                xy_scale *= 0.22
                z_scale *= 0.25
                yaw_scale *= 0.90
            elif phase == "stable_hold":
                xy_scale *= 0.16
                z_scale *= 0.18
                yaw_scale *= 0.25
            else:
                xy_scale *= 0.90
                z_scale *= 0.95
                yaw_scale *= 0.45

        dx = float(action[0]) * xy_scale
        dy = float(action[1]) * xy_scale
        dz = float(action[2]) * z_scale
        dyaw = float(action[3]) * yaw_scale
        return dx, dy, dz, dyaw, phase

    def _apply_action(self, action: np.ndarray) -> None:
        assert self.robot is not None

        alpha = float(self.cfg.action.action_smoothing)
        alpha = min(max(alpha, 0.0), 0.95)

        smoothed = (1.0 - alpha) * action + alpha * self.prev_action
        smoothed = smoothed.astype(np.float32)

        dx, dy, dz, dyaw, phase = self._scale_action(smoothed)
        self.last_phase = phase

        self.robot.move_ee_delta(
            dx=dx,
            dy=dy,
            dz=dz,
            dyaw=dyaw,
            substeps=self.cfg.sim.substeps,
        )

    def _debug_draw(self) -> None:
        if not self.cfg.sim.use_gui:
            return
        if not self.cfg.debug.draw_target and not self.cfg.debug.draw_ee_path:
            return

        assert self.robot is not None
        ee_pos, _ = self.robot.get_ee_pose()

        text = f"target | phase={self.last_phase} | stable={self.stable_pose_steps}"
        yaw_error = abs(self._wrap_angle(self.target_yaw - self._get_ee_yaw()))
        text += f" | yaw_err={yaw_error:.3f}"

        if self.cfg.debug.draw_target:
            p.addUserDebugText(
                text,
                self.target_pos.tolist(),
                [1, 0, 0],
                lifeTime=0.05,
            )

            if self._is_orientation_stage():
                self.robot.draw_target_pose(
                    target_pos=self.target_pos,
                    target_yaw=self.target_yaw,
                    axis_len=0.05,
                    life_time=0.05,
                )

        if self.cfg.debug.draw_ee_path:
            p.addUserDebugLine(
                ee_pos.tolist(),
                self.target_pos.tolist(),
                [0, 1, 0],
                lineWidth=2.0,
                lifeTime=0.05,
            )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        assert self.robot is not None

        self.step_count = 0
        self.episode_count += 1
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_ee_pos = np.zeros(3, dtype=np.float32)
        self.prev_ee_yaw = 0.0
        self.last_phase = "far"
        self.stable_pose_steps = 0
        self.phase_state = "far"
        self.xy_locked = False
        self.z_locked = False
        self.rewarder.reset()
        self._apply_1e_curriculum()

        self.robot.reset_home(open_gripper=True)
        self._spawn_object()

        self.target_pos = self._compute_target_pos()
        self.target_yaw = self._compute_target_yaw()

        # 1E bây giờ phải học full sequence từ home
        # nên không còn move robot đến near target nữa

        ee_pos, _ = self.robot.get_ee_pose()
        self.prev_ee_pos = ee_pos.astype(np.float32)
        self.prev_ee_yaw = self._get_ee_yaw()

        dist, xy_dist, z_dist = self._compute_dist_metrics()
        yaw_error = abs(self._wrap_angle(self.target_yaw - self.prev_ee_yaw))
        self.last_phase = self._update_orientation_phase_state(dist, xy_dist, z_dist, yaw_error) if self._is_orientation_stage() else self.last_phase

        obs = self._get_obs()

        info = {
            "episode_idx": self.episode_count,
            "step": self.step_count,
            "object_pos": self.object_pos.copy(),
            "object_yaw": float(self.object_yaw),
            "target_pos": self.target_pos.copy(),
            "target_yaw": float(self.target_yaw),
            "dist": dist,
            "xy_dist": xy_dist,
            "z_dist": z_dist,
            "success": False,
            "truncated": False,
            "xy_aligned": xy_dist <= float(self.cfg.target.xy_align_threshold),
            "phase": self.phase_state if self._is_orientation_stage() else self._get_phase(dist, xy_dist, z_dist),
            "substage": self.cfg.substage,
            "stable_pose_steps": self.stable_pose_steps,
            "yaw_error": abs(self._wrap_angle(self.target_yaw - self.prev_ee_yaw)),
            "curriculum_level": self.curriculum_level,
        }
        return obs, info

    def step(self, action: np.ndarray):
        assert self.robot is not None

        action = np.asarray(action, dtype=np.float32).reshape(4,)

        prev_ee_pos, _ = self.robot.get_ee_pose()
        prev_ee_yaw = self._get_ee_yaw()

        self.prev_ee_pos = prev_ee_pos.astype(np.float32)
        self.prev_ee_yaw = prev_ee_yaw

        self._apply_action(action)
        self.step_count += 1

        curr_ee_pos, _ = self.robot.get_ee_pose()
        curr_ee_yaw = self._get_ee_yaw()

        curr_dist, curr_xy_dist, curr_z_dist = self._compute_dist_metrics()
        curr_yaw_error = abs(self._wrap_angle(self.target_yaw - curr_ee_yaw))
        if self._is_orientation_stage():
            self.last_phase = self._update_orientation_phase_state(curr_dist, curr_xy_dist, curr_z_dist, curr_yaw_error)

        success = self._check_success()
        truncated = self._check_truncated()
        workspace_violated = self._check_workspace_violated()
        xy_aligned = self._is_xy_aligned()

        reward, reward_info = self.rewarder.compute(
            prev_ee_pos=self.prev_ee_pos,
            curr_ee_pos=curr_ee_pos.astype(np.float32),
            target_pos=self.target_pos.astype(np.float32),
            action=action,
            success=success,
            truncated=truncated,
            xy_aligned=xy_aligned,
            workspace_violated=workspace_violated,
            prev_action=self.prev_action.copy(),
            prev_ee_yaw=prev_ee_yaw,
            curr_ee_yaw=curr_ee_yaw,
            target_yaw=self.target_yaw,
            phase=self.last_phase,
            xy_locked=self.xy_locked,
            z_locked=self.z_locked,
            curriculum_level=self.curriculum_level,
        )

        self.prev_action = action.copy()
        obs = self._get_obs()
        terminated = bool(success)

        dist, xy_dist, z_dist = self._compute_dist_metrics()

        info: Dict[str, Any] = {
            "episode_idx": self.episode_count,
            "step": self.step_count,
            "object_pos": self.object_pos.copy(),
            "object_yaw": float(self.object_yaw),
            "target_pos": self.target_pos.copy(),
            "target_yaw": float(self.target_yaw),
            "dist": dist,
            "xy_dist": xy_dist,
            "z_dist": z_dist,
            "success": success,
            "truncated": truncated,
            "xy_aligned": xy_aligned,
            "workspace_violated": workspace_violated,
            "phase": self.last_phase,
            "substage": self.cfg.substage,
            "stable_pose_steps": self.stable_pose_steps,
            "yaw_error": abs(self._wrap_angle(self.target_yaw - curr_ee_yaw)),
            "curriculum_level": self.curriculum_level,
        }
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