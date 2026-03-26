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

        self.prev_ee_pos = np.zeros(3, dtype=np.float32)
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.last_phase = "far"

        self.rewarder = RewardReach(self.cfg)

        self._connect()
        self._build_static_world()
        self._build_robot()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
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
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=table_half, rgbaColor=[0.75, 0.75, 0.75, 1.0])
        self.table_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=table_pos)

    def _build_robot(self) -> None:
        self.robot = PandaController(client_id=self.client_id, use_gui=self.cfg.sim.use_gui, workspace=self.cfg.workspace)

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
            pos = np.array([
                np.random.uniform(spawn_cfg.roi_x[0], spawn_cfg.roi_x[1]),
                np.random.uniform(spawn_cfg.roi_y[0], spawn_cfg.roi_y[1]),
                spawn_cfg.object_z,
            ], dtype=np.float32)
        else:
            raise ValueError(f"spawn_mode không hợp lệ: {spawn_cfg.spawn_mode}")
        yaw = np.random.uniform(spawn_cfg.yaw_range[0], spawn_cfg.yaw_range[1]) if spawn_cfg.random_yaw else 0.0
        return pos, float(yaw)

    def _spawn_object(self) -> None:
        self._remove_object()
        self.object_pos, self.object_yaw = self._sample_object_pose()
        color_name = np.random.choice(self.cfg.spawn.object_colors)
        color_map = {
            "red": [1.0, 0.2, 0.2, 1.0],
            "green": [0.2, 1.0, 0.2, 1.0],
            "blue": [0.2, 0.4, 1.0, 1.0],
            "yellow": [1.0, 1.0, 0.2, 1.0],
        }
        rgba = color_map.get(color_name, [1.0, 0.2, 0.2, 1.0])
        half_extents = [0.02, 0.02, 0.02]
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
        orn = p.getQuaternionFromEuler([0.0, 0.0, self.object_yaw])
        self.object_id = p.createMultiBody(baseMass=0.2, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=self.object_pos.tolist(), baseOrientation=orn)

    def _compute_hover_target(self) -> np.ndarray:
        if self.cfg.target.use_hover_target:
            return np.array([self.object_pos[0], self.object_pos[1], self.cfg.target.z_hover], dtype=np.float32)
        return self.object_pos.copy()

    def _get_obs_dim(self) -> int:
        dim = 12
        if self.cfg.include_joint_state:
            dim += 7
        if self.cfg.include_joint_velocity:
            dim += 7
        if self.cfg.include_prev_action:
            dim += 3
        return dim

    def _apply_obs_noise(self, arr: np.ndarray) -> np.ndarray:
        if not self.cfg.noise.enable_obs_noise:
            return arr
        noise = np.random.normal(0.0, self.cfg.noise.pos_noise_std, size=arr.shape).astype(np.float32)
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
        obs_parts = [
            ee_pos.astype(np.float32),
            self.target_pos.astype(np.float32),
            delta.astype(np.float32),
            np.array([dist], dtype=np.float32),
            np.array([xy_dist], dtype=np.float32),
            np.array([z_dist], dtype=np.float32),
        ]
        if self.cfg.include_joint_state:
            obs_parts.append(joint_q.astype(np.float32))
        if self.cfg.include_joint_velocity:
            obs_parts.append(joint_dq.astype(np.float32))
        if self.cfg.include_prev_action:
            obs_parts.append(self.prev_action.astype(np.float32))
        return self._apply_obs_noise(np.concatenate(obs_parts, axis=0).astype(np.float32))

    def _compute_dist_metrics(self) -> Tuple[float, float, float]:
        assert self.robot is not None
        ee_pos, _ = self.robot.get_ee_pose()
        delta = self.target_pos - ee_pos
        return float(np.linalg.norm(delta)), float(np.linalg.norm(delta[:2])), abs(float(delta[2]))

    def _check_success(self) -> bool:
        dist, _, _ = self._compute_dist_metrics()
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
        if dist <= tcfg.success_dist * 1.25:
            return "settle"
        if xy_dist <= tcfg.xy_align_threshold and z_dist > acfg.descend_z_gate:
            return "descend"
        if xy_dist <= tcfg.xy_align_threshold:
            return "near_aligned"
        if dist <= acfg.near_dist:
            return "align"
        return "far"

    def _scale_action(self, action: np.ndarray) -> Tuple[float, float, float, str]:
        if self.cfg.action.clip_action:
            action = np.clip(action, -1.0, 1.0)

        dist, xy_dist, z_dist = self._compute_dist_metrics()
        phase = self._get_phase(dist, xy_dist, z_dist)
        substage = self.cfg.substage

        xy_scale = float(self.cfg.action.action_scale_xy)
        z_scale = float(self.cfg.action.action_scale_z)

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
        else:  # 1C
            if phase == "align":
                xy_scale *= 0.75
                z_scale *= 0.80
            elif phase == "descend":
                xy_scale *= 0.42
                z_scale *= 0.70
            elif phase == "near_aligned":
                xy_scale *= 0.35
                z_scale *= 0.55
            elif phase == "settle":
                xy_scale *= 0.25
                z_scale *= 0.40

        dx = float(action[0]) * xy_scale
        dy = float(action[1]) * xy_scale
        dz = float(action[2]) * z_scale
        return dx, dy, dz, phase

    def _apply_action(self, action: np.ndarray) -> None:
        assert self.robot is not None
        alpha = float(self.cfg.action.action_smoothing)
        alpha = min(max(alpha, 0.0), 0.95)
        smoothed = (1.0 - alpha) * action + alpha * self.prev_action
        smoothed = smoothed.astype(np.float32)
        dx, dy, dz, phase = self._scale_action(smoothed)
        self.last_phase = phase
        self.robot.move_ee_delta(dx=dx, dy=dy, dz=dz, substeps=self.cfg.sim.substeps)

    def _debug_draw(self) -> None:
        if not self.cfg.sim.use_gui:
            return
        if not self.cfg.debug.draw_target and not self.cfg.debug.draw_ee_path:
            return
        assert self.robot is not None
        ee_pos, _ = self.robot.get_ee_pose()
        if self.cfg.debug.draw_target:
            p.addUserDebugText("hover_target", self.target_pos.tolist(), [1, 0, 0], lifeTime=0.05)
        if self.cfg.debug.draw_ee_path:
            p.addUserDebugLine(ee_pos.tolist(), self.target_pos.tolist(), [0, 1, 0], lineWidth=2.0, lifeTime=0.05)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        assert self.robot is not None
        self.step_count = 0
        self.episode_count += 1
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.last_phase = "far"
        self.rewarder.reset()

        self.robot.reset_home(open_gripper=True)
        self._spawn_object()
        self.target_pos = self._compute_hover_target()
        ee_pos, _ = self.robot.get_ee_pose()
        self.prev_ee_pos = ee_pos.astype(np.float32)
        obs = self._get_obs()
        dist, xy_dist, z_dist = self._compute_dist_metrics()
        info = {
            "episode_idx": self.episode_count,
            "step": self.step_count,
            "object_pos": self.object_pos.copy(),
            "target_pos": self.target_pos.copy(),
            "dist": dist,
            "xy_dist": xy_dist,
            "z_dist": z_dist,
            "success": False,
            "truncated": False,
            "xy_aligned": xy_dist <= float(self.cfg.target.xy_align_threshold),
            "phase": self._get_phase(dist, xy_dist, z_dist),
            "substage": self.cfg.substage,
        }
        return obs, info

    def step(self, action: np.ndarray):
        assert self.robot is not None
        action = np.asarray(action, dtype=np.float32).reshape(3,)
        prev_ee_pos, _ = self.robot.get_ee_pose()
        self.prev_ee_pos = prev_ee_pos.astype(np.float32)

        self._apply_action(action)
        self.step_count += 1
        curr_ee_pos, _ = self.robot.get_ee_pose()

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
        )

        self.prev_action = action.copy()
        obs = self._get_obs()
        terminated = bool(success)
        dist, xy_dist, z_dist = self._compute_dist_metrics()

        info: Dict[str, Any] = {
            "episode_idx": self.episode_count,
            "step": self.step_count,
            "object_pos": self.object_pos.copy(),
            "target_pos": self.target_pos.copy(),
            "dist": dist,
            "xy_dist": xy_dist,
            "z_dist": z_dist,
            "success": success,
            "truncated": truncated,
            "xy_aligned": xy_aligned,
            "workspace_violated": workspace_violated,
            "phase": self.last_phase,
            "substage": self.cfg.substage,
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
