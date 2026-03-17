import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

from config.place_env_config import PlaceEnvConfig
from env.camera import Camera
from env.panda_controller import PandaController
from env.reward_place import RewardModulePlace
from env.utils_debug import DebugDraw


class PlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode="rgb_array", use_gui=False, start_held=True, substage="3A"):
        super().__init__()

        self.render_mode = render_mode
        self.config = PlaceEnvConfig()
        self.use_gui = bool(use_gui)
        self.start_held = bool(start_held)
        self.substage = str(substage).upper()

        self.x_range = self.config.X_RANGE
        self.y_range = self.config.Y_RANGE
        self.z_range = self.config.Z_RANGE

        self.cid = p.connect(p.GUI) if self.use_gui else p.connect(p.DIRECT)
        if self.use_gui:
            print("Connected GUI:", self.cid)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(1.0 / self.config.PHYSICS_HZ, physicsClientId=self.cid)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 3 * self.config.FRAME_STACK),
            dtype=np.uint8,
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.camera = Camera(
            img_size=self.config.IMG_SIZE,
            render_w=self.config.RENDER_W,
            render_h=self.config.RENDER_H,
            physics_client_id=self.cid,
            use_gui=self.use_gui,
            use_crop=self.config.USE_CROP,
            crop_box=self.config.CROP_BOX,
        )
        self.debug = DebugDraw(physics_client_id=self.cid)

        self.target_pos = np.array(self.config.TARGET_POS_3A, dtype=np.float32)
        params = self.config.get_success_params(self.substage)

        self.rewarder = RewardModulePlace(
            ee_link=11,
            target_pos=self.target_pos,
            time_penalty=self.config.TIME_PENALTY,
            dist_weight=self.config.DIST_WEIGHT,
            release_bonus=self.config.RELEASE_BONUS,
            success_bonus=self.config.SUCCESS_BONUS,
            success_dist=params["success_dist"],
            table_z=params["table_z"],
            delta_clip=self.config.DELTA_CLIP,
            z_release_max=params["z_release_max"],
            high_release_penalty=self.config.HIGH_RELEASE_PENALTY,
            physics_client_id=self.cid,
        )

        self.frame_buffer = None
        self.step_count = 0
        self.last_grip = 0.0

        self.robot = None
        self.ctrl = None
        self.object_id = None
        self.target_vis_id = None
        self.table_id = None
        self.table_top_z = 0.06

        self.hold_cid = None
        self.release_cooldown = 0

        # Pose tay cong đẹp như ảnh tham chiếu
        self.pretty_q = [0.00, -0.72, 0.00, -2.15, 0.00, 1.52, 0.78]

        # độ cao hover / place
        self.hover_h = 0.10
        self.pre_drop_h = 0.040
        self.drop_h = 0.022

        self._setup_scene()

    # =========================================================
    # BASIC
    # =========================================================
    def _sample_target(self):
        if self.substage in ("3B", "3C"):
            x = np.random.uniform(*self.config.TARGET_X_RANGE_3BC)
            y = np.random.uniform(*self.config.TARGET_Y_RANGE_3BC)
            return np.array([x, y, self.config.TARGET_Z], dtype=np.float32)
        return np.array(self.config.TARGET_POS_3A, dtype=np.float32)

    def _pack_obs(self):
        return np.concatenate(list(self.frame_buffer), axis=-1)

    def _ee_pose(self):
        ee_pos, ee_orn = p.getLinkState(
            self.robot,
            self.ctrl.EE_LINK,
            physicsClientId=self.cid,
        )[4:6]
        return np.array(ee_pos, dtype=np.float32), ee_orn

    def _object_pose(self):
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.cid)
        return np.array(obj_pos, dtype=np.float32), obj_orn

    def _object_half_extents(self):
        aabb_min, aabb_max = p.getAABB(self.object_id, physicsClientId=self.cid)
        aabb_min = np.array(aabb_min, dtype=np.float32)
        aabb_max = np.array(aabb_max, dtype=np.float32)
        return np.maximum(0.5 * (aabb_max - aabb_min), 1e-4)

    def _step_sim(self, n=1):
        for _ in range(int(n)):
            p.stepSimulation(physicsClientId=self.cid)

    def _open_gripper(self):
        self.ctrl.open_gripper()

    def _close_gripper(self):
        self.ctrl.close_gripper()

    def _draw_target_circle_fallback(self, center, radius, life=0.25):
        pts = []
        segments = 28
        for i in range(segments):
            th = 2.0 * math.pi * i / segments
            pts.append([
                center[0] + radius * math.cos(th),
                center[1] + radius * math.sin(th),
                center[2],
            ])
        for i in range(segments):
            self.debug.line(
                pts[i],
                pts[(i + 1) % segments],
                color=(1.0, 0.0, 1.0),
                width=3,
                life=life,
            )

    # =========================================================
    # PRETTY POSE
    # =========================================================
    def _apply_pretty_arm_pose(self):
        for i, j in enumerate(self.ctrl.ARM_JOINTS):
            q = self.pretty_q[i]
            p.resetJointState(self.robot, j, q, physicsClientId=self.cid)
            p.setJointMotorControl2(
                self.robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=q,
                force=300,
                physicsClientId=self.cid,
            )

        # gripper chúc thẳng xuống
        self.ctrl.grip_yaw = 0.0
        self.ctrl.ee_orn = p.getQuaternionFromEuler([math.pi, 0.0, self.ctrl.grip_yaw])

        self._open_gripper()
        self._step_sim(30)

        ee_pos, _ = self._ee_pose()
        self.ctrl.target_pos = list(ee_pos)

    def _ee_target_for_object_center(self, obj_center_xyz, lift_extra=0.0):
        obj_center_xyz = np.array(obj_center_xyz, dtype=np.float32)
        grasp_offset = np.array(self.config.GRASP_OFFSET_LOCAL, dtype=np.float32)

        ee_target = obj_center_xyz.copy()
        ee_target[0] -= grasp_offset[0]
        ee_target[1] -= grasp_offset[1]
        ee_target[2] -= grasp_offset[2]
        ee_target[2] += float(lift_extra)
        return ee_target

    def _move_tcp_toward(self, target_xyz, steps=40, close_gripper=True):
        target_xyz = np.array(target_xyz, dtype=np.float32)

        for _ in range(steps):
            cur = np.array(self.ctrl.target_pos, dtype=np.float32)
            delta = target_xyz - cur

            dx = float(np.clip(delta[0], -self.config.ACTION_SCALE_XY, self.config.ACTION_SCALE_XY))
            dy = float(np.clip(delta[1], -self.config.ACTION_SCALE_XY, self.config.ACTION_SCALE_XY))
            dz = float(np.clip(delta[2], -self.config.ACTION_SCALE_Z, self.config.ACTION_SCALE_Z))

            self.ctrl.apply_delta_action(
                dx,
                dy,
                dz,
                self.x_range,
                self.y_range,
                self.z_range,
            )

            if close_gripper:
                self._close_gripper()
            else:
                self._open_gripper()

            self._step_sim(1)

            if self.hold_cid is not None:
                self._stabilize_held_object()

    # =========================================================
    # HOLD OBJECT NICELY IN GRIPPER
    # =========================================================
    def _grasp_center_world(self):
        ee_pos, ee_orn = self._ee_pose()

        try:
            left_state = p.getLinkState(
                self.robot,
                self.ctrl.GRIPPER_JOINTS[0],
                physicsClientId=self.cid,
            )
            right_state = p.getLinkState(
                self.robot,
                self.ctrl.GRIPPER_JOINTS[1],
                physicsClientId=self.cid,
            )

            left = np.array(left_state[4], dtype=np.float32)
            right = np.array(right_state[4], dtype=np.float32)
            finger_mid = 0.5 * (left + right)

            grasp_pos, _ = p.multiplyTransforms(
                finger_mid.tolist(),
                ee_orn,
                list(self.config.GRASP_OFFSET_LOCAL),
                [0, 0, 0, 1],
            )
            return np.array(grasp_pos, dtype=np.float32)
        except Exception:
            grasp_pos, _ = p.multiplyTransforms(
                ee_pos.tolist(),
                ee_orn,
                list(self.config.GRASP_OFFSET_LOCAL),
                [0, 0, 0, 1],
            )
            return np.array(grasp_pos, dtype=np.float32)

    def _locked_object_pose_in_gripper(self):
        grasp_center = self._grasp_center_world()
        half_ext = self._object_half_extents()

        world_pos = grasp_center.copy()
        world_pos[2] -= float(half_ext[2]) * 0.55

        # giữ cube thẳng và gọn
        world_orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        return np.array(world_pos, dtype=np.float32), world_orn

    def _stabilize_held_object(self):
        if self.hold_cid is None:
            return

        obj_pos_locked, obj_orn_locked = self._locked_object_pose_in_gripper()
        p.resetBasePositionAndOrientation(
            self.object_id,
            obj_pos_locked,
            obj_orn_locked,
            physicsClientId=self.cid,
        )
        p.resetBaseVelocity(
            self.object_id,
            [0, 0, 0],
            [0, 0, 0],
            physicsClientId=self.cid,
        )

    def _disable_robot_object_collision(self):
        for link_idx in range(p.getNumJoints(self.robot, physicsClientId=self.cid)):
            try:
                p.setCollisionFilterPair(
                    self.robot,
                    self.object_id,
                    link_idx,
                    -1,
                    0,
                    physicsClientId=self.cid,
                )
            except Exception:
                pass

    def _enable_robot_object_collision(self):
        for link_idx in range(p.getNumJoints(self.robot, physicsClientId=self.cid)):
            try:
                p.setCollisionFilterPair(
                    self.robot,
                    self.object_id,
                    link_idx,
                    -1,
                    1,
                    physicsClientId=self.cid,
                )
            except Exception:
                pass

    def _attach(self, snap=True):
        if self.hold_cid is not None:
            return True

        obj_pos_locked, obj_orn_locked = self._locked_object_pose_in_gripper()

        if snap:
            p.resetBasePositionAndOrientation(
                self.object_id,
                obj_pos_locked,
                obj_orn_locked,
                physicsClientId=self.cid,
            )
            p.resetBaseVelocity(
                self.object_id,
                [0, 0, 0],
                [0, 0, 0],
                physicsClientId=self.cid,
            )

        ee_pos, ee_orn = self._ee_pose()

        inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos.tolist(), ee_orn)
        child_local_pos, child_local_orn = p.multiplyTransforms(
            inv_ee_pos,
            inv_ee_orn,
            obj_pos_locked.tolist(),
            obj_orn_locked,
        )

        self.hold_cid = p.createConstraint(
            parentBodyUniqueId=self.robot,
            parentLinkIndex=self.ctrl.EE_LINK,
            childBodyUniqueId=self.object_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=child_local_pos,
            parentFrameOrientation=child_local_orn,
            childFramePosition=[0, 0, 0],
            childFrameOrientation=[0, 0, 0, 1],
            physicsClientId=self.cid,
        )

        try:
            p.changeConstraint(
                self.hold_cid,
                maxForce=self.config.GRASP_FORCE,
                physicsClientId=self.cid,
            )
        except Exception:
            pass

        self._disable_robot_object_collision()

        for _ in range(10):
            self._close_gripper()
            self._step_sim(1)
            self._stabilize_held_object()

        return True

    def _detach(self):
        if self.hold_cid is None:
            return
        try:
            p.removeConstraint(self.hold_cid, physicsClientId=self.cid)
        except Exception:
            pass
        self.hold_cid = None
        self.release_cooldown = self.config.RELEASE_COOLDOWN_STEPS
        self._enable_robot_object_collision()

    # =========================================================
    # PLACE SEQUENCE
    # =========================================================
    def _place_sequence_from_held(self):
        if self.hold_cid is None:
            return

        hover_target = self._ee_target_for_object_center(
            [self.target_pos[0], self.target_pos[1], self.target_pos[2]],
            lift_extra=self.hover_h,
        )
        self._move_tcp_toward(hover_target, steps=45, close_gripper=True)

        pre_drop_target = self._ee_target_for_object_center(
            [self.target_pos[0], self.target_pos[1], self.target_pos[2]],
            lift_extra=self.pre_drop_h,
        )
        self._move_tcp_toward(pre_drop_target, steps=35, close_gripper=True)

        drop_target = self._ee_target_for_object_center(
            [self.target_pos[0], self.target_pos[1], self.target_pos[2]],
            lift_extra=self.drop_h,
        )
        self._move_tcp_toward(drop_target, steps=35, close_gripper=True)

        for _ in range(12):
            self.ctrl.apply_delta_action(
                0.0, 0.0, 0.0,
                self.x_range, self.y_range, self.z_range,
            )
            self._close_gripper()
            self._step_sim(1)
            self._stabilize_held_object()

        for _ in range(14):
            self.ctrl.apply_delta_action(
                0.0, 0.0, 0.0,
                self.x_range, self.y_range, self.z_range,
            )
            self._open_gripper()
            self._step_sim(1)

        self._detach()

        retreat_target = self._ee_target_for_object_center(
            [self.target_pos[0], self.target_pos[1], self.target_pos[2]],
            lift_extra=0.08,
        )
        self._move_tcp_toward(retreat_target, steps=20, close_gripper=False)

    # =========================================================
    # SCENE
    # =========================================================
    def _setup_scene(self):
        p.resetSimulation(physicsClientId=self.cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(1.0 / self.config.PHYSICS_HZ, physicsClientId=self.cid)

        p.loadURDF("plane.urdf", physicsClientId=self.cid)

        table_half = [0.22, 0.18, 0.02]
        self.table_top_z = table_half[2] * 2.0

        table_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=table_half,
            physicsClientId=self.cid,
        )
        table_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=table_half,
            rgbaColor=[0.90, 0.90, 0.96, 0.85],
            physicsClientId=self.cid,
        )
        self.table_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=table_col,
            baseVisualShapeIndex=table_vis,
            basePosition=[0.46, 0.0, table_half[2]],
            physicsClientId=self.cid,
        )

        self.target_pos = self._sample_target()
        self.target_pos[2] = self.table_top_z
        self.rewarder.set_target(self.target_pos)

        self.robot = p.loadURDF(
            "franka_panda/panda.urdf",
            useFixedBase=True,
            physicsClientId=self.cid,
        )

        self.ctrl = PandaController(
            self.robot,
            physics_client_id=self.cid,
            grip_yaw=0.0,
        )

        self._apply_pretty_arm_pose()
        self.rewarder.ee_link = self.ctrl.EE_LINK

        # object spawn trước khi snap vào gripper
        self.object_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=[0.34, -0.12, self.table_top_z + 0.03],
            physicsClientId=self.cid,
        )
        try:
            p.changeVisualShape(
                self.object_id,
                -1,
                rgbaColor=[0.08, 0.25, 1.0, 1.0],
                physicsClientId=self.cid,
            )
        except Exception:
            pass

        self.target_vis_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=self.target_pos.tolist(),
            useFixedBase=True,
            physicsClientId=self.cid,
        )
        try:
            p.changeVisualShape(
                self.target_vis_id,
                -1,
                rgbaColor=[1.0, 0.0, 0.0, 0.65],
                physicsClientId=self.cid,
            )
        except Exception:
            pass

        for _ in range(60):
            self._step_sim(1)

        self._detach()
        self.release_cooldown = 0
        self._enable_robot_object_collision()

        # BẮT ĐẦU TRONG TRẠNG THÁI ĐANG CẦM VẬT
        if self.start_held:
            self._attach(snap=True)

            for _ in range(10):
                self._close_gripper()
                self._step_sim(1)
                self._stabilize_held_object()

    # =========================================================
    # GYM API
    # =========================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._setup_scene()
        self.rewarder.reset()

        self.step_count = 0
        self.last_grip = 1.0

        # reset là đã cầm sẵn, đưa tới target luôn
        if self.start_held:
            self._place_sequence_from_held()

        rgb = self.camera.render_rgb()
        self.frame_buffer = np.repeat(rgb[None, ...], self.config.FRAME_STACK, axis=0)

        return self._pack_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        rgb = self.camera.render_rgb()
        self.frame_buffer = np.roll(self.frame_buffer, -1, axis=0)
        self.frame_buffer[-1] = rgb

        ee_pos, ee_orn = p.getLinkState(
            self.robot,
            self.ctrl.EE_LINK,
            physicsClientId=self.cid,
        )[4:6]
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.cid)

        ee_pos_np = np.array(ee_pos, dtype=np.float32)
        obj_pos_np = np.array(obj_pos, dtype=np.float32)
        target_pos_np = np.array(self.target_pos, dtype=np.float32)

        ee_obj_dist = float(np.linalg.norm(ee_pos_np - obj_pos_np))
        obj_target_dist = float(np.linalg.norm(obj_pos_np - target_pos_np))
        holding = self.hold_cid is not None

        reward, terminated, info = self.rewarder.compute(
            self.robot,
            self.object_id,
            grip=self.last_grip,
            holding=holding,
        )

        self.step_count += 1
        truncated = self.step_count >= self.config.MAX_STEPS

        info.update(
            {
                "step_count": self.step_count,
                "place_substage": self.substage,
                "substage": self.substage,
                "holding": bool(holding),
                "ee_pos": [float(x) for x in ee_pos],
                "obj_pos": [float(x) for x in obj_pos],
                "target_pos": [float(x) for x in self.target_pos.tolist()],
                "target_ee_pos": [float(x) for x in self.ctrl.target_pos],
                "ee_obj_dist": ee_obj_dist,
                "obj_target_dist": obj_target_dist,
                "success_dist": float(self.rewarder.success_dist),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

        if self.use_gui:
            self.debug.clear()
            self.debug.axes(ee_pos, ee_orn, life=0.25, width=4)
            self.debug.point(ee_pos, color=(1, 0, 1), text="TCP", life=0.25)
            self.debug.point(obj_pos, color=(0.08, 0.25, 1.0), text="blue", life=0.25)
            self.debug.point(self.target_pos.tolist(), color=(1, 0, 0), text="red", life=0.25)
            self._draw_gripper_direction(ee_pos, ee_orn, length=0.18)

            circle_center = [
                float(self.target_pos[0]),
                float(self.target_pos[1]),
                float(self.target_pos[2]) + 0.002,
            ]
            try:
                if hasattr(self.debug, "circle"):
                    self.debug.circle(
                        center=circle_center,
                        radius=float(self.rewarder.success_dist),
                        color=(1.0, 0.0, 1.0),
                        width=3,
                        segments=28,
                        life=0.25,
                        axis="z",
                    )
                else:
                    self._draw_target_circle_fallback(
                        circle_center,
                        float(self.rewarder.success_dist),
                        life=0.25,
                    )
            except Exception:
                self._draw_target_circle_fallback(
                    circle_center,
                    float(self.rewarder.success_dist),
                    life=0.25,
                )

        return self._pack_obs(), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self._detach()
        try:
            p.disconnect(self.cid)
        except Exception:
            pass