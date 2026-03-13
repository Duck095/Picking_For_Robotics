# env/reach_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import random
import math

from config.reach_env_config import EnvConfig
from env.camera import Camera
from env.panda_controller import PandaController
from env.reward_reach import ReachReward
from env.utils_debug import DebugDraw


class ReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, use_gui: bool = False, config=None):
        super().__init__()
        
        if config is None:
            config = EnvConfig()
    
        self.cfg = config or EnvConfig()
        self.use_gui = bool(use_gui)

        # ✅ lưu physics client id
        self.cid = p.connect(p.GUI) if self.use_gui else p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(1.0 / self.cfg.physics_hz, physicsClientId=self.cid)

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.cfg.img_size, self.cfg.img_size, 3 * self.cfg.frame_stack),
            dtype=np.uint8,
        )

        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # ✅ camera đúng world + tiny renderer khi headless + crop ROI
        self.camera = Camera(
            img_size=self.cfg.img_size,
            physics_client_id=self.cid,
            use_gui=self.use_gui,
            use_crop=True,
            crop_box=(16, 112, 16, 112),
        )

        self.debug = DebugDraw(physics_client_id=self.cid)

        self.robot_id = None
        self.ctrl = None
        self.obj_id = None
        self.rewarder = None

        self.frame_buf = None
        self.step_count = 0

        self._setup_scene()

    def _sample_obj_xy(self):
        if self.cfg.stage1_substage == "1A":
            xr, yr = self.cfg.stage1a_x, self.cfg.stage1a_y
        elif self.cfg.stage1_substage == "1B":
            xr, yr = self.cfg.stage1b_x, self.cfg.stage1b_y
        elif self.cfg.stage1_substage == "1C":
            xr, yr = self.cfg.stage1c_x, self.cfg.stage1c_y
        else:
            raise ValueError(f"Unknown stage1_substage: {self.cfg.stage1_substage}")

        x = np.random.uniform(*xr)
        y = np.random.uniform(*yr)
        return x, y

    def _setup_scene(self):
        p.resetSimulation(physicsClientId=self.cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(1.0 / self.cfg.physics_hz, physicsClientId=self.cid)

        p.loadURDF("plane.urdf", physicsClientId=self.cid)
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, physicsClientId=self.cid)

        self.ctrl = PandaController(self.robot_id, physics_client_id=self.cid, grip_yaw=math.pi / 2)
        self.ctrl.reset_home()

        x, y = self._sample_obj_xy()
        self.obj_id = p.loadURDF("cube_small.urdf", basePosition=[x, y, self.cfg.obj_z], physicsClientId=self.cid)

        obj_pos, _ = p.getBasePositionAndOrientation(self.obj_id, physicsClientId=self.cid)
        self.ctrl.target_pos = [obj_pos[0], obj_pos[1], 0.25]

        # ✅ chọn success_dist theo substage
        if self.cfg.stage1_substage == "1A":
            success_dist = self.cfg.success_dist_1a
        elif self.cfg.stage1_substage == "1B":
            success_dist = self.cfg.success_dist_1b
        elif self.cfg.stage1_substage == "1C":
            success_dist = self.cfg.success_dist_1c
        else:
            raise ValueError(f"Unknown stage1_substage: {self.cfg.stage1_substage}")

        # ✅ reward dùng đúng EE link controller tìm ra
        self.rewarder = ReachReward(
            ee_link=self.ctrl.EE_LINK,
            success_dist=success_dist,
            dist_weight=self.cfg.dist_weight,
            time_penalty=self.cfg.time_penalty,
            success_bonus=self.cfg.success_bonus,
            physics_client_id=self.cid,
            delta_clip=getattr(self.cfg, "delta_clip", 0.04),  # fallback nếu config chưa có
        )

        for _ in range(30):
            p.stepSimulation(physicsClientId=self.cid)

    def _obs(self):
        return np.concatenate(list(self.frame_buf), axis=2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup_scene()
        self.rewarder.reset()

        self.step_count = 0
        rgb = self.camera.render_rgb()
        self.frame_buf = np.repeat(rgb[None, ...], self.cfg.frame_stack, axis=0)
        return self._obs(), {}

    def step(self, action):
        # ✅ clip action
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        dx, dy, dz = action

        dx = float(dx) * self.cfg.action_scale_xy
        dy = float(dy) * self.cfg.action_scale_xy
        dz = float(dz) * self.cfg.action_scale_z

        self.ctrl.apply_delta_action(
            dx, dy, dz,
            x_range=self.cfg.x_range,
            y_range=self.cfg.y_range,
            z_range=self.cfg.z_range,
        )

        for _ in range(self.cfg.substeps):
            p.stepSimulation(physicsClientId=self.cid)

        rgb = self.camera.render_rgb()
        self.frame_buf = np.roll(self.frame_buf, -1, axis=0)
        self.frame_buf[-1] = rgb

        reward, terminated, info = self.rewarder.compute(self.robot_id, self.obj_id)

        self.step_count += 1
        truncated = self.step_count >= self.cfg.max_steps

        if self.use_gui:
            self.debug.clear()
            ee_pos = p.getLinkState(self.robot_id, self.ctrl.EE_LINK, physicsClientId=self.cid)[4]
            ee_orn = p.getLinkState(self.robot_id, self.ctrl.EE_LINK, physicsClientId=self.cid)[5]
            obj_pos, _ = p.getBasePositionAndOrientation(self.obj_id, physicsClientId=self.cid)
            self.debug.axes(ee_pos, ee_orn, life=0.2, width=4)
            self.debug.point(ee_pos, color=(1, 0, 1), text="TCP", life=0.2)
            self.debug.point(obj_pos, color=(1, 1, 0), text="OBJ", life=0.2)
            self.debug.line(ee_pos, obj_pos, life=0.2)


        info["step_count"] = self.step_count
        info["stage1_substage"] = self.cfg.stage1_substage
        info["target_ee_pos"] = [float(x) for x in self.ctrl.target_pos]
        info["action"] = [float(x) for x in action.tolist()]
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)

        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        try:
            p.disconnect(self.cid)
        except Exception:
            pass