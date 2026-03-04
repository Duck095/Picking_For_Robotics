import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import math

from config.env_config import EnvConfig
from .camera_module import CameraModule
from env.robot_controller import PandaController
from env.grasp_module import SimpleAttachGrasp
from env.reward_module import RewardModuleStage1


class ConveyorEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode="rgb_array", use_gui=False):
        super().__init__()

        self.config = EnvConfig()
        self.use_gui = use_gui

        # ✅ lưu client id
        self.cid = p.connect(p.GUI) if use_gui else p.connect(p.DIRECT)
        if use_gui:
            print("Connected GUI:", self.cid)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(1.0 / self.config.PHYSICS_HZ, physicsClientId=self.cid)

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.config.FRAME_STACK, self.config.IMG_SIZE, self.config.IMG_SIZE, 3),
            dtype=np.uint8,
        )

        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self.camera = CameraModule(self.config.IMG_SIZE, physics_client_id=self.cid, use_tiny_renderer=not self.use_gui)
        self.frame_buffer = None
        self.step_count = 0
        self.last_grip = 0.0

        self.grasper = SimpleAttachGrasp(ee_link=11, max_dist=0.09, max_force=2500,  physics_client_id=self.cid, close_th=0.5, open_th=0.5)
        self.rewarder = RewardModuleStage1(
            ee_link=11,
            lift_height=self.config.LIFT_HEIGHT,
            time_penalty=self.config.TIME_PENALTY,
            dist_weight=self.config.DIST_WEIGHT,
            grasp_reward=self.config.GRASP_REWARD,
            success_bonus=self.config.SUCCESS_BONUS,
        )

        self.robot = None
        self.ctrl = None
        self.object_id = None

        self._setup_scene()

    def _setup_scene(self):
        p.resetSimulation(physicsClientId=self.cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(1.0 / self.config.PHYSICS_HZ, physicsClientId=self.cid)

        p.loadURDF("plane.urdf", physicsClientId=self.cid)

        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, physicsClientId=self.cid)
        self.ctrl = PandaController(self.robot, grip_yaw=math.pi / 2)
        self.ctrl.reset_home()

        # ✅ đồng bộ EE link cho grasp + reward
        ee = self.ctrl.EE_LINK
        self.grasper.ee_link = ee
        self.rewarder.ee_link = ee

        self.object_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=[0.55, 0.0, 0.02],
            physicsClientId=self.cid
        )

        for _ in range(30):
            p.stepSimulation(physicsClientId=self.cid)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._setup_scene()
        self.rewarder.reset()
        self.grasper.reset()

        self.last_grip = 0.0
        self.step_count = 0

        obs = self.camera.render()
        self.frame_buffer = np.repeat(obs[None, ...], self.config.FRAME_STACK, axis=0)
        return self.frame_buffer, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        dx, dy, dz, grip = action

        dx = float(dx) * self.config.ACTION_SCALE_XY
        dy = float(dy) * self.config.ACTION_SCALE_XY
        dz = float(dz) * self.config.ACTION_SCALE_Z
        self.last_grip = float(grip)

        # 1) move robot
        self.ctrl.apply_action(dx, dy, dz, self.last_grip)

        # 2) attach/detach BEFORE physics
        self.grasper.detach_if_open(self.last_grip)
        self.grasper.try_attach(self.robot, self.object_id, self.last_grip)

        # 3) physics
        for _ in range(self.config.SUBSTEPS):
            p.stepSimulation(physicsClientId=self.cid)

        # 4) update obs buffer
        obs = self.camera.render()
        self.frame_buffer = np.roll(self.frame_buffer, -1, axis=0)
        self.frame_buffer[-1] = obs

        # 5) reward
        # ✅ lấy holding từ grasper (tùy bạn đặt tên thuộc tính)
        holding = False
        if hasattr(self.grasper, "holding"):
            holding = self.grasper.holding
        elif hasattr(self.grasper, "constraint_id"):
            holding = self.grasper.constraint_id is not None
        else:
            holding = self.grasper.cid is not None  # legacy fallback

        reward, terminated, info = self.rewarder.compute(
            self.robot, self.object_id, self.last_grip, holding
        )
        
        self.step_count += 1
        truncated = self.step_count >= self.config.MAX_STEPS

        return self.frame_buffer, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        try:
            p.disconnect(self.cid)
        except Exception:
            pass