import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import math

from config.env_config import EnvConfig
from .camera import Camera
from env.panda_controller import PandaController
from env.grasp_module import SimpleAttachGrasp
from env.reward_place import RewardModulePlace


class PlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode="rgb_array", use_gui=False, start_held=True, substage="3A"):
        super().__init__()
        self.config = EnvConfig()
        self.use_gui = use_gui
        self.start_held = start_held
        self.substage = str(substage)

        # ✅ workspace ranges for PandaController.apply_delta_action
        self.x_range = (0.10, 0.85)
        self.y_range = (-0.60, 0.60)
        self.z_range = (0.005, 0.55)

        self.cid = p.connect(p.GUI) if use_gui else p.connect(p.DIRECT)
        if use_gui:
            print("Connected GUI:", self.cid)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(1.0 / self.config.PHYSICS_HZ, physicsClientId=self.cid)

        self.observation_space = spaces.Box(
        low=0, high=255,
        shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 3 * self.config.FRAME_STACK),
        dtype=np.uint8,
        )
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # ✅ camera.py uses render_rgb()
        self.camera = Camera(
            img_size=self.config.IMG_SIZE,
            physics_client_id=self.cid,
            use_gui=self.use_gui,
            use_crop=True,
            crop_box=(16, 112, 16, 112),
        )

        # ✅ legacy grasp module
        self.grasper = SimpleAttachGrasp(ee_link=11, max_dist=0.12, max_force=2500)

        # default target
        self.target_pos = np.array([0.45, 0.20, 0.02], dtype=np.float32)

        success_dist, table_z, z_release_max = self._params_for_substage(self.substage)
        self.rewarder = RewardModulePlace(
            ee_link=11,
            target_pos=self.target_pos,
            time_penalty=0.01,
            dist_weight=0.3,
            release_bonus=1.0,
            success_bonus=3.0,
            success_dist=success_dist,
            table_z=table_z,
            z_release_max=z_release_max,
            high_release_penalty=1.0,
            physics_client_id=self.cid,
        )

        self.frame_buffer = None
        self.step_count = 0
        self.last_grip = 0.0

        self.robot = None
        self.ctrl = None
        self.object_id = None
        self.target_vis_id = None

        self._setup_scene()

    def _params_for_substage(self, substage: str):
        if substage == "3A":
            return 0.12, 0.06, None
        if substage == "3B":
            return 0.07, 0.06, None
        if substage == "3C":
            return 0.06, 0.06, 0.08
        return 0.12, 0.06, None

    def _sample_target(self):
        if self.substage in ("3B", "3C"):
            x = np.random.uniform(0.40, 0.55)
            y = np.random.uniform(0.10, 0.30)
            return np.array([x, y, 0.02], dtype=np.float32)
        return np.array([0.45, 0.20, 0.02], dtype=np.float32)

    def _setup_scene(self):
        p.resetSimulation(physicsClientId=self.cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(1.0 / self.config.PHYSICS_HZ, physicsClientId=self.cid)

        p.loadURDF("plane.urdf", physicsClientId=self.cid)

        # target per episode
        self.target_pos = self._sample_target()
        self.rewarder.set_target(self.target_pos)

        # robot
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, physicsClientId=self.cid)
        self.ctrl = PandaController(self.robot, physics_client_id=self.cid, grip_yaw=math.pi / 2)
        self.ctrl.reset_home()

        ee = self.ctrl.EE_LINK
        self.grasper.ee_link = ee
        self.rewarder.ee_link = ee

        # object
        self.object_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=[0.55, 0.0, 0.02],
            physicsClientId=self.cid
        )

        # target marker
        self.target_vis_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=self.target_pos.tolist(),
            useFixedBase=True,
            physicsClientId=self.cid
        )
        try:
            p.changeVisualShape(self.target_vis_id, -1, rgbaColor=[1, 0, 0, 0.6], physicsClientId=self.cid)
        except Exception:
            pass

        for _ in range(60):
            p.stepSimulation(physicsClientId=self.cid)

        self.grasper.reset()
        if self.start_held:
            self._force_start_holding()

    def _force_start_holding(self):
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.cid)
        self.ctrl.target_pos = [obj_pos[0], obj_pos[1], max(obj_pos[2] + 0.10, 0.12)]

        # down
        for _ in range(40):
            self.ctrl.apply_delta_action(
                0.0, 0.0, -0.5 * self.config.ACTION_SCALE_Z,
                x_range=self.x_range, y_range=self.y_range, z_range=self.z_range
            )
            self.ctrl.open_gripper()
            p.stepSimulation(physicsClientId=self.cid)

        # close + attach
        for _ in range(30):
            self.ctrl.apply_delta_action(
                0.0, 0.0, 0.0,
                x_range=self.x_range, y_range=self.y_range, z_range=self.z_range
            )
            self.ctrl.close_gripper()
            self.grasper.try_attach(self.robot, self.object_id, 1.0)
            p.stepSimulation(physicsClientId=self.cid)

        # lift
        for _ in range(30):
            self.ctrl.apply_delta_action(
                0.0, 0.0, 0.5 * self.config.ACTION_SCALE_Z,
                x_range=self.x_range, y_range=self.y_range, z_range=self.z_range
            )
            self.ctrl.close_gripper()
            p.stepSimulation(physicsClientId=self.cid)
            
    def _pack_obs(self):
    # frame_buffer: (S, H, W, 3) -> (H, W, 3*S)
        return np.concatenate(list(self.frame_buffer), axis=-1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup_scene()
        self.rewarder.reset()

        self.last_grip = 0.0
        self.step_count = 0

        obs = self.camera.render_rgb()
        self.frame_buffer = np.repeat(obs[None, ...], self.config.FRAME_STACK, axis=0)
        return self._pack_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        dx, dy, dz, grip = action

        dx = float(dx) * self.config.ACTION_SCALE_XY
        dy = float(dy) * self.config.ACTION_SCALE_XY
        dz = float(dz) * self.config.ACTION_SCALE_Z
        self.last_grip = float(grip)

        # move
        self.ctrl.apply_action(dx, dy, dz, self.last_grip)

        # grasp attach/detach
        self.grasper.detach_if_open(self.last_grip)
        self.grasper.try_attach(self.robot, self.object_id, self.last_grip)

        # physics
        for _ in range(self.config.SUBSTEPS):
            p.stepSimulation(physicsClientId=self.cid)

        # update observation buffer
        obs = self.camera.render()
        self.frame_buffer = np.roll(self.frame_buffer, -1, axis=0)
        self.frame_buffer[-1] = obs

        # **Thêm đoạn này**: kiểm tra xem có đang cầm vật không
        holding = self.grasper.cid is not None

        # reward
        reward, terminated, info = self.rewarder.compute(
            self.robot, self.object_id, grip=self.last_grip, holding=holding
        )

        # Thêm điều kiện này để không terminate quá sớm
        min_steps_before_done = 10  # Bắt đầu kết thúc sau 10 bước
        if self.step_count < min_steps_before_done:
            terminated = False
            if isinstance(info, dict):
                info["success"] = False  # Chưa thành công, không cho terminate

        self.step_count += 1
        truncated = self.step_count >= self.config.MAX_STEPS

        return self.frame_buffer, float(reward), bool(terminated), bool(truncated), info