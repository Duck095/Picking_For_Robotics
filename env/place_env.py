import math
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

# Đảm bảo các module này nằm đúng cấu trúc thư mục của Boss Hướng
from config.place_env_config import PlaceEnvConfig
from env.camera import Camera
from env.panda_controller import PandaController
from env.reward_place import RewardModulePlace
from env.utils_debug import DebugDraw

class PlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode="rgb_array", use_gui=False, start_held=True, substage="3A"):
        super().__init__()
        self.config = PlaceEnvConfig()
        self.use_gui = bool(use_gui)
        self.start_held = bool(start_held)
        self.substage = str(substage).upper()

        # Kết nối PyBullet duy nhất 1 lần
        self.cid = p.connect(p.GUI if self.use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 3 * self.config.FRAME_STACK), dtype=np.uint8)
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, 0], dtype=np.float32), high=np.array([1, 1, 1, 1], dtype=np.float32), dtype=np.float32)

        self.camera = Camera(img_size=self.config.IMG_SIZE, physics_client_id=self.cid, use_gui=self.use_gui)
        self.debug = DebugDraw(physics_client_id=self.cid)
        
        self.target_pos = np.array(self.config.TARGET_POS_3A, dtype=np.float32)
        params = self.config.get_success_params(self.substage)
        self.rewarder = RewardModulePlace(ee_link=11, target_pos=self.target_pos, physics_client_id=self.cid, **params)

        self.hold_cid = None
        self.step_count = 0
        self._setup_scene()

    def _setup_scene(self):
        p.resetSimulation(physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.loadURDF("plane.urdf", physicsClientId=self.cid)

        # 1. Robot & Controller
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, physicsClientId=self.cid)
        self.ctrl = PandaController(self.robot, physics_client_id=self.cid)
        self.ctrl.reset_home()
        self.rewarder.ee_link = self.ctrl.EE_LINK

        # 2. Vật thể MÀU HỒNG (Boss yêu cầu)
        self.object_id = p.loadURDF("cube_small.urdf", basePosition=list(self.config.OBJECT_SPAWN_POS), physicsClientId=self.cid)
        p.changeVisualShape(self.object_id, -1, rgbaColor=[1.0, 0.08, 0.58, 1.0], physicsClientId=self.cid) 

        # 3. Target Ô MÀU ĐỎ
        self.target_pos = self._sample_target()
        self.target_vis_id = p.loadURDF("cube_small.urdf", basePosition=self.target_pos.tolist(), useFixedBase=True, physicsClientId=self.cid)
        p.changeVisualShape(self.target_vis_id, -1, rgbaColor=[1, 0, 0, 0.5], physicsClientId=self.cid)

        if self.start_held:
            self._force_start_holding()

    def _attach(self):
        """Khóa vật chuẩn tư thế vào giữa gripper"""
        if self.hold_cid is not None: return True
        ee_pos, ee_orn = p.getLinkState(self.robot, self.ctrl.EE_LINK, physicsClientId=self.cid)[4:6]
        snap_pos, _ = p.multiplyTransforms(ee_pos, ee_orn, [0, 0, -0.015], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.object_id, snap_pos, ee_orn, physicsClientId=self.cid)
        
        for j in range(p.getNumJoints(self.robot)):
            p.setCollisionFilterPair(self.robot, self.object_id, j, -1, 0, physicsClientId=self.cid)

        inv_ee_p, inv_ee_o = p.invertTransform(ee_pos, ee_orn)
        loc_p, loc_o = p.multiplyTransforms(inv_ee_p, inv_ee_o, snap_pos, ee_orn)
        
        self.hold_cid = p.createConstraint(self.robot, self.ctrl.EE_LINK, self.object_id, -1, p.JOINT_FIXED, [0, 0, 0], loc_p, [0, 0, 0, 1], parentFrameOrientation=loc_o, physicsClientId=self.cid)
        return True

    def _detach(self):
        """Thả vật chuẩn xác"""
        if self.hold_cid is not None:
            p.removeConstraint(self.hold_cid, physicsClientId=self.cid)
            self.hold_cid = None
            for j in range(p.getNumJoints(self.robot)):
                p.setCollisionFilterPair(self.robot, self.object_id, j, -1, 1, physicsClientId=self.cid)

    def _force_start_holding(self):
        """Tự động cầm vật và bay đến trên đầu Target khi Reset"""
        self.ctrl.open_gripper()
        self._attach()
        for _ in range(20):
            self.ctrl.close_gripper()
            p.stepSimulation(physicsClientId=self.cid)
        
        hover_pos = [self.target_pos[0], self.target_pos[1], self.target_pos[2] + 0.15]
        self.ctrl.target_pos = hover_pos
        for _ in range(40):
            self.ctrl.apply_delta_action(0, 0, 0, self.config.X_RANGE, self.config.Y_RANGE, self.config.Z_RANGE)
            p.stepSimulation(physicsClientId=self.cid)

    def step(self, action):
        dx, dy, dz, grip = action
        
        if grip > 0.5:
            if self.hold_cid is None: self._attach()
            self.ctrl.close_gripper()
        else:
            self._detach()
            self.ctrl.open_gripper()

        self.ctrl.apply_delta_action(dx * 0.05, dy * 0.05, dz * 0.05, self.config.X_RANGE, self.config.Y_RANGE, self.config.Z_RANGE)
        for _ in range(self.config.SUBSTEPS): p.stepSimulation(physicsClientId=self.cid)

        # Fix lỗi AttributeError: Vẽ tia vàng hướng xuống (thay thế cho _draw_gripper_direction)
        if self.use_gui:
            self.debug.clear()
            ee_p, ee_o = p.getLinkState(self.robot, self.ctrl.EE_LINK, physicsClientId=self.cid)[4:6]
            ray_e, _ = p.multiplyTransforms(ee_p, ee_o, [0, 0, -0.3], [0, 0, 0, 1])
            self.debug.line(ee_p, ray_e, color=(1, 1, 0), width=4) # Đường kẻ vàng Boss yêu cầu

        rgb = self.camera.render_rgb()
        self.frame_buffer = np.roll(self.frame_buffer, -1, axis=0)
        self.frame_buffer[-1] = rgb
        
        reward, done, info = self.rewarder.compute(self.robot, self.object_id, grip, self.hold_cid is not None)
        self.step_count += 1
        return self._pack_obs(), float(reward), bool(done), self.step_count >= self.config.MAX_STEPS, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup_scene()
        self.step_count = 0
        rgb = self.camera.render_rgb()
        self.frame_buffer = np.repeat(rgb[None, ...], self.config.FRAME_STACK, axis=0)
        return self._pack_obs(), {}

    def _pack_obs(self): return np.concatenate(list(self.frame_buffer), axis=-1)
    def _sample_target(self): return np.array(self.config.TARGET_POS_3A)
    def close(self): p.disconnect(self.cid)

# --- VÒNG LẶP TEST DUY TRÌ GUI ---
if __name__ == "__main__":
    env = PlaceEnv(use_gui=True, start_held=True, substage="3A")
    env.reset()
    print("Boss Hướng chạy thử nhé! Nhấn Ctrl+C để tắt.")
    try:
        while True:
            env.step([0, 0, 0, 1]) # Đứng yên giữ vật
            time.sleep(1./60.)
    except KeyboardInterrupt:
        env.close()