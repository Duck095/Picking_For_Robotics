import math
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

# Các module từ folder env và config của Boss
from config.place_env_config import PlaceEnvConfig
from env.camera import Camera
from env.panda_controller import PandaController
from env.reward_place import RewardModulePlace
from env.utils_debug import DebugDraw

class PlaceEnv(gym.Env):
    def __init__(self, render_mode="rgb_array", use_gui=False, start_held=True, substage="3A"):
        super().__init__()
        self.config = PlaceEnvConfig()
        self.use_gui = bool(use_gui)
        self.start_held = bool(start_held) 
        self.substage = str(substage).upper()

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

        # 1. Khởi tạo Robot tại Home
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, physicsClientId=self.cid)
        self.ctrl = PandaController(self.robot, physics_client_id=self.cid)
        self.ctrl.reset_home()

        # 2. Vật HỒNG & Target ĐỎ
        self.object_id = p.loadURDF("cube_small.urdf", basePosition=list(self.config.OBJECT_SPAWN_POS), physicsClientId=self.cid)
        p.changeVisualShape(self.object_id, -1, rgbaColor=[1.0, 0.08, 0.58, 1.0], physicsClientId=self.cid)

        self.target_pos = self._sample_target()
        self.rewarder.set_target(self.target_pos)
        self.target_vis_id = p.loadURDF("cube_small.urdf", basePosition=self.target_pos.tolist(), useFixedBase=True, physicsClientId=self.cid)
        p.changeVisualShape(self.target_vis_id, -1, rgbaColor=[1, 0, 0, 0.5], physicsClientId=self.cid)
        # Tắt hoàn toàn va chạm vật lý của khối target để vật có thể rơi xuyên qua chạm mặt bàn
        p.setCollisionFilterGroupMask(self.target_vis_id, -1, collisionFilterGroup=0, collisionFilterMask=0, physicsClientId=self.cid)

        if self.start_held:
            self._force_start_holding()

    def _pretty_object_grasp_pose(self):
        ee_pos, ee_orn = self.ctrl.get_ee_pose()
        # Vị trí End-Effector (link 11 - grasptarget) vốn dĩ đã được thiết kế 
        # nằm chính giữa điểm tiếp xúc của 2 phần đệm kẹp.
        # Chúng ta khóa thẳng tâm khối lập phương vào đúng vị trí này để kẹp hoàn hảo.
        snap_pos, _ = p.multiplyTransforms(ee_pos, ee_orn, [0, 0, 0], [0, 0, 0, 1])
        return np.array(snap_pos, dtype=np.float32), ee_orn

    def _attach(self):
        if self.hold_cid is not None: return True
        target_pos, target_orn = self._pretty_object_grasp_pose()
        for j in range(p.getNumJoints(self.robot)):
            p.setCollisionFilterPair(self.robot, self.object_id, j, -1, 0, physicsClientId=self.cid)
        p.resetBasePositionAndOrientation(self.object_id, target_pos, target_orn, physicsClientId=self.cid)
        ee_pos, ee_orn = self.ctrl.get_ee_pose()
        inv_ee_p, inv_ee_o = p.invertTransform(ee_pos, ee_orn)
        loc_p, loc_o = p.multiplyTransforms(inv_ee_p, inv_ee_o, target_pos, target_orn)
        
        self.hold_cid = p.createConstraint(
            parentBodyUniqueId=self.robot,
            parentLinkIndex=self.ctrl.EE_LINK,
            childBodyUniqueId=self.object_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=loc_p,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=loc_o,
            childFrameOrientation=[0, 0, 0, 1],
            physicsClientId=self.cid
        )
        # Tăng lực khóa lên mức khổng lồ để vật không bị văng khi di chuyển nhanh
        p.changeConstraint(self.hold_cid, maxForce=50000, physicsClientId=self.cid) 
        return True

    def _detach(self):
        if self.hold_cid is not None:
            p.removeConstraint(self.hold_cid, physicsClientId=self.cid)
            self.hold_cid = None
            # BỎ QUA việc bật lại va chạm ở đây. Việc bật lại ngay lập tức khi kẹp chưa kịp mở ra vật lý sẽ khiến 2 vật thể bị lồng vào nhau và sinh ra lực đẩy khổng lồ (vật bị nổ văng đi).
            # Vật vẫn sẽ rớt xuống và va chạm với bàn cũng như các vật thể khác bình thường.

    def _force_start_holding(self):
        self.ctrl.open_gripper()
        p.stepSimulation(physicsClientId=self.cid)
        self._attach() 
        for _ in range(20):
            self.ctrl.close_gripper(width=0.024)
            p.stepSimulation(physicsClientId=self.cid)

    def step(self, action):
        dx, dy, dz, grip = action
        if grip > 0.5:
            if self.hold_cid is None and not getattr(self, "has_dropped", False): 
                self._attach()
            self.ctrl.close_gripper(width=0.024)
        else:
            self._detach()
            self.ctrl.open_gripper()
            self.has_dropped = True

        self.ctrl.apply_delta_action(
            dx * self.config.ACTION_SCALE_XY, 
            dy * self.config.ACTION_SCALE_XY, 
            dz * self.config.ACTION_SCALE_Z, 
            self.config.X_RANGE, 
            self.config.Y_RANGE, 
            self.config.Z_RANGE
        )
        for _ in range(self.config.SUBSTEPS): p.stepSimulation(physicsClientId=self.cid)

        if self.use_gui:
            self.debug.clear()
            ee_p, ee_o = self.ctrl.get_ee_pose()
            ray_e, _ = p.multiplyTransforms(ee_p, ee_o, [0, 0, 0.3], [0, 0, 0, 1])
            self.debug.line(ee_p, ray_e, color=(1, 1, 0), width=4)

        rgb = self.camera.render_rgb()
        self.frame_buffer = np.roll(self.frame_buffer, -1, axis=0)
        self.frame_buffer[-1] = rgb
        
        reward, done, info = self.rewarder.compute(self.robot, self.object_id, grip, self.hold_cid is not None)
        self.step_count += 1
        
        ee_p, _ = self.ctrl.get_ee_pose()
        info.update({"ee_pos": ee_p, "target_pos": self.target_pos.tolist()})
        return self._pack_obs(), float(reward), bool(done), self.step_count >= self.config.MAX_STEPS, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # FIX CỰC QUAN TRỌNG: Xóa trí nhớ về cái khóa kẹp cũ trước khi setup cảnh mới!
        self.hold_cid = None 
        self.has_dropped = False
        self._setup_scene()
        self.step_count = 0
        rgb = self.camera.render_rgb()
        self.frame_buffer = np.repeat(rgb[None, ...], self.config.FRAME_STACK, axis=0)
        ee_p, _ = self.ctrl.get_ee_pose()
        info = {"ee_pos": ee_p, "target_pos": self.target_pos.tolist()}
        return self._pack_obs(), info

    def _pack_obs(self): return np.concatenate(list(self.frame_buffer), axis=-1)
    def _sample_target(self): return np.array(self.config.TARGET_POS_3A)
    def close(self):
        try:
            p.disconnect(self.cid)
        except Exception:
            pass

# ==========================================
# KỊCH BẢN ĐIỀU KHIỂN CHI TIẾT
# ==========================================
if __name__ == "__main__":
    env = PlaceEnv(use_gui=True, start_held=True)
    obs, info = env.reset()
    state = "MOVE_TO_TARGET"
    timer = 0
    try:
        while True:
            ee_pos = np.array(info["ee_pos"])
            target_pos = np.array(info["target_pos"])
            obj_pos, _ = p.getBasePositionAndOrientation(env.object_id, physicsClientId=env.cid)
            bottom_z = obj_pos[2] - 0.025 # Chiều cao khối hộp 5cm -> nửa là 2.5cm
            
            dx, dy, dz, grip = 0.0, 0.0, 0.0, 1.0 # Mặc định cầm vật
            target_top_z = target_pos[2] + 0.025 # Mặt trên của hộp đỏ

            if state == "MOVE_TO_TARGET":
                # Bay ngang qua phía trên target
                dest = np.array([target_pos[0], target_pos[1], target_top_z + 0.15])
                diff = dest - ee_pos
                dx, dy, dz = diff * 12.0
                if np.linalg.norm(diff) < 0.015: state = "LOWER_TO_TARGET"

            elif state == "LOWER_TO_TARGET":
                # Khom tay hạ thấp để đáy vật chạm sát lên mặt trên khối đích (lấy đà lún thêm 5mm)
                dest = np.array([target_pos[0], target_pos[1], target_top_z + 0.020]) 
                diff = dest - ee_pos
                dx, dy, dz = diff * 6.0 # Hạ xuống nhẹ nhàng cho đẹp
                if bottom_z <= target_top_z + 0.002: # Đáy vật chạm đích (chỉ còn cách <= 2mm) thì mới thả
                    state = "DROP_OBJECT"
                    timer = 20

            elif state == "DROP_OBJECT":
                grip = 0.0 # MỞ GRIPPER THẢ VẬT
                timer -= 1
                if timer <= 0: state = "GO_HOME"

            elif state == "GO_HOME":
                grip = 0.0
                dest = np.array([0.55, 0.0, 0.25]) 
                diff = dest - ee_pos
                dx, dy, dz = diff * 10.0

            obs, reward, done, truncated, info = env.step([dx, dy, dz, grip])
            time.sleep(1./60.)
            if truncated or done: 
                print(">>> Thả thành công! Reset môi trường...")
                obs, info = env.reset()
                state = "MOVE_TO_TARGET"
    except KeyboardInterrupt: env.close()