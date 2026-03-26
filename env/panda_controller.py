from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data


# ============================================================
# PandaController
# ------------------------------------------------------------
# File này chịu trách nhiệm điều khiển robot Panda ở mức thấp:
#   - load robot
#   - reset home
#   - đọc EE pose
#   - đọc joint state
#   - IK
#   - move EE theo delta
#
# File này KHÔNG chịu trách nhiệm:
#   - reward
#   - reset object
#   - success / truncate
#   - episode logic
# ============================================================


@dataclass
class PandaHomePose:
    """
    Home pose mặc định cho 7 joint arm của Franka Panda.

    Bạn có thể thay bộ này theo demo / pose đẹp của project bạn.
    """
    q: Tuple[float, float, float, float, float, float, float] = (
        0.00,
        -0.40,
        0.00,
        -2.15,
        0.00,
        2.00,
        0.80,
    )


class PandaController:
    """
    Bộ điều khiển mức thấp cho Panda.

    Ý tưởng thiết kế:
    - Env sẽ gọi controller để robot di chuyển.
    - Controller chỉ lo phần robot / IK / joint control.
    - Env không nên nhồi hết robot logic vào bên trong.

    Các hàm quan trọng cho Stage 1:
    - reset_home()
    - get_ee_pose()
    - get_arm_joint_positions()
    - get_arm_joint_velocities()
    - move_ee_delta(dx, dy, dz, substeps)
    - clip_to_workspace(pos)

    Ngoài ra mình để sẵn một số hàm hữu ích cho Stage 2+:
    - open_gripper()
    - close_gripper()
    - move_ee_to(...)
    """

    def __init__(
        self,
        client_id: Optional[int] = None,
        use_gui: bool = False,
        urdf_path: Optional[str] = None,
        base_position: Sequence[float] = (0.0, 0.0, 0.0),
        base_orientation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        time_step: float = 1.0 / 120.0,
        ee_link_index: int = 11,
        workspace: Optional[object] = None,
        workspace_x: Tuple[float, float] = (0.30, 0.75),
        workspace_y: Tuple[float, float] = (-0.10, 0.25),
        workspace_z: Tuple[float, float] = (0.01, 0.55),
        home_pose: Optional[PandaHomePose] = None,
    ) -> None:
        """
        Parameters
        ----------
        client_id:
            ID của PyBullet client. Nếu None thì dùng client hiện tại.
        use_gui:
            Chỉ để debug/log, không bắt buộc.
        urdf_path:
            Đường dẫn URDF robot. Nếu None thì dùng Panda mặc định từ pybullet_data.
        base_position/base_orientation:
            Pose gốc của robot trong world.
        time_step:
            Physics timestep để step mô phỏng.
        ee_link_index:
            Link index của end-effector.
            Trong nhiều setup Panda, EE / hand thường quanh 11.
            Nếu project bạn dùng link khác, đổi tại đây.
        workspace_x/y/z:
            Giới hạn an toàn cho target EE.
        home_pose:
            Pose home 7 joint.
        """
        self.client_id = client_id
        self.use_gui = use_gui
        self.time_step = float(time_step)

        self.ee_link_index = int(ee_link_index)

        if workspace is not None:
            self.workspace_x = tuple(getattr(workspace, "x_range", workspace_x))
            self.workspace_y = tuple(getattr(workspace, "y_range", workspace_y))
            self.workspace_z = tuple(getattr(workspace, "z_range", workspace_z))
        else:
            self.workspace_x = workspace_x
            self.workspace_y = workspace_y
            self.workspace_z = workspace_z

        self.base_position = tuple(base_position)
        self.base_orientation = tuple(base_orientation)

        self.home_pose = home_pose if home_pose is not None else PandaHomePose()

        # ----------------------------------------------------
        # URDF
        # ----------------------------------------------------
        if urdf_path is None:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            urdf_path = "franka_panda/panda.urdf"
        self.urdf_path = urdf_path

        # ----------------------------------------------------
        # Robot body id
        # ----------------------------------------------------
        self.robot_id: Optional[int] = None

        # ----------------------------------------------------
        # Joint index mapping
        # ----------------------------------------------------
        # Panda arm có 7 joint quay
        self.arm_joint_indices: List[int] = []

        # Panda gripper thường là 2 finger joint
        self.gripper_joint_indices: List[int] = []

        # Joint lower / upper limits cho arm
        self.arm_lower_limits: List[float] = []
        self.arm_upper_limits: List[float] = []
        self.arm_joint_ranges: List[float] = []
        self.arm_rest_poses: List[float] = []

        # Default orientation cho EE khi làm task pick từ trên xuống
        # Bạn có thể chỉnh nếu orientation của demo khác.
        self.default_ee_orn = p.getQuaternionFromEuler([np.pi, 0.0, np.pi / 2.0])

        self._load_robot()
        self._cache_joint_info()
        self.reset_home()

    # ========================================================
    # LOAD / JOINT INFO
    # ========================================================
    def _load_robot(self) -> None:
        """Load URDF robot Panda."""
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=self.base_position,
            baseOrientation=self.base_orientation,
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )

    def _cache_joint_info(self) -> None:
        """
        Đọc thông tin joint để biết:
        - arm joint nào
        - gripper joint nào
        - limit của joint
        """
        assert self.robot_id is not None, "Robot chưa được load."

        num_joints = p.getNumJoints(self.robot_id)

        for joint_idx in range(num_joints):
            info = p.getJointInfo(self.robot_id, joint_idx)

            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            lower_limit = float(info[8])
            upper_limit = float(info[9])

            # Arm joints của Panda thường tên panda_joint1..7
            if joint_name.startswith("panda_joint") and joint_name not in ("panda_joint8", "panda_joint9"):
                if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                    self.arm_joint_indices.append(joint_idx)
                    self.arm_lower_limits.append(lower_limit)
                    self.arm_upper_limits.append(upper_limit)
                    self.arm_joint_ranges.append(upper_limit - lower_limit)
                    # rest pose mặc định lấy từ home pose
                    arm_pos_idx = len(self.arm_joint_indices) - 1
                    self.arm_rest_poses.append(self.home_pose.q[arm_pos_idx])

            # Finger joints
            if joint_name in ("panda_finger_joint1", "panda_finger_joint2"):
                self.gripper_joint_indices.append(joint_idx)

        if len(self.arm_joint_indices) != 7:
            raise RuntimeError(
                f"Không tìm đủ 7 arm joints của Panda. Tìm được: {self.arm_joint_indices}"
            )

        if len(self.gripper_joint_indices) != 2:
            raise RuntimeError(
                f"Không tìm đủ 2 gripper joints của Panda. Tìm được: {self.gripper_joint_indices}"
            )

    # ========================================================
    # RESET / BASIC STATE
    # ========================================================
    def reset_home(self, open_gripper: bool = True) -> None:
        """
        Reset robot về home pose ổn định.

        Lý do phải làm kỹ:
        - resetJointState chỉ đổi state tức thời
        - nhưng motor target cũ từ episode trước vẫn còn
        - nếu step simulation ngay, arm sẽ bị kéo lệch khỏi home
        """
        assert self.robot_id is not None

        # 1) Reset 7 arm joints về đúng home pose
        for i, joint_idx in enumerate(self.arm_joint_indices):
            q = float(self.home_pose.q[i])
            p.resetJointState(self.robot_id, joint_idx, q, targetVelocity=0.0)

        # 2) Reset finger joints về open state để state sạch
        #    Dùng resetJointState luôn cho chắc trước khi set motor control
        if open_gripper:
            half = 0.08 / 2.0
            for joint_idx in self.gripper_joint_indices:
                p.resetJointState(self.robot_id, joint_idx, half, targetVelocity=0.0)
        else:
            for joint_idx in self.gripper_joint_indices:
                p.resetJointState(self.robot_id, joint_idx, 0.0, targetVelocity=0.0)

        # 3) QUAN TRỌNG NHẤT:
        #    Set lại arm motor target = home pose
        #    để xóa ảnh hưởng target cũ từ episode trước
        self.apply_arm_joint_positions(
            joint_targets=self.home_pose.q,
            position_gain=0.08,
            velocity_gain=1.0,
            max_force=120.0,
        )

        # 4) Set lại gripper motor target cho khớp với state reset
        if open_gripper:
            self.set_gripper_opening(width=0.08, force=40.0)
        else:
            self.set_gripper_opening(width=0.0, force=40.0)

        # 5) Step vài frame để robot ổn định ở home
        self.step_simulation(20)

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lấy pose hiện tại của end-effector.

        Returns
        -------
        pos : np.ndarray shape (3,)
        orn : np.ndarray shape (4,)
        """
        assert self.robot_id is not None
        link_state = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True)
        pos = np.array(link_state[4], dtype=np.float32)  # worldLinkFramePosition
        orn = np.array(link_state[5], dtype=np.float32)  # worldLinkFrameOrientation
        return pos, orn

    def get_arm_joint_positions(self) -> np.ndarray:
        """Lấy q của 7 joint arm."""
        assert self.robot_id is not None
        states = p.getJointStates(self.robot_id, self.arm_joint_indices)
        q = np.array([s[0] for s in states], dtype=np.float32)
        return q

    def get_arm_joint_velocities(self) -> np.ndarray:
        """Lấy dq của 7 joint arm."""
        assert self.robot_id is not None
        states = p.getJointStates(self.robot_id, self.arm_joint_indices)
        dq = np.array([s[1] for s in states], dtype=np.float32)
        return dq

    def get_full_state(self) -> dict:
        """
        Hàm tiện ích để debug nhanh.
        """
        ee_pos, ee_orn = self.get_ee_pose()
        return {
            "ee_pos": ee_pos,
            "ee_orn": ee_orn,
            "joint_q": self.get_arm_joint_positions(),
            "joint_dq": self.get_arm_joint_velocities(),
        }

    # ========================================================
    # WORKSPACE HELPERS
    # ========================================================
    def clip_to_workspace(self, pos: Sequence[float]) -> np.ndarray:
        """
        Clip một target EE vào workspace an toàn.
        """
        pos = np.asarray(pos, dtype=np.float32).copy()
        pos[0] = float(np.clip(pos[0], self.workspace_x[0], self.workspace_x[1]))
        pos[1] = float(np.clip(pos[1], self.workspace_y[0], self.workspace_y[1]))
        pos[2] = float(np.clip(pos[2], self.workspace_z[0], self.workspace_z[1]))
        return pos

    def is_inside_workspace(self, pos: Sequence[float]) -> bool:
        """
        Kiểm tra một điểm có nằm trong workspace hay không.
        """
        return (
            self.workspace_x[0] <= pos[0] <= self.workspace_x[1]
            and self.workspace_y[0] <= pos[1] <= self.workspace_y[1]
            and self.workspace_z[0] <= pos[2] <= self.workspace_z[1]
        )

    # ========================================================
    # IK / JOINT CONTROL
    # ========================================================
    def solve_ik(
        self,
        target_pos: Sequence[float],
        target_orn: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """
        Giải IK từ target EE pose -> 7 arm joint targets.

        Parameters
        ----------
        target_pos:
            Vị trí đích của EE trong world.
        target_orn:
            Orientation đích của EE.
            Nếu None thì dùng default_ee_orn.

        Returns
        -------
        np.ndarray shape (7,)
            7 joint targets cho arm.
        """
        assert self.robot_id is not None

        if target_orn is None:
            target_orn = self.default_ee_orn

        target_pos = self.clip_to_workspace(target_pos)

        ik_solution = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_link_index,
            targetPosition=target_pos,
            targetOrientation=target_orn,
            lowerLimits=self.arm_lower_limits,
            upperLimits=self.arm_upper_limits,
            jointRanges=self.arm_joint_ranges,
            restPoses=self.arm_rest_poses,
            maxNumIterations=100,
            residualThreshold=1e-4,
        )

        # PyBullet có thể trả nhiều joint hơn; ta chỉ lấy 7 joint arm đầu
        joint_targets = np.array(ik_solution[: len(self.arm_joint_indices)], dtype=np.float32)
        return joint_targets

    def apply_arm_joint_positions(
        self,
        joint_targets: Sequence[float],
        position_gain: float = 0.08,
        velocity_gain: float = 1.0,
        max_force: float = 120.0,
    ) -> None:
        """
        Gửi lệnh POSITION_CONTROL cho 7 arm joints.
        """
        assert self.robot_id is not None

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.arm_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=list(joint_targets),
            positionGains=[position_gain] * len(self.arm_joint_indices),
            velocityGains=[velocity_gain] * len(self.arm_joint_indices),
            forces=[max_force] * len(self.arm_joint_indices),
        )

    def step_simulation(self, substeps: int = 1) -> None:
        """
        Step physics N lần.
        """
        for _ in range(int(substeps)):
            p.stepSimulation()

    # ========================================================
    # HIGH-LEVEL EE MOTION
    # ========================================================
    def move_ee_to(
        self,
        target_pos: Sequence[float],
        target_orn: Optional[Sequence[float]] = None,
        substeps: int = 6,
        position_gain: float = 0.08,
        velocity_gain: float = 1.0,
        max_force: float = 120.0,
    ) -> np.ndarray:
        """
        Đi EE tới target_pos bằng IK + joint position control.

        Đây là primitive rất quan trọng.
        Env có thể gọi hàm này nếu muốn robot đi đến 1 điểm cụ thể.

        Returns
        -------
        np.ndarray
            joint_targets đã gửi xuống robot
        """
        joint_targets = self.solve_ik(target_pos=target_pos, target_orn=target_orn)
        self.apply_arm_joint_positions(
            joint_targets=joint_targets,
            position_gain=position_gain,
            velocity_gain=velocity_gain,
            max_force=max_force,
        )
        self.step_simulation(substeps=substeps)
        return joint_targets

    def move_ee_delta(
        self,
        dx: float,
        dy: float,
        dz: float,
        target_orn: Optional[Sequence[float]] = None,
        substeps: int = 6,
        position_gain: float = 0.08,
        velocity_gain: float = 1.0,
        max_force: float = 120.0,
    ) -> np.ndarray:
        """
        Di chuyển EE theo delta trong world frame.

        Đây là hàm phù hợp nhất cho Stage 1.
        Agent sẽ output action -> env scale thành dx,dy,dz -> gọi hàm này.

        Returns
        -------
        np.ndarray
            joint_targets của arm
        """
        ee_pos, ee_orn = self.get_ee_pose()
        curr_orn = ee_orn if target_orn is None else target_orn

        target_pos = np.array(
            [ee_pos[0] + dx, ee_pos[1] + dy, ee_pos[2] + dz],
            dtype=np.float32,
        )
        target_pos = self.clip_to_workspace(target_pos)

        return self.move_ee_to(
            target_pos=target_pos,
            target_orn=curr_orn,
            substeps=substeps,
            position_gain=position_gain,
            velocity_gain=velocity_gain,
            max_force=max_force,
        )

    # ========================================================
    # GRIPPER
    # --------------------------------------------------------
    # Stage 1 chưa dùng nhiều, nhưng mình để sẵn cho Stage 2.
    # ========================================================
    def set_gripper_opening(self, width: float = 0.08, force: float = 40.0) -> None:
        """
        Đặt độ mở gripper.

        width:
            Khoảng mở tổng của 2 ngón.
            Mỗi finger joint sẽ đi khoảng width / 2.

        Lưu ý:
            Panda finger joint thường là prismatic.
        """
        assert self.robot_id is not None

        half = max(0.0, float(width) / 2.0)
        target_positions = [half, half]

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.gripper_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
            forces=[force, force],
        )

    def open_gripper(self, force: float = 40.0, substeps: int = 20) -> None:
        """Mở gripper."""
        self.set_gripper_opening(width=0.08, force=force)
        self.step_simulation(substeps)

    def close_gripper(self, force: float = 80.0, substeps: int = 20) -> None:
        """Đóng gripper."""
        self.set_gripper_opening(width=0.0, force=force)
        self.step_simulation(substeps)

    # ========================================================
    # DEBUG HELPERS
    # ========================================================
    def draw_ee_axes(self, axis_len: float = 0.08, life_time: float = 0.1) -> None:
        """
        Vẽ trục tọa độ tại EE để debug orientation.
        """
        ee_pos, ee_orn = self.get_ee_pose()
        rot = np.array(p.getMatrixFromQuaternion(ee_orn), dtype=np.float32).reshape(3, 3)

        origin = ee_pos
        x_axis = origin + rot[:, 0] * axis_len
        y_axis = origin + rot[:, 1] * axis_len
        z_axis = origin + rot[:, 2] * axis_len

        p.addUserDebugLine(origin, x_axis, [1, 0, 0], lineWidth=2, lifeTime=life_time)
        p.addUserDebugLine(origin, y_axis, [0, 1, 0], lineWidth=2, lifeTime=life_time)
        p.addUserDebugLine(origin, z_axis, [0, 0, 1], lineWidth=2, lifeTime=life_time)

    # ========================================================
    # CLEANUP
    # ========================================================
    def disconnect(self) -> None:
        """
        Ngắt kết nối PyBullet.
        Chỉ dùng nếu controller là bên sở hữu phiên pybullet.
        """
        try:
            p.disconnect()
        except Exception:
            pass


# ============================================================
# TEST NHANH
# ------------------------------------------------------------
# Chạy file này trực tiếp để test robot có load và move được không.
# ============================================================
if __name__ == "__main__":
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 120.0)

    p.loadURDF("plane.urdf")

    # Tạo bàn đơn giản
    table_half = [0.45, 0.60, 0.02]
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half)
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=table_half, rgbaColor=[0.75, 0.75, 0.75, 1.0])
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[0.55, 0.0, -0.02],
    )

    robot = PandaController(client_id=client_id, use_gui=True)

    # Test move delta
    for _ in range(60):
        robot.move_ee_delta(dx=0.002, dy=0.0, dz=0.0, substeps=4)

    for _ in range(60):
        robot.move_ee_delta(dx=0.0, dy=0.002, dz=0.0, substeps=4)

    for _ in range(60):
        robot.move_ee_delta(dx=0.0, dy=0.0, dz=-0.001, substeps=4)

    print("EE pose:", robot.get_ee_pose()[0])

    while True:
        p.stepSimulation()