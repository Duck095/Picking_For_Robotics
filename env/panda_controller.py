from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data


@dataclass
class PandaHomePose:
    """
    Home pose mặc định cho 7 joint arm của Franka Panda.
    Có thể chỉnh lại nếu project của bạn đang dùng một home khác.
    """
    q: Tuple[float, float, float, float, float, float, float] = (
        0.00,
        -0.40,
        0.00,
        -2.20,
        0.00,
        1.85,
        0.80,
    )


class PandaController:
    """
    Bộ điều khiển robot Panda ở mức thấp.

    Mục tiêu:
    - Cho env gọi các primitive robot đơn giản, rõ ràng
    - Hỗ trợ pipeline Stage 1 mới dùng action 4 chiều:
        [dx, dy, dz, dyaw]

    Ý tưởng chính:
    - Position của EE di chuyển theo delta xyz
    - Orientation của EE giữ roll/pitch cố định
    - Chỉ thay đổi yaw khi cần

    Dùng tốt cho:
    - 1A, 1B, 1C: gần như chỉ dùng xyz
    - 1D: precision hover
    - 1E: orientation-aware pre-grasp
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

        if urdf_path is None:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            urdf_path = "franka_panda/panda.urdf"
        self.urdf_path = urdf_path

        self.robot_id: Optional[int] = None

        self.arm_joint_indices: List[int] = []
        self.gripper_joint_indices: List[int] = []

        self.arm_lower_limits: List[float] = []
        self.arm_upper_limits: List[float] = []
        self.arm_joint_ranges: List[float] = []
        self.arm_rest_poses: List[float] = []

        # Orientation mặc định kiểu "nhìn từ trên xuống"
        # Bạn có thể chỉnh nếu tool của bạn đang cần orientation khác.
        self.default_ee_orn = p.getQuaternionFromEuler([np.pi, 0.0, np.pi / 2.0])

        self._load_robot()
        self._cache_joint_info()
        self.reset_home(open_gripper=True)

    # =========================================================
    # LOAD / JOINT INFO
    # =========================================================
    def _load_robot(self) -> None:
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=self.base_position,
            baseOrientation=self.base_orientation,
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )

    def _cache_joint_info(self) -> None:
        """
        Cache:
        - arm joint indices
        - gripper joint indices
        - lower/upper/range/rest pose cho IK
        """
        assert self.robot_id is not None, "Robot chưa được load."

        num_joints = p.getNumJoints(self.robot_id)

        for joint_idx in range(num_joints):
            info = p.getJointInfo(self.robot_id, joint_idx)

            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            lower_limit = float(info[8])
            upper_limit = float(info[9])

            if joint_name.startswith("panda_joint") and joint_name not in ("panda_joint8", "panda_joint9"):
                if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                    self.arm_joint_indices.append(joint_idx)
                    self.arm_lower_limits.append(lower_limit)
                    self.arm_upper_limits.append(upper_limit)
                    self.arm_joint_ranges.append(upper_limit - lower_limit)

                    arm_pos_idx = len(self.arm_joint_indices) - 1
                    self.arm_rest_poses.append(float(self.home_pose.q[arm_pos_idx]))

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

    # =========================================================
    # BASIC HELPERS
    # =========================================================
    @staticmethod
    def wrap_angle(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def clip_to_workspace(self, pos: Sequence[float]) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float32).copy()
        pos[0] = float(np.clip(pos[0], self.workspace_x[0], self.workspace_x[1]))
        pos[1] = float(np.clip(pos[1], self.workspace_y[0], self.workspace_y[1]))
        pos[2] = float(np.clip(pos[2], self.workspace_z[0], self.workspace_z[1]))
        return pos

    def is_inside_workspace(self, pos: Sequence[float]) -> bool:
        return (
            self.workspace_x[0] <= pos[0] <= self.workspace_x[1]
            and self.workspace_y[0] <= pos[1] <= self.workspace_y[1]
            and self.workspace_z[0] <= pos[2] <= self.workspace_z[1]
        )

    # =========================================================
    # RESET / STATES
    # =========================================================
    def reset_home(self, open_gripper: bool = True) -> None:
        """
        Reset về home pose sạch và ổn định.
        """
        assert self.robot_id is not None

        for i, joint_idx in enumerate(self.arm_joint_indices):
            q = float(self.home_pose.q[i])
            p.resetJointState(self.robot_id, joint_idx, q, targetVelocity=0.0)

        if open_gripper:
            half = 0.08 / 2.0
            for joint_idx in self.gripper_joint_indices:
                p.resetJointState(self.robot_id, joint_idx, half, targetVelocity=0.0)
        else:
            for joint_idx in self.gripper_joint_indices:
                p.resetJointState(self.robot_id, joint_idx, 0.0, targetVelocity=0.0)

        self.apply_arm_joint_positions(
            joint_targets=self.home_pose.q,
            position_gain=0.08,
            velocity_gain=1.0,
            max_force=120.0,
        )

        if open_gripper:
            self.set_gripper_opening(width=0.08, force=40.0)
        else:
            self.set_gripper_opening(width=0.0, force=40.0)

        self.step_simulation(20)

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        pos : np.ndarray shape (3,)
        orn : np.ndarray shape (4,)
        """
        assert self.robot_id is not None
        link_state = p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            computeForwardKinematics=True,
        )
        pos = np.array(link_state[4], dtype=np.float32)
        orn = np.array(link_state[5], dtype=np.float32)
        return pos, orn

    def get_ee_euler(self) -> np.ndarray:
        _, ee_orn = self.get_ee_pose()
        return np.array(p.getEulerFromQuaternion(ee_orn), dtype=np.float32)

    def get_ee_yaw(self) -> float:
        return float(self.get_ee_euler()[2])

    def get_arm_joint_positions(self) -> np.ndarray:
        assert self.robot_id is not None
        states = p.getJointStates(self.robot_id, self.arm_joint_indices)
        return np.array([s[0] for s in states], dtype=np.float32)

    def get_arm_joint_velocities(self) -> np.ndarray:
        assert self.robot_id is not None
        states = p.getJointStates(self.robot_id, self.arm_joint_indices)
        return np.array([s[1] for s in states], dtype=np.float32)

    def get_full_state(self) -> dict:
        ee_pos, ee_orn = self.get_ee_pose()
        return {
            "ee_pos": ee_pos,
            "ee_orn": ee_orn,
            "ee_euler": self.get_ee_euler(),
            "ee_yaw": self.get_ee_yaw(),
            "joint_q": self.get_arm_joint_positions(),
            "joint_dq": self.get_arm_joint_velocities(),
        }

    # =========================================================
    # ORIENTATION HELPERS
    # =========================================================
    def get_default_ee_euler(self) -> Tuple[float, float, float]:
        """
        Lấy roll/pitch/yaw mặc định từ default_ee_orn.
        """
        e = p.getEulerFromQuaternion(self.default_ee_orn)
        return float(e[0]), float(e[1]), float(e[2])

    def build_ee_quat_from_yaw(self, yaw: float) -> np.ndarray:
        """
        Xây quaternion của EE với:
        - roll giữ như default
        - pitch giữ như default
        - yaw theo input
        """
        default_roll, default_pitch, _ = self.get_default_ee_euler()
        quat = p.getQuaternionFromEuler([default_roll, default_pitch, float(yaw)])
        return np.array(quat, dtype=np.float32)

    def build_ee_quat_from_rpy(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        quat = p.getQuaternionFromEuler([float(roll), float(pitch), float(yaw)])
        return np.array(quat, dtype=np.float32)

    # =========================================================
    # IK / JOINT CONTROL
    # =========================================================
    def solve_ik(
        self,
        target_pos: Sequence[float],
        target_orn: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """
        Giải IK:
        target EE pose -> 7 arm joint targets
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

        return np.array(ik_solution[: len(self.arm_joint_indices)], dtype=np.float32)

    def apply_arm_joint_positions(
        self,
        joint_targets: Sequence[float],
        position_gain: float = 0.08,
        velocity_gain: float = 1.0,
        max_force: float = 120.0,
    ) -> None:
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
        for _ in range(int(substeps)):
            p.stepSimulation()

    # =========================================================
    # EE MOTION
    # =========================================================
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
        Move EE tới target pose bằng IK + POSITION_CONTROL.
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

    def move_ee_to_with_yaw(
        self,
        target_pos: Sequence[float],
        target_yaw: float,
        substeps: int = 6,
        position_gain: float = 0.08,
        velocity_gain: float = 1.0,
        max_force: float = 120.0,
    ) -> np.ndarray:
        """
        Move EE tới target_pos với orientation được build từ target_yaw.
        roll/pitch giữ cố định.
        """
        target_quat = self.build_ee_quat_from_yaw(target_yaw)
        return self.move_ee_to(
            target_pos=target_pos,
            target_orn=target_quat,
            substeps=substeps,
            position_gain=position_gain,
            velocity_gain=velocity_gain,
            max_force=max_force,
        )

    def move_ee_delta(
        self,
        dx: float,
        dy: float,
        dz: float,
        dyaw: float = 0.0,
        target_orn: Optional[Sequence[float]] = None,
        substeps: int = 6,
        position_gain: float = 0.08,
        velocity_gain: float = 1.0,
        max_force: float = 120.0,
    ) -> np.ndarray:
        """
        Primitive quan trọng nhất cho RL.

        Nếu target_orn is None:
            - target_pos = ee_pos + [dx, dy, dz]
            - target_yaw = current_yaw + dyaw
            - roll/pitch giữ cố định theo default

        Nếu target_orn được truyền vào:
            - dùng target_orn đó luôn
            - dyaw bị bỏ qua
        """
        ee_pos, ee_orn = self.get_ee_pose()

        target_pos = np.array(
            [ee_pos[0] + dx, ee_pos[1] + dy, ee_pos[2] + dz],
            dtype=np.float32,
        )
        target_pos = self.clip_to_workspace(target_pos)

        if target_orn is None:
            current_yaw = float(p.getEulerFromQuaternion(ee_orn)[2])
            target_yaw = self.wrap_angle(current_yaw + float(dyaw))
            target_quat = self.build_ee_quat_from_yaw(target_yaw)
        else:
            target_quat = np.asarray(target_orn, dtype=np.float32)

        return self.move_ee_to(
            target_pos=target_pos,
            target_orn=target_quat,
            substeps=substeps,
            position_gain=position_gain,
            velocity_gain=velocity_gain,
            max_force=max_force,
        )

    def move_ee_delta_with_target_yaw(
        self,
        dx: float,
        dy: float,
        dz: float,
        target_yaw: float,
        yaw_blend: float = 1.0,
        substeps: int = 6,
        position_gain: float = 0.08,
        velocity_gain: float = 1.0,
        max_force: float = 120.0,
    ) -> np.ndarray:
        """
        Hữu ích nếu sau này bạn muốn:
        - policy vẫn chỉ điều khiển xyz
        - còn yaw được controller kéo dần về target_yaw

        yaw_blend:
            0.0 -> giữ yaw hiện tại
            1.0 -> nhảy thẳng về target_yaw
            0.0~1.0 -> blend dần
        """
        ee_pos, ee_orn = self.get_ee_pose()
        current_yaw = float(p.getEulerFromQuaternion(ee_orn)[2])

        target_pos = np.array(
            [ee_pos[0] + dx, ee_pos[1] + dy, ee_pos[2] + dz],
            dtype=np.float32,
        )
        target_pos = self.clip_to_workspace(target_pos)

        yaw_blend = float(np.clip(yaw_blend, 0.0, 1.0))
        yaw_delta = self.wrap_angle(float(target_yaw) - current_yaw)
        blended_yaw = self.wrap_angle(current_yaw + yaw_blend * yaw_delta)

        target_quat = self.build_ee_quat_from_yaw(blended_yaw)

        return self.move_ee_to(
            target_pos=target_pos,
            target_orn=target_quat,
            substeps=substeps,
            position_gain=position_gain,
            velocity_gain=velocity_gain,
            max_force=max_force,
        )

    # =========================================================
    # GRIPPER
    # =========================================================
    def set_gripper_opening(self, width: float = 0.08, force: float = 40.0) -> None:
        """
        width = tổng độ mở của 2 ngón
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
        self.set_gripper_opening(width=0.08, force=force)
        self.step_simulation(substeps)

    def close_gripper(self, force: float = 80.0, substeps: int = 20) -> None:
        self.set_gripper_opening(width=0.0, force=force)
        self.step_simulation(substeps)

    # =========================================================
    # DEBUG HELPERS
    # =========================================================
    def draw_ee_axes(self, axis_len: float = 0.08, life_time: float = 0.1) -> None:
        ee_pos, ee_orn = self.get_ee_pose()
        rot = np.array(p.getMatrixFromQuaternion(ee_orn), dtype=np.float32).reshape(3, 3)

        origin = ee_pos
        x_axis = origin + rot[:, 0] * axis_len
        y_axis = origin + rot[:, 1] * axis_len
        z_axis = origin + rot[:, 2] * axis_len

        p.addUserDebugLine(origin, x_axis, [1, 0, 0], lineWidth=2, lifeTime=life_time)
        p.addUserDebugLine(origin, y_axis, [0, 1, 0], lineWidth=2, lifeTime=life_time)
        p.addUserDebugLine(origin, z_axis, [0, 0, 1], lineWidth=2, lifeTime=life_time)

    def draw_target_pose(
        self,
        target_pos: Sequence[float],
        target_yaw: Optional[float] = None,
        axis_len: float = 0.06,
        life_time: float = 0.1,
    ) -> None:
        """
        Debug target pose nếu dùng GUI.
        """
        pos = np.asarray(target_pos, dtype=np.float32)

        if target_yaw is None:
            quat = np.asarray(self.default_ee_orn, dtype=np.float32)
        else:
            quat = self.build_ee_quat_from_yaw(float(target_yaw))

        rot = np.array(p.getMatrixFromQuaternion(quat), dtype=np.float32).reshape(3, 3)

        x_axis = pos + rot[:, 0] * axis_len
        y_axis = pos + rot[:, 1] * axis_len
        z_axis = pos + rot[:, 2] * axis_len

        p.addUserDebugLine(pos, x_axis, [1, 0, 0], lineWidth=2, lifeTime=life_time)
        p.addUserDebugLine(pos, y_axis, [0, 1, 0], lineWidth=2, lifeTime=life_time)
        p.addUserDebugLine(pos, z_axis, [0, 0, 1], lineWidth=2, lifeTime=life_time)

    # =========================================================
    # CLEANUP
    # =========================================================
    def disconnect(self) -> None:
        try:
            p.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 120.0)

    p.loadURDF("plane.urdf")

    table_half = [0.45, 0.60, 0.02]
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half)
    vis_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=table_half,
        rgbaColor=[0.75, 0.75, 0.75, 1.0],
    )
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[0.55, 0.0, -0.02],
    )

    robot = PandaController(client_id=client_id, use_gui=True)

    for _ in range(60):
        robot.move_ee_delta(dx=0.002, dy=0.0, dz=0.0, dyaw=0.0, substeps=4)

    for _ in range(60):
        robot.move_ee_delta(dx=0.0, dy=0.002, dz=0.0, dyaw=0.0, substeps=4)

    for _ in range(60):
        robot.move_ee_delta(dx=0.0, dy=0.0, dz=-0.001, dyaw=0.0, substeps=4)

    for _ in range(80):
        robot.move_ee_delta(dx=0.0, dy=0.0, dz=0.0, dyaw=0.01, substeps=4)

    print("EE pose:", robot.get_ee_pose()[0])
    print("EE euler:", robot.get_ee_euler())

    while True:
        robot.draw_ee_axes()
        p.stepSimulation()