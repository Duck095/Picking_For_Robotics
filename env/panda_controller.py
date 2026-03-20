# env/panda_controller.py
import math
from typing import Iterable, Optional
import pybullet as p


def quat_down_with_yaw(yaw: float, cid=None):
    return p.getQuaternionFromEuler([math.pi, 0.0, float(yaw)], physicsClientId=cid)


class PandaController:
    ARM_JOINTS = list(range(7))
    GRIPPER_JOINTS = [9, 10]
    HOME_Q = [0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8]

    # ưu tiên chung cho nhiều robot
    DEFAULT_EE_NAME_PRIORITY = [
        "grasptarget",
        "grasp_target",
        "grasp",
        "tcp",
        "tool0",
        "tool",
        "ee_link",
        "end_effector",
        "end-effector",
        "eef",
        "hand",
        "gripper",
        "flange",
        "wrist",
        "link8",
    ]

    # dùng để tương thích model cũ của bạn
    LEGACY_PANDA_PRIORITY = [
        "panda_link8",
        "panda_hand",
        "panda_grasptarget",
    ]

    # dùng để train mới chuẩn grasp
    GRASP_PANDA_PRIORITY = [
        "panda_grasptarget",
        "panda_hand",
        "panda_link8",
    ]

    def __init__(
        self,
        robot_id: int,
        physics_client_id=None,
        grip_yaw=math.pi / 2,
        ee_mode: str = "grasp",   # "grasp" | "legacy" | "auto"
        verbose: bool = False,
    ):
        self.robot_id = int(robot_id)
        self.cid = physics_client_id
        self.verbose = bool(verbose)

        self.grip_yaw = float(grip_yaw)
        self.ee_orn = quat_down_with_yaw(self.grip_yaw, cid=self.cid)

        self.ee_mode = str(ee_mode).lower().strip()
        self.EE_LINK = self._resolve_ee_link()

        self.ll, self.ul, self.jr = [], [], []
        for j in self.ARM_JOINTS:
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.cid)
            low, high = info[8], info[9]
            self.ll.append(low)
            self.ul.append(high)
            rng = (high - low) if (high > low and abs(high - low) < 10) else 2 * math.pi
            self.jr.append(rng)

        self.target_pos = [0.55, 0.0, 0.25]

        if self.verbose:
            joint_info = p.getJointInfo(self.robot_id, self.EE_LINK, physicsClientId=self.cid)
            joint_name = joint_info[1].decode("utf-8")
            link_name = joint_info[12].decode("utf-8")
            print(f"[EE SELECT] mode={self.ee_mode} -> index={self.EE_LINK} | joint={joint_name} | link={link_name}")

    def _collect_joint_and_link_names(self):
        name_to_idx = {}
        n = p.getNumJoints(self.robot_id, physicsClientId=self.cid)

        for j in range(n):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.cid)
            joint_name = info[1].decode("utf-8").lower()
            link_name = info[12].decode("utf-8").lower() if info[12] else ""

            name_to_idx[joint_name] = j
            if link_name:
                name_to_idx[link_name] = j

        return name_to_idx

    def _find_link_index_by_priority(
        self,
        priorities: Iterable[str],
        fallback: Optional[int] = None,
        substring_match: bool = True,
    ) -> int:
        name_to_idx = self._collect_joint_and_link_names()

        # 1) exact match theo đúng thứ tự ưu tiên
        for name in priorities:
            key = name.lower()
            if key in name_to_idx:
                return int(name_to_idx[key])

        # 2) substring match theo đúng thứ tự ưu tiên
        if substring_match:
            for name in priorities:
                key = name.lower()
                for known_name, idx in name_to_idx.items():
                    if key in known_name:
                        return int(idx)

        # 3) fallback
        if fallback is not None:
            return int(fallback)

        n = p.getNumJoints(self.robot_id, physicsClientId=self.cid)
        return int(n - 1)

    def _resolve_ee_link(self) -> int:
        if self.ee_mode == "legacy":
            return self._find_link_index_by_priority(
                self.LEGACY_PANDA_PRIORITY,
                fallback=7,
                substring_match=False,
            )

        if self.ee_mode == "grasp":
            return self._find_link_index_by_priority(
                self.GRASP_PANDA_PRIORITY,
                fallback=11,
                substring_match=False,
            )

        # auto/general mode cho robot khác
        return self._find_link_index_by_priority(
            self.DEFAULT_EE_NAME_PRIORITY,
            fallback=None,
            substring_match=True,
        )

    def print_all_links(self):
        n = p.getNumJoints(self.robot_id, physicsClientId=self.cid)
        print(f"Num joints: {n}")
        for j in range(n):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.cid)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8") if info[12] else ""
            print(f"{j:02d} joint={joint_name:<24} link={link_name}")

    def set_grip_yaw(self, yaw: float):
        self.grip_yaw = float(yaw)
        self.ee_orn = quat_down_with_yaw(self.grip_yaw, cid=self.cid)

    def reset_home(self):
        for i, j in enumerate(self.ARM_JOINTS):
            p.resetJointState(self.robot_id, j, self.HOME_Q[i], physicsClientId=self.cid)
            p.setJointMotorControl2(
                self.robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=self.HOME_Q[i],
                force=300,
                physicsClientId=self.cid,
            )

        self.open_gripper()
        ee_pos, _ = self.get_ee_pose()
        self.target_pos = [float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])]

    def get_ee_pose(self):
        ls = p.getLinkState(self.robot_id, self.EE_LINK, physicsClientId=self.cid)
        return ls[4], ls[5]

    def get_arm_q(self):
        return [
            p.getJointState(self.robot_id, j, physicsClientId=self.cid)[0]
            for j in self.ARM_JOINTS
        ]

    def open_gripper(self):
        for j in self.GRIPPER_JOINTS:
            p.setJointMotorControl2(
                self.robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=0.04,
                force=80,
                physicsClientId=self.cid,
            )

    def close_gripper(self, width=0.024):
        for j in self.GRIPPER_JOINTS:
            p.setJointMotorControl2(
                self.robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=width,
                force=200,
                physicsClientId=self.cid,
            )

    def apply_delta_action(self, dx, dy, dz, x_range, y_range, z_range, arm_force=350):
        tx = min(x_range[1], max(x_range[0], self.target_pos[0] + float(dx)))
        ty = min(y_range[1], max(y_range[0], self.target_pos[1] + float(dy)))
        tz = min(z_range[1], max(z_range[0], self.target_pos[2] + float(dz)))
        self.target_pos = [tx, ty, tz]

        current_q = self.get_arm_q()
        rest = [0.7 * current_q[i] + 0.3 * self.HOME_Q[i] for i in range(7)]

        q = p.calculateInverseKinematics(
            self.robot_id,
            self.EE_LINK,
            targetPosition=self.target_pos,
            targetOrientation=self.ee_orn,
            lowerLimits=self.ll,
            upperLimits=self.ul,
            jointRanges=self.jr,
            restPoses=rest,
            maxNumIterations=220,
            residualThreshold=1e-4,
            physicsClientId=self.cid,
        )

        for i, j in enumerate(self.ARM_JOINTS):
            p.setJointMotorControl2(
                self.robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=q[i],
                force=arm_force,
                physicsClientId=self.cid,
            )