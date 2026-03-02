import math
import pybullet as p


def quat_down_with_yaw(yaw: float):
    # pointing down + yaw quanh trục Z (thay yaw để đổi hướng “ngang” của kẹp)
    return p.getQuaternionFromEuler([math.pi, 0.0, yaw])


class PandaController:
    ARM_JOINTS = list(range(7))
    GRIPPER_JOINTS = [9, 10]
    HOME_Q = [0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8]

    def __init__(self, robot_id: int, grip_yaw=math.pi / 2):
        self.robot_id = robot_id

        # ✅ Tìm link “grasptarget” nếu có (tâm giữa 2 ngón). Nếu không có thì fallback 11.
        self.EE_LINK = self._find_link_index(["panda_grasptarget", "panda_hand", "panda_link8"], fallback=11)

        # ✅ Bạn chỉnh yaw ở đây: 0 hoặc pi/2 là 2 hướng phổ biến
        self.grip_yaw = grip_yaw
        self.ee_orn = quat_down_with_yaw(self.grip_yaw)

        # IK limits
        self.ll, self.ul, self.jr = [], [], []
        for j in self.ARM_JOINTS:
            info = p.getJointInfo(self.robot_id, j)
            low, high = info[8], info[9]
            self.ll.append(low)
            self.ul.append(high)
            rng = (high - low) if (high > low and abs(high - low) < 10) else 2 * math.pi
            self.jr.append(rng)

        self.target_pos = [0.55, 0.0, 0.25]

    def _find_link_index(self, name_candidates, fallback=11):
        # scan all joints: link name is jointInfo[12]
        num_j = p.getNumJoints(self.robot_id)
        for i in range(num_j):
            info = p.getJointInfo(self.robot_id, i)
            link_name = info[12].decode("utf-8")
            if link_name in name_candidates:
                return i
        return fallback

    def set_grip_yaw(self, yaw: float):
        self.grip_yaw = float(yaw)
        self.ee_orn = quat_down_with_yaw(self.grip_yaw)

    def get_ee_pose(self):
        ls = p.getLinkState(self.robot_id, self.EE_LINK)
        return ls[4], ls[5]

    def get_arm_q(self):
        return [p.getJointState(self.robot_id, j)[0] for j in self.ARM_JOINTS]

    def open_gripper(self):
        for j in self.GRIPPER_JOINTS:
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=0.04, force=80)

    def close_gripper(self):
        for j in self.GRIPPER_JOINTS:
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=0.0, force=200)

    def reset_home(self):
        for i, j in enumerate(self.ARM_JOINTS):
            p.resetJointState(self.robot_id, j, self.HOME_Q[i])
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=self.HOME_Q[i], force=300)

        self.open_gripper()

        ee_pos, _ = self.get_ee_pose()
        self.target_pos = list(ee_pos)

    def apply_action(
        self, dx, dy, dz, grip,
        arm_force=350,
        z_min=0.005, z_max=0.55,
        x_min=0.10, x_max=0.85,
        y_min=-0.60, y_max=0.60
    ):
        tx = min(x_max, max(x_min, self.target_pos[0] + dx))
        ty = min(y_max, max(y_min, self.target_pos[1] + dy))
        tz = min(z_max, max(z_min, self.target_pos[2] + dz))
        self.target_pos = [tx, ty, tz]

        rest = self.get_arm_q()
        q = p.calculateInverseKinematics(
            self.robot_id,
            self.EE_LINK,
            targetPosition=self.target_pos,
            targetOrientation=self.ee_orn,
            lowerLimits=self.ll,
            upperLimits=self.ul,
            jointRanges=self.jr,
            restPoses=rest,
            maxNumIterations=140,
            residualThreshold=1e-4
        )

        for i, j in enumerate(self.ARM_JOINTS):
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=q[i], force=arm_force)

        if float(grip) >= 0.5:
            self.close_gripper()
        else:
            self.open_gripper()