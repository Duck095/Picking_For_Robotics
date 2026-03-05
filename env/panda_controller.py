# env/panda_controller.py
import math
import pybullet as p


def quat_down_with_yaw(yaw: float, cid=None):
    return p.getQuaternionFromEuler([math.pi, 0.0, float(yaw)], physicsClientId=cid)


class PandaController:
    ARM_JOINTS = list(range(7))
    GRIPPER_JOINTS = [9, 10]
    HOME_Q = [0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8]

    def __init__(self, robot_id: int, physics_client_id=None, grip_yaw=math.pi / 2):
        self.robot_id = int(robot_id)
        self.cid = physics_client_id

        self.grip_yaw = float(grip_yaw)
        self.ee_orn = quat_down_with_yaw(self.grip_yaw, cid=self.cid)

        # ✅ tự tìm EE link theo tên (đỡ hardcode 11)
        self.EE_LINK = self._find_link_index(
            ["panda_grasptarget", "panda_hand", "panda_link8"],
            fallback=11,
        )

        self.ll, self.ul, self.jr = [], [], []
        for j in self.ARM_JOINTS:
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.cid)
            low, high = info[8], info[9]
            self.ll.append(low)
            self.ul.append(high)
            rng = (high - low) if (high > low and abs(high - low) < 10) else 2 * math.pi
            self.jr.append(rng)

        self.target_pos = [0.55, 0.0, 0.25]

    def _find_link_index(self, names, fallback=11):
        n = p.getNumJoints(self.robot_id, physicsClientId=self.cid)
        targets = [s.lower() for s in names]
        for j in range(n):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.cid)
            joint_name = info[1].decode("utf-8").lower()
            link_name = (info[12].decode("utf-8").lower() if info[12] else "")
            if joint_name in targets or link_name in targets:
                return j
        return int(fallback)

    def reset_home(self):
        for i, j in enumerate(self.ARM_JOINTS):
            p.resetJointState(self.robot_id, j, self.HOME_Q[i], physicsClientId=self.cid)
            p.setJointMotorControl2(
                self.robot_id, j,
                p.POSITION_CONTROL,
                targetPosition=self.HOME_Q[i],
                force=300,
                physicsClientId=self.cid
            )
        self.open_gripper()
        ee_pos, _ = self.get_ee_pose()
        self.target_pos = list(ee_pos)

    def get_ee_pose(self):
        ls = p.getLinkState(self.robot_id, self.EE_LINK, physicsClientId=self.cid)
        return ls[4], ls[5]

    def get_arm_q(self):
        return [p.getJointState(self.robot_id, j, physicsClientId=self.cid)[0] for j in self.ARM_JOINTS]

    def open_gripper(self):
        for j in self.GRIPPER_JOINTS:
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=0.04, force=80, physicsClientId=self.cid)

    def close_gripper(self):
        for j in self.GRIPPER_JOINTS:
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=0.0, force=200, physicsClientId=self.cid)

    def apply_delta_action(self, dx, dy, dz, x_range, y_range, z_range, arm_force=350):
        tx = min(x_range[1], max(x_range[0], self.target_pos[0] + float(dx)))
        ty = min(y_range[1], max(y_range[0], self.target_pos[1] + float(dy)))
        tz = min(z_range[1], max(z_range[0], self.target_pos[2] + float(dz)))
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
            maxNumIterations=160,
            residualThreshold=1e-4,
            physicsClientId=self.cid
        )

        for i, j in enumerate(self.ARM_JOINTS):
            p.setJointMotorControl2(
                self.robot_id, j,
                p.POSITION_CONTROL,
                targetPosition=q[i],
                force=arm_force,
                physicsClientId=self.cid
            )