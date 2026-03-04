import math
import pybullet as p


class SimpleAttachGrasp:
    """
    Stage-1 helper: nếu gripper đang đóng và EE đủ gần object -> attach constraint.
    """
    def __init__(self, ee_link=11, max_dist=0.09, max_force=2500,
                 close_th=0.5, open_th=0.5, physics_client_id=None):
        self.ee_link = ee_link
        self.max_dist = max_dist
        self.max_force = max_force

        self.close_th = close_th
        self.open_th = open_th

        self.physics_client_id = physics_client_id
        self.constraint_id = None
        self.held = None

    def reset(self):
        if self.constraint_id is not None:
            try:
                p.removeConstraint(self.constraint_id, physicsClientId=self.physics_client_id)
            except Exception:
                pass
        self.constraint_id = None
        self.held = None

    def try_attach(self, robot_id: int, obj_id: int, grip: float) -> bool:
        if self.constraint_id is not None:
            return False
        if float(grip) < self.close_th:
            return False

        ee_pos, ee_orn = p.getLinkState(robot_id, self.ee_link, physicsClientId=self.physics_client_id)[4:6]
        obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.physics_client_id)

        if math.dist(ee_pos, obj_pos) > self.max_dist:
            return False

        inv_obj_pos, inv_obj_orn = p.invertTransform(obj_pos, obj_orn)
        child_pos, child_orn = p.multiplyTransforms(inv_obj_pos, inv_obj_orn, ee_pos, ee_orn)

        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=robot_id,
            parentLinkIndex=self.ee_link,
            childBodyUniqueId=obj_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFramePosition=child_pos,
            childFrameOrientation=child_orn,
            physicsClientId=self.physics_client_id,
        )
        p.changeConstraint(self.constraint_id, maxForce=self.max_force, physicsClientId=self.physics_client_id)
        self.held = obj_id
        return True

    def detach_if_open(self, grip: float) -> bool:
        if self.constraint_id is None:
            return False
        if float(grip) >= self.open_th:
            return False
        try:
            p.removeConstraint(self.constraint_id, physicsClientId=self.physics_client_id)
        except Exception:
            pass
        self.constraint_id = None
        self.held = None
        return True

    @property
    def holding(self) -> bool:
        return self.constraint_id is not None