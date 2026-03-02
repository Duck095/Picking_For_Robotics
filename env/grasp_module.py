import math
import pybullet as p


class SimpleAttachGrasp:
    """
    Stage-1 helper: nếu gripper đang đóng và EE đủ gần object -> attach constraint.
    """
    def __init__(self, ee_link=11, max_dist=0.09, max_force=2500):
        self.ee_link = ee_link
        self.max_dist = max_dist
        self.max_force = max_force
        self.cid = None
        self.held = None

    def reset(self):
        if self.cid is not None:
            try:
                p.removeConstraint(self.cid)
            except Exception:
                pass
        self.cid = None
        self.held = None

    def try_attach(self, robot_id: int, obj_id: int, grip: float) -> bool:
        if self.cid is not None:
            return False
        if float(grip) < 0.5:
            return False

        ee_pos, ee_orn = p.getLinkState(robot_id, self.ee_link)[4:6]
        obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)

        if math.dist(ee_pos, obj_pos) > self.max_dist:
            return False

        inv_obj_pos, inv_obj_orn = p.invertTransform(obj_pos, obj_orn)
        child_pos, child_orn = p.multiplyTransforms(inv_obj_pos, inv_obj_orn, ee_pos, ee_orn)

        self.cid = p.createConstraint(
            parentBodyUniqueId=robot_id,
            parentLinkIndex=self.ee_link,
            childBodyUniqueId=obj_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFramePosition=child_pos,
            childFrameOrientation=child_orn
        )
        p.changeConstraint(self.cid, maxForce=self.max_force)
        self.held = obj_id
        return True

    def detach_if_open(self, grip: float) -> bool:
        if self.cid is None:
            return False
        if float(grip) >= 0.5:
            return False
        try:
            p.removeConstraint(self.cid)
        except Exception:
            pass
        self.cid = None
        self.held = None
        return True