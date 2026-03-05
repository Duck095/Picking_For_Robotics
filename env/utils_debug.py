# env/utils_debug.py
import pybullet as p


class DebugDraw:
    def __init__(self, physics_client_id=None):
        self.cid = physics_client_id
        self.ids = []

    def clear(self):
        for _id in self.ids:
            try:
                p.removeUserDebugItem(_id, physicsClientId=self.cid)
            except Exception:
                pass
        self.ids = []

    def axes(self, pos, orn, axis_len=0.10, life=0.0, width=3):
        x_end = p.multiplyTransforms(pos, orn, [axis_len, 0, 0], [0, 0, 0, 1], physicsClientId=self.cid)[0]
        y_end = p.multiplyTransforms(pos, orn, [0, axis_len, 0], [0, 0, 0, 1], physicsClientId=self.cid)[0]
        z_end = p.multiplyTransforms(pos, orn, [0, 0, axis_len], [0, 0, 0, 1], physicsClientId=self.cid)[0]
        self.ids.append(p.addUserDebugLine(pos, x_end, [1, 0, 0], width, lifeTime=life, physicsClientId=self.cid))
        self.ids.append(p.addUserDebugLine(pos, y_end, [0, 1, 0], width, lifeTime=life, physicsClientId=self.cid))
        self.ids.append(p.addUserDebugLine(pos, z_end, [0, 0, 1], width, lifeTime=life, physicsClientId=self.cid))

    def point(self, pos, color=(1, 1, 1), size=1.2, life=0.0, text=""):
        self.ids.append(p.addUserDebugText(str(text), pos, textColorRGB=list(color), textSize=size, lifeTime=life, physicsClientId=self.cid))

    def line(self, a, b, color=(1, 1, 0), width=2, life=0.0):
        self.ids.append(p.addUserDebugLine(a, b, list(color), width, lifeTime=life, physicsClientId=self.cid))