import pybullet as p
import numpy as np
import cv2


class CameraModule:
    def __init__(self, img_size=84, physics_client_id=None, use_tiny_renderer=False):
        self.img_size = img_size
        self.render_w = 128
        self.render_h = 128
        self.cid = physics_client_id
        self.use_tiny_renderer = use_tiny_renderer

        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.55, 0.0, 1.0],
            cameraTargetPosition=[0.55, 0.0, 0.0],
            cameraUpVector=[0, 1, 0],
        )

        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.01,
            farVal=2.0,
        )

    def render(self):
        w, h = self.render_w, self.render_h
        renderer = p.ER_TINY_RENDERER if self.use_tiny_renderer else p.ER_BULLET_HARDWARE_OPENGL

        img = p.getCameraImage(
            width=w,
            height=h,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=renderer,
            physicsClientId=self.cid,
        )

        rgba = np.asarray(img[2], dtype=np.uint8).reshape((h, w, 4))
        rgb = rgba[:, :, :3].copy()
        rgb_small = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return rgb_small