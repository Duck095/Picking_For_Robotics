# env/camera.py
import pybullet as p
import numpy as np
import cv2


class Camera:
    def __init__(
        self,
        img_size=84,
        render_w=128,
        render_h=128,
        physics_client_id=None,
        use_gui=False,
        use_crop=True,
        crop_box=(16, 112, 16, 112),  # (y0, y1, x0, x1)
    ):
        self.img_size = int(img_size)
        self.render_w = int(render_w)
        self.render_h = int(render_h)
        self.cid = physics_client_id
        self.use_gui = bool(use_gui)

        self.use_crop = bool(use_crop)
        self.crop_box = tuple(crop_box)

        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.55, 0.0, 1.0],
            cameraTargetPosition=[0.55, 0.0, 0.0],
            cameraUpVector=[0, 1, 0],
            physicsClientId=self.cid,
        )

        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.render_w) / float(self.render_h),
            nearVal=0.01,
            farVal=2.0,
        )

        self.renderer = p.ER_BULLET_HARDWARE_OPENGL if self.use_gui else p.ER_TINY_RENDERER

    def render_rgb(self):
        img = p.getCameraImage(
            width=self.render_w,
            height=self.render_h,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=self.renderer,
            physicsClientId=self.cid,
        )
        rgba = np.asarray(img[2], dtype=np.uint8).reshape((self.render_h, self.render_w, 4))
        rgb = rgba[:, :, :3].copy()

        if self.use_crop:
            y0, y1, x0, x1 = self.crop_box
            y0 = max(0, min(self.render_h - 1, int(y0)))
            y1 = max(y0 + 1, min(self.render_h, int(y1)))
            x0 = max(0, min(self.render_w - 1, int(x0)))
            x1 = max(x0 + 1, min(self.render_w, int(x1)))
            rgb = rgb[y0:y1, x0:x1]

        rgb_small = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return rgb_small