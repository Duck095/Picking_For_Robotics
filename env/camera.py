from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pybullet as p


# ============================================================
# Camera Utilities
# ------------------------------------------------------------
# File này chuyên lo phần camera / render:
#   - top-down camera
#   - oblique camera
#   - wrist / ee-follow camera
#   - rgb / depth / segmentation
#
# Stage 1:
#   - chưa bắt buộc dùng để train
#   - rất hữu ích để debug
#
# Stage 5+:
#   - sorting theo màu
#   - tracking vật trên conveyor
#   - domain randomization bằng image
# ============================================================


@dataclass
class CameraConfig:
    """
    Cấu hình cơ bản của camera.
    """
    width: int = 84
    height: int = 84
    fov: float = 60.0
    near: float = 0.01
    far: float = 3.0


class Camera:
    """
    Bộ tiện ích camera cho PyBullet.

    Hỗ trợ:
    - render bằng view matrix + projection matrix
    - camera top-down
    - camera nhìn xiên
    - camera bám theo EE / wrist

    Output:
    - rgb: np.ndarray (H, W, 3), uint8
    - depth: np.ndarray (H, W), float32
    - seg: np.ndarray (H, W), int32
    """

    def __init__(self, cfg: Optional[CameraConfig] = None):
        self.cfg = cfg if cfg is not None else CameraConfig()

    # ========================================================
    # LOW-LEVEL RENDER
    # ========================================================
    def _projection_matrix(self) -> Sequence[float]:
        """
        Tạo projection matrix theo FOV.
        """
        aspect = float(self.cfg.width) / float(self.cfg.height)
        proj = p.computeProjectionMatrixFOV(
            fov=self.cfg.fov,
            aspect=aspect,
            nearVal=self.cfg.near,
            farVal=self.cfg.far,
        )
        return proj

    def render(
        self,
        view_matrix: Sequence[float],
        projection_matrix: Optional[Sequence[float]] = None,
        renderer: int = p.ER_BULLET_HARDWARE_OPENGL,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Render scene bằng view matrix + projection matrix.

        Parameters
        ----------
        view_matrix:
            Camera view matrix.
        projection_matrix:
            Nếu None thì dùng projection mặc định từ config.
        renderer:
            Renderer của PyBullet.

        Returns
        -------
        rgb : np.ndarray (H, W, 3), uint8
        depth : np.ndarray (H, W), float32
        seg : np.ndarray (H, W), int32
        """
        if projection_matrix is None:
            projection_matrix = self._projection_matrix()

        w, h, rgba, depth_buf, seg = p.getCameraImage(
            width=self.cfg.width,
            height=self.cfg.height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=renderer,
        )

        rgba = np.reshape(rgba, (h, w, 4)).astype(np.uint8)
        rgb = rgba[:, :, :3].copy()

        depth_buf = np.reshape(depth_buf, (h, w)).astype(np.float32)
        seg = np.reshape(seg, (h, w)).astype(np.int32)

        # Chuyển depth buffer sang metric depth
        near = self.cfg.near
        far = self.cfg.far
        depth = (far * near) / (far - (far - near) * depth_buf)

        return rgb, depth, seg

    # ========================================================
    # VIEW MATRIX HELPERS
    # ========================================================
    @staticmethod
    def view_from_eye_target_up(
        eye: Sequence[float],
        target: Sequence[float],
        up: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> Sequence[float]:
        """
        Tạo view matrix từ camera eye -> target.
        """
        return p.computeViewMatrix(
            cameraEyePosition=eye,
            cameraTargetPosition=target,
            cameraUpVector=up,
        )

    def render_from_eye_target_up(
        self,
        eye: Sequence[float],
        target: Sequence[float],
        up: Sequence[float] = (0.0, 0.0, 1.0),
        renderer: int = p.ER_BULLET_HARDWARE_OPENGL,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Render trực tiếp từ eye/target/up.
        """
        view = self.view_from_eye_target_up(eye=eye, target=target, up=up)
        return self.render(view_matrix=view, renderer=renderer)

    # ========================================================
    # TOP-DOWN CAMERA
    # ========================================================
    def render_top_down(
        self,
        center_xy: Tuple[float, float] = (0.55, 0.05),
        height: float = 0.90,
        target_z: float = 0.0,
        up: Sequence[float] = (0.0, 1.0, 0.0),
        renderer: int = p.ER_BULLET_HARDWARE_OPENGL,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Camera nhìn từ trên xuống.

        Rất phù hợp để:
        - debug Stage 1
        - quan sát ROI pick
        - chuẩn bị sorting / conveyor về sau
        """
        eye = [center_xy[0], center_xy[1], height]
        target = [center_xy[0], center_xy[1], target_z]
        return self.render_from_eye_target_up(
            eye=eye,
            target=target,
            up=up,
            renderer=renderer,
        )

    # ========================================================
    # OBLIQUE / DIAGONAL CAMERA
    # ========================================================
    def render_oblique(
        self,
        eye: Sequence[float] = (1.10, -0.80, 0.75),
        target: Sequence[float] = (0.55, 0.05, 0.05),
        up: Sequence[float] = (0.0, 0.0, 1.0),
        renderer: int = p.ER_BULLET_HARDWARE_OPENGL,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Camera nhìn xiên toàn cảnh.

        Hữu ích cho:
        - quay video debug
        - xem robot + bàn + object cùng lúc
        """
        return self.render_from_eye_target_up(
            eye=eye,
            target=target,
            up=up,
            renderer=renderer,
        )

    # ========================================================
    # EE / WRIST-FOLLOW CAMERA
    # ========================================================
    def render_ee_follow(
        self,
        ee_pos: Sequence[float],
        ee_orn: Sequence[float],
        eye_offset_local: Sequence[float] = (0.0, 0.0, 0.08),
        target_offset_local: Sequence[float] = (0.08, 0.0, 0.0),
        renderer: int = p.ER_BULLET_HARDWARE_OPENGL,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Camera bám theo end-effector.

        Ý tưởng:
        - camera nằm gần EE
        - target nhìn về phía trước của EE

        Cách làm:
        - lấy rotation matrix từ quaternion EE
        - transform offset local -> world
        - tính eye / target trong world

        Hữu ích cho:
        - debug wrist view
        - giai đoạn sau nếu muốn dùng camera gắn theo tay robot
        """
        ee_pos = np.array(ee_pos, dtype=np.float32)
        ee_orn = np.array(ee_orn, dtype=np.float32)

        rot = np.array(p.getMatrixFromQuaternion(ee_orn), dtype=np.float32).reshape(3, 3)

        eye_offset_local = np.array(eye_offset_local, dtype=np.float32)
        target_offset_local = np.array(target_offset_local, dtype=np.float32)

        eye_world = ee_pos + rot @ eye_offset_local
        target_world = ee_pos + rot @ target_offset_local

        # Chọn up vector theo trục Z local của EE
        up_world = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float32)

        return self.render_from_eye_target_up(
            eye=eye_world.tolist(),
            target=target_world.tolist(),
            up=up_world.tolist(),
            renderer=renderer,
        )

    # ========================================================
    # SAVE / NORMALIZE HELPERS
    # ========================================================
    @staticmethod
    def rgb_to_float01(rgb: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa RGB uint8 -> float32 [0, 1].
        """
        return rgb.astype(np.float32) / 255.0

    @staticmethod
    def depth_to_uint8(depth: np.ndarray, near_clip: Optional[float] = None, far_clip: Optional[float] = None) -> np.ndarray:
        """
        Chuyển depth metric -> ảnh uint8 để debug nhanh.

        near_clip / far_clip:
            Khoảng depth muốn hiển thị rõ.
            Nếu không truyền, dùng min/max của ảnh depth.
        """
        d = depth.copy()

        if near_clip is None:
            near_clip = float(np.nanmin(d))
        if far_clip is None:
            far_clip = float(np.nanmax(d))

        denom = max(1e-6, far_clip - near_clip)
        d = np.clip((d - near_clip) / denom, 0.0, 1.0)
        return (d * 255.0).astype(np.uint8)


# ============================================================
# TEST NHANH
# ------------------------------------------------------------
# Chạy file này trực tiếp để test render.
# ============================================================
if __name__ == "__main__":
    import pybullet_data
    import time

    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    p.loadURDF("plane.urdf")

    # Bàn đơn giản
    table_half = [0.45, 0.60, 0.02]
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half)
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=table_half, rgbaColor=[0.75, 0.75, 0.75, 1.0])
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[0.55, 0.0, -0.02],
    )

    # Spawn cube mẫu
    cube_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
    cube_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(
        baseMass=0.2,
        baseCollisionShapeIndex=cube_col,
        baseVisualShapeIndex=cube_vis,
        basePosition=[0.55, 0.05, 0.02],
    )

    cam = Camera(CameraConfig(width=128, height=128, fov=60))

    rgb_top, depth_top, seg_top = cam.render_top_down(center_xy=(0.55, 0.05))
    print("Top-down RGB shape:", rgb_top.shape)
    print("Top-down depth shape:", depth_top.shape)
    print("Top-down seg shape:", seg_top.shape)

    rgb_oblique, depth_oblique, seg_oblique = cam.render_oblique()
    print("Oblique RGB shape:", rgb_oblique.shape)

    # Giữ GUI để bạn nhìn scene
    while True:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)