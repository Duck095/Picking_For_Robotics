from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class PlaceEnvConfig:
    # Observation / rendering
    IMG_SIZE: int = 84
    FRAME_STACK: int = 4
    RENDER_W: int = 128
    RENDER_H: int = 128
    USE_CROP: bool = True
    CROP_BOX: Tuple[int, int, int, int] = (16, 112, 16, 112)  # (y0, y1, x0, x1)

    # Physics / control
    PHYSICS_HZ: int = 240
    RL_HZ: int = 20
    SUBSTEPS: int = 12
    MAX_STEPS: int = 250

    ACTION_SCALE_XY: float = 0.012
    ACTION_SCALE_Z: float = 0.012

    # Workspace
    X_RANGE: Tuple[float, float] = (0.10, 0.85)
    Y_RANGE: Tuple[float, float] = (-0.60, 0.60)
    Z_RANGE: Tuple[float, float] = (0.005, 0.55)

    # Place task setup
    START_HELD: bool = True
    SUBSTAGE: str = "3A"

    OBJECT_SPAWN_POS: Tuple[float, float, float] = (0.55, 0.0, 0.02)

    TARGET_POS_3A: Tuple[float, float, float] = (0.45, 0.20, 0.02)
    TARGET_X_RANGE_3BC: Tuple[float, float] = (0.40, 0.55)
    TARGET_Y_RANGE_3BC: Tuple[float, float] = (0.10, 0.30)
    TARGET_Z: float = 0.02

    # Success params
    SUCCESS_DIST_3A: float = 0.09
    SUCCESS_DIST_3B: float = 0.06
    SUCCESS_DIST_3C: float = 0.03

    # Đặt bằng 0.02 (bằng với tọa độ Z của vật khi nằm ổn định trên mặt phẳng pybullet)
    TABLE_Z_3A: float = 0.02
    TABLE_Z_3B: float = 0.02
    TABLE_Z_3C: float = 0.02

    Z_RELEASE_MAX_3A: Optional[float] = None
    Z_RELEASE_MAX_3B: Optional[float] = None
    Z_RELEASE_MAX_3C: Optional[float] = 0.08

    # Reward
    TIME_PENALTY: float = 0.01
    DIST_WEIGHT: float = 0.6
    RELEASE_BONUS: float = 0.2
    SUCCESS_BONUS: float = 3.0
    DELTA_CLIP: float = 0.03
    HIGH_RELEASE_PENALTY: float = 1.0

    # Start-held helper
    START_HOLD_DESCEND_STEPS: int = 40
    START_HOLD_CLOSE_STEPS: int = 15
    START_HOLD_ATTACH_SETTLE_STEPS: int = 5
    START_HOLD_LIFT_STEPS: int = 20
    START_HOLD_APPROACH_Z: float = 0.10
    START_HOLD_MIN_Z: float = 0.12

    # Visual grasp tuning
    RELEASE_COOLDOWN_STEPS: int = 4
    # Dịch vật xuống 3.5cm theo trục Z của kẹp để lọt đúng vào đệm cao su của ngón tay thay vì lún vào lòng bàn tay
    GRASP_OFFSET_LOCAL: Tuple[float, float, float] = (0.0, 0.0, 0.035)
    GRASP_DIST_THRESH: float = 0.08
    GRASP_XY_THRESH: float = 0.06
    GRASP_Z_THRESH: float = 0.06
    GRASP_FORCE: float = 8000.0
       
    def get_success_params(self, substage: str):
        substage = str(substage).upper()

        if substage == "3A":
            return {
                "success_dist": self.SUCCESS_DIST_3A,
                "table_z": self.TABLE_Z_3A,
                "z_release_max": self.Z_RELEASE_MAX_3A,
            }
        if substage == "3B":
            return {
                "success_dist": self.SUCCESS_DIST_3B,
                "table_z": self.TABLE_Z_3B,
                "z_release_max": self.Z_RELEASE_MAX_3B,
            }
        if substage == "3C":
            return {
                "success_dist": self.SUCCESS_DIST_3C,
                "table_z": self.TABLE_Z_3C,
                "z_release_max": self.Z_RELEASE_MAX_3C,
            }

        raise ValueError(f"Unknown place substage: {substage}")