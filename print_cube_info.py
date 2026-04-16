import os
import sys
from pprint import pprint

import pybullet as p
import pybullet_data

# Cho phép chạy file trực tiếp từ thư mục project hoặc độc lập
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from reach_env_config import EnvConfig
except Exception:
    # fallback nếu user đặt file này trong project root với cấu trúc config/reach_env_config.py
    from config.reach_env_config import EnvConfig


def print_cube_info(use_gui: bool = False):
    cfg = EnvConfig()

    cid = p.connect(p.GUI) if use_gui else p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setTimeStep(1.0 / cfg.physics_hz, physicsClientId=cid)

        p.loadURDF("plane.urdf", physicsClientId=cid)

        # Đặt cube giống reach_env.py
        spawn_x = 0.55
        spawn_y = 0.00
        spawn_z = cfg.obj_z

        obj_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=[spawn_x, spawn_y, spawn_z],
            physicsClientId=cid,
        )

        # Cho physics settle một chút
        for _ in range(60):
            p.stepSimulation(physicsClientId=cid)

        obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id, physicsClientId=cid)
        aabb_min, aabb_max = p.getAABB(obj_id, physicsClientId=cid)

        size_x = aabb_max[0] - aabb_min[0]
        size_y = aabb_max[1] - aabb_min[1]
        size_z = aabb_max[2] - aabb_min[2]

        center = [
            0.5 * (aabb_min[0] + aabb_max[0]),
            0.5 * (aabb_min[1] + aabb_max[1]),
            0.5 * (aabb_min[2] + aabb_max[2]),
        ]

        info = {
            "urdf": "cube_small.urdf",
            "obj_id": obj_id,
            "base_position": [float(v) for v in obj_pos],
            "base_orientation_quat": [float(v) for v in obj_orn],
            "aabb_min": [float(v) for v in aabb_min],
            "aabb_max": [float(v) for v in aabb_max],
            "size_xyz_m": {
                "x": float(size_x),
                "y": float(size_y),
                "z": float(size_z),
            },
            "center_from_aabb": [float(v) for v in center],
            "top_z": float(aabb_max[2]),
            "recommended_grasp_width_xy_m": float(max(size_x, size_y)),
            "recommended_pregrasp_opening_m": float(max(size_x, size_y) + 0.01),
        }

        print("\n===== CUBE INFO =====")
        pprint(info, sort_dicts=False)

        print("\n===== READABLE SUMMARY =====")
        print(f"Cube width X : {size_x:.6f} m")
        print(f"Cube width Y : {size_y:.6f} m")
        print(f"Cube height Z: {size_z:.6f} m")
        print(f"Top surface Z: {aabb_max[2]:.6f} m")
        print(f"Center       : ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")
        print(f"Grasp width  : {max(size_x, size_y):.6f} m")
        print(f"Pregrasp open: {max(size_x, size_y) + 0.01:.6f} m")

    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    print_cube_info(use_gui=False)
