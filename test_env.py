# source D:/Picking-For-Robot/rl_robot/Scripts/activate
import time
from env.conveyor_env import ConveyorEnv
import pybullet as p
env = ConveyorEnv(use_gui=True)
env.reset()

try:
    # 1) hạ xuống
    for t in range(120):
        obs, r, term, trunc, info = env.step([0.0, 0.0, -1.0, 0.0])
        if t % 30 == 0:
            print("down", t, "r", r, "dist", info.get("ee_obj_dist"), "z", info.get("obj_z"))
        time.sleep(1/120)

    # 2) đóng kẹp (attach sẽ kích hoạt nếu đủ gần)
    for t in range(60):
        obs, r, term, trunc, info = env.step([0.0, 0.0, 0.0, 1.0])
        if t % 10 == 0:
            print("close", t, "r", r, "dist", info.get("ee_obj_dist"), "z", info.get("obj_z"), "succ", info.get("success"))
        time.sleep(1/120)
        if term or trunc:
            break

    # 3) nhấc lên
    for t in range(180):
        obs, r, term, trunc, info = env.step([0.0, 0.0, 1.0, 1.0])
        if t % 30 == 0:
            print("lift", t, "r", r, "dist", info.get("ee_obj_dist"), "z", info.get("obj_z"), "succ", info.get("success"))
        time.sleep(1/120)
        if term or trunc:
            print("DONE", "term", term, "trunc", trunc, info)
            break
    print("cube pos:", p.getBasePositionAndOrientation(env.object_id)[0])

except KeyboardInterrupt:
    pass

env.close()
