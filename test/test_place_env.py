import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
import numpy as np
from env.place_env import PlaceEnv  

env = PlaceEnv(use_gui=True, start_held=True)

obs, _ = env.reset()

for step in range(5000):
    action = env.action_space.sample()  # random action

    obs, reward, done, truncated, info = env.step(action)

    print(
        "step:",
        step,
        "reward:",
        round(float(reward), 3),
        "success:",
        info.get("success"),
        "holding:",
        info.get("holding"),
        "released:",
        info.get("released", info.get("drop_event")),
        "dist:",
        round(float(info.get("obj_target_dist", -1.0)), 3),
        "z:",
        round(float(info.get("obj_z", -1.0)), 3),
    )

    if done or truncated:
        print("RESET\n")
        obs, _ = env.reset()

    time.sleep(1 / 60)

env.close()