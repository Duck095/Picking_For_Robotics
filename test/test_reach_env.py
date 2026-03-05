import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
import numpy as np
from env.reach_env import ReachEnv

env = ReachEnv(use_gui=True)

obs, _ = env.reset()

for step in range(5000):

    action = env.action_space.sample()  # random action

    obs, reward, done, truncated, info = env.step(action)

    print(
        "step:",
        step,
        "reward:",
        round(reward, 3),
        "dist:",
        round(info["ee_obj_dist"], 3)
    )

    if done or truncated:
        print("RESET\n")
        obs, _ = env.reset()

    time.sleep(1/60)

env.close()