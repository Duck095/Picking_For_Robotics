from __future__ import annotations

import os
import glob
import time
from typing import List, Optional, Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from stable_baselines3 import PPO

from config.reach_env_config import build_reach_config
from env.reach_env import ReachEnv


# ============================================================
# TEST CONFIG
# ============================================================
USE_GUI = True
N_EPISODES = 5
SLEEP_SEC = 0.1 if USE_GUI else 0.0
DETERMINISTIC = True

MODEL_DIR = "models"


# ============================================================
# MODEL DISCOVERY
# ============================================================
def get_available_models() -> List[str]:
    models = []

    priority = [
        "models/stage1_reach_1C.zip",
        "models/stage1_reach_1C_latest.zip",
        "models/stage1_reach_1B.zip",
        "models/stage1_reach_1B_latest.zip",
        "models/stage1_reach_1A.zip",
        "models/stage1_reach_1A_latest.zip",
    ]

    for p in priority:
        if os.path.exists(p):
            models.append(p)

    # checkpoints
    for sub in ["1C", "1B", "1A"]:
        ckpt = glob.glob(f"models/checkpoints_stage1_{sub}/*.zip")
        ckpt = sorted(ckpt, reverse=True)
        models.extend(ckpt)

    return models


def choose_model(models: List[str]) -> Tuple[Optional[str], str]:
    print("\nAvailable models:")
    for i, m in enumerate(models):
        print(f"{i+1}. {m}")

    print("\nEnter number / Enter for best / r for random")
    choice = input("> ").strip().lower()

    if choice == "r":
        return None, "random"

    if choice == "":
        return models[0], "model"

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx], "model"

    return models[0], "model"


def infer_substage(path: str) -> str:
    p = path.lower()
    if "1c" in p:
        return "1C"
    if "1b" in p:
        return "1B"
    return "1A"

# ============================================================
# MAIN
# ============================================================
def main():
    models = get_available_models()

    if len(models) == 0:
        print("No model found → random test")
        model = None
        substage = "1A"
        mode = "random"
        model_path = None
    else:
        path, mode = choose_model(models)

        if mode == "model":
            print(f"Using model: {path}")
            model = PPO.load(path)
            substage = infer_substage(path)
            model_path = path
        else:
            model = None
            substage = "1A"
            model_path = None

    cfg = build_reach_config(substage)
    cfg.sim.use_gui = USE_GUI
    env = ReachEnv(cfg)

    success = 0
    rewards = []
    final_dists = []

    try:
        for ep in range(N_EPISODES):
            obs, info = env.reset()
            done = False
            truncated = False
            total = 0.0
            step = 0
            last_dist = None

            # ===== IN 1 LẦN DUY NHẤT CHO MỖI EPISODE =====
            ee_pos_reset, _ = env.robot.get_ee_pose()

            print(f"\n=== Episode {ep+1} ===")
            print("[RESET]")
            print(f"  ee_pos      = {np.round(ee_pos_reset, 4)}")
            print(f"  object_pos  = {np.round(info.get('object_pos'), 4) if info.get('object_pos') is not None else None}")
            print(f"  target_pos  = {np.round(info.get('target_pos'), 4) if info.get('target_pos') is not None else None}")
            print(f"  dist        = {info.get('dist', -1):.4f}")
            print(f"  xy_dist     = {info.get('xy_dist', -1):.4f}")
            print(f"  z_dist      = {info.get('z_dist', -1):.4f}")

            while not (done or truncated):
                if model:
                    action, _ = model.predict(obs, deterministic=DETERMINISTIC)
                else:
                    action = env.action_space.sample()

                obs, reward, done, truncated, info = env.step(action)
                total += reward
                step += 1
                last_dist = info.get("dist", None)

                # ===== MỖI STEP CHỈ IN NGẮN =====
                print(
                    f"step={step:03d} | reward={reward:.3f} | "
                    f"dist={info.get('dist', -1):.3f} | "
                    f"xy={info.get('xy_dist', -1):.3f} | "
                    f"z={info.get('z_dist', -1):.3f} | "
                    f"success={info.get('success', False)}"
                )

                if USE_GUI:
                    time.sleep(SLEEP_SEC)

            if info.get("success", False):
                success += 1

            rewards.append(total)
            if last_dist is not None:
                final_dists.append(last_dist)

            print(
                f"[EP DONE] total_reward={total:.3f} | "
                f"final_dist={last_dist if last_dist is not None else 'N/A'} | "
                f"success={info.get('success', False)} | "
                f"terminated={done} | truncated={truncated}"
            )

    finally:
        env.close()

    print("\n===== RESULT =====")
    print("\n" + "=" * 70)
    print(f"Mode: {'MODEL TEST' if mode == 'model' else 'RANDOM TEST'}")

    if model_path is not None:
        print(f"Model used: {model_path}")

    print(f"Episodes: {N_EPISODES}")
    print(f"Success rate: {success / N_EPISODES:.2f}")
    print(f"Mean reward: {np.mean(rewards):.3f}")

    if len(final_dists) > 0:
        print(f"Mean final dist: {np.mean(final_dists):.3f}")
    else:
        print("Mean final dist: N/A")

if __name__ == "__main__":
    main()