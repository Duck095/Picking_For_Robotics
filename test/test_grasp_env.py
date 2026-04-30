from __future__ import annotations

import glob
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from config.grasp_env_config import build_stage2_grasp_config
from env.grasp_env import GraspEnv


USE_GUI = True
N_EPISODES = 10
SLEEP_SEC = 0.08 if USE_GUI else 0.0
DETERMINISTIC = True
USE_REPRODUCIBLE_SEEDS = False
BASE_SEED = 42

VALID_SUBSTAGES = ["2A", "2B", "2C", "2D"]


def normalize_path(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def choose_substage() -> str:
    print("\n===== CHỌN SUBSTAGE STAGE 2 MUỐN TEST =====")
    for i, s in enumerate(VALID_SUBSTAGES, start=1):
        print(f"{i}. {s}")

    choice = input("\nEnter số / Enter mặc định = 2A\n> ").strip()
    if choice == "":
        return "2A"

    if choice.isdigit() and 0 <= int(choice) - 1 < len(VALID_SUBSTAGES):
        return VALID_SUBSTAGES[int(choice) - 1]

    choice = choice.upper()
    return choice if choice in VALID_SUBSTAGES else "2A"


def substage_to_tag(substage: str) -> str:
    return substage.replace(".", "_")


def get_available_models_for_substage(substage: str) -> List[str]:
    tag = substage_to_tag(substage)
    models: List[str] = []
    seen = set()

    def add(path: str):
        norm = normalize_path(path)
        if os.path.exists(path) and norm not in seen:
            seen.add(norm)
            models.append(os.path.normpath(path))

    add(f"models/stage2_grasp_mastery_{tag}.zip")
    add(f"models/stage2_grasp_mastery_{tag}_latest.zip")

    for pattern in [
        f"models/checkpoints_stage2_{tag}/*.zip",
        f"models/checkpoints_grasp_{tag}/*.zip",
    ]:
        for p in sorted(glob.glob(pattern), reverse=True):
            add(p)

    return models


def choose_model(models: List[str], substage: str) -> Tuple[Optional[str], str]:
    print(f"\n===== MODEL CHO SUBSTAGE {substage} =====")
    if not models:
        print("Không tìm thấy model phù hợp -> dùng random policy")
        return None, "random"

    for i, m in enumerate(models, start=1):
        print(f"{i}. {m}")

    choice = input("\nEnter số / Enter để dùng model đầu tiên / r để random\n> ").strip().lower()
    if choice == "r":
        return None, "random"
    if choice == "":
        return models[0], "model"
    if choice.isdigit() and 0 <= int(choice) - 1 < len(models):
        return models[int(choice) - 1], "model"
    return models[0], "model"


def print_phase_flow(substage: str):
    flows = {
        "2A": "xy_align -> descend -> close",
        "2B": "xy_align -> descend -> close -> hold",
        "2C": "xy_align -> descend -> close -> hold -> lift",
        "2D": "xy_align -> descend -> close -> hold -> lift -> return_home",
    }
    print(f"\n[FLOW {substage}] {flows[substage]}")


def main():
    substage = choose_substage()
    models = get_available_models_for_substage(substage)
    model_path, mode = choose_model(models, substage)

    model = PPO.load(model_path) if mode == "model" and model_path is not None else None
    print(f"\nUsing {'model: ' + model_path if model else 'RANDOM POLICY for substage ' + substage}")

    cfg = build_stage2_grasp_config(substage)
    cfg.sim.use_gui = USE_GUI
    cfg.sim.seed = BASE_SEED

    env = GraspEnv(cfg)
    print_phase_flow(substage)

    try:
        for ep in range(N_EPISODES):
            episode_seed = cfg.sim.seed + ep if USE_REPRODUCIBLE_SEEDS else None
            obs, info = env.reset(seed=episode_seed) if episode_seed is not None else env.reset()
            done = truncated = False
            total_reward = 0.0
            step = 0

            print(f"\n=== Episode {ep+1} | substage={substage} ===")
            print(
                f"[RESET] seed={episode_seed} | "
                f"object={np.round(info.get('object_pos'),4)} | "
                f"target={np.round(info.get('target_pos'),4)} | "
                f"phase={info.get('phase')} | "
                f"grasp_z={info.get('grasp_z','N/A')} | "
                f"lift_z={info.get('lift_z','N/A')} | "
                f"home={np.round(info.get('home_pos'),4)}"
            )

            while not (done or truncated):
                if model is not None:
                    action, _ = model.predict(obs, deterministic=DETERMINISTIC)
                else:
                    action = env.action_space.sample()

                obs, reward, done, truncated, info = env.step(action)
                total_reward += float(reward)
                step += 1

                print(
                    f"step={step:03d} | "
                    f"phase={info.get('phase','N/A'):>11s} | "
                    f"reward={reward:+.3f} | "
                    f"xy={info.get('xy_dist',-1):.3f} | "
                    f"z={info.get('z_dist',-1):.3f} | "
                    f"yaw={info.get('yaw_error',-1):.3f} | "
                    f"grip={info.get('grip_width',-1):.3f} | "
                    f"home_err={info.get('ee_to_home_xyz_error',-1):.3f} | "
                    f"contact=({int(info.get('left_contact',False))},{int(info.get('right_contact',False))}) | "
                    f"grasp={info.get('grasp_established',False)} | "
                    f"success={info.get('success',False)}"
                )

                if USE_GUI:
                    time.sleep(SLEEP_SEC)

            print(
                f"[EP DONE] total_reward={total_reward:.3f} "
                f"success={info.get('success',False)} "
                f"truncated={truncated}"
            )

    finally:
        env.close()


if __name__ == "__main__":
    main()