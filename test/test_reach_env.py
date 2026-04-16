from __future__ import annotations

import glob
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.reach_env_config import build_reach_config
from env.reach_env import ReachEnv


# ============================================================
# TEST CONFIG
# ============================================================
USE_GUI = True
N_EPISODES = 20
SLEEP_SEC = 0.10 if USE_GUI else 0.0
DETERMINISTIC = True

MODEL_DIR = "models"


def normalize_path(p: str) -> str:
    return os.path.normcase(os.path.normpath(p))


def substage_to_tag(substage: str) -> str:
    return substage.replace(".", "_")


# ============================================================
# SUBSTAGE CHOICE
# ============================================================
VALID_SUBSTAGES = ["1A", "1B", "1C", "1D", "1E", "1F"]


def choose_substage() -> str:
    print("\n===== CHỌN SUBSTAGE MUỐN TEST =====")
    for i, s in enumerate(VALID_SUBSTAGES, start=1):
        print(f"{i}. {s}")

    print("\nEnter số / Enter mặc định = 1A")
    choice = input("> ").strip()

    if choice == "":
        return "1A"

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(VALID_SUBSTAGES):
            return VALID_SUBSTAGES[idx]

    choice_upper = choice.upper()
    if choice_upper in VALID_SUBSTAGES:
        return choice_upper

    print("Input không hợp lệ -> dùng mặc định 1A")
    return "1A"


# ============================================================
# MODEL DISCOVERY
# ============================================================
def infer_substage(path: str) -> str:
    p = path.lower()

    if "1f" in p:
        return "1F"
    if "1e" in p:
        return "1E"
    if "1d" in p:
        return "1D"
    if "1c" in p:
        return "1C"
    if "1b" in p:
        return "1B"
    return "1A"


def get_available_models_for_substage(substage: str) -> List[str]:
    models: List[str] = []
    seen = set()

    def add_path(p: str):
        norm = normalize_path(p)
        if os.path.exists(p) and norm not in seen:
            seen.add(norm)
            models.append(os.path.normpath(p))

    tag = substage_to_tag(substage)

    priority = [
        f"models/stage1_pregrasp_mastery_{tag}.zip",
        f"models/stage1_pregrasp_mastery_{tag}_latest.zip",
        f"models/stage1_{tag}.zip",
        f"models/stage1_{tag}_latest.zip",
        f"models/reach_{tag}.zip",
        f"models/reach_{tag}_latest.zip",
    ]

    for p in priority:
        add_path(p)

    checkpoint_patterns = [
        f"models/checkpoints_stage1_{tag}/*.zip",
        f"models/checkpoints_pregrasp_mastery_{tag}/*.zip",
        f"models/checkpoints_{tag}/*.zip",
    ]

    for pattern in checkpoint_patterns:
        ckpts = sorted(glob.glob(pattern), reverse=True)
        for p in ckpts:
            add_path(p)

    all_zip = sorted(
        glob.glob(os.path.join(MODEL_DIR, "**", "*.zip"), recursive=True),
        reverse=True,
    )
    for p in all_zip:
        if infer_substage(p) == substage:
            add_path(p)

    return models


def choose_model(models: List[str], substage: str) -> Tuple[Optional[str], str]:
    print(f"\n===== MODEL CHO SUBSTAGE {substage} =====")

    if len(models) == 0:
        print("Không tìm thấy model phù hợp -> dùng random policy")
        return None, "random"

    for i, m in enumerate(models, start=1):
        print(f"{i}. {m}")

    print("\nEnter số / Enter để dùng model đầu tiên / r để random")
    choice = input("> ").strip().lower()

    if choice == "r":
        return None, "random"

    if choice == "":
        return models[0], "model"

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx], "model"

    print("Input không hợp lệ -> dùng model đầu tiên")
    return models[0], "model"


# ============================================================
# PRINT HELPERS
# ============================================================
def print_reset_info(env: ReachEnv, info: dict, substage: str):
    ee_pos_reset, _ = env.robot.get_ee_pose()

    print("[RESET]")
    print("Tọa độ ban đầu sau reset:")
    print(f"  ee_pos            = {np.round(ee_pos_reset, 4)}")
    print(
        f"  object_pos        = "
        f"{np.round(info.get('object_pos'), 4) if info.get('object_pos') is not None else None}"
    )
    print(f"  object_yaw        = {info.get('object_yaw', 0.0):.4f}")
    print(
        f"  target_pos        = "
        f"{np.round(info.get('target_pos'), 4) if info.get('target_pos') is not None else None}"
    )
    print(f"  target_yaw        = {info.get('target_yaw', 0.0):.4f}")
    print(f"  dist              = {info.get('dist', -1):.4f}")
    print(f"  xy_dist           = {info.get('xy_dist', -1):.4f}")
    print(f"  z_dist            = {info.get('z_dist', -1):.4f}")
    print(f"  yaw_error         = {info.get('yaw_error', 0.0):.4f}")
    print(f"  phase             = {info.get('phase', 'N/A')}")
    print(f"  stable_pose_steps = {info.get('stable_pose_steps', 0)}")


def print_final_info(env: ReachEnv, info: dict, truncated: bool, substage: str):
    ee_pos_final, _ = env.robot.get_ee_pose()

    print("Tọa độ sau khi đạt success hoặc hết episode:")
    print(f"  ee_pos            = {np.round(ee_pos_final, 4)}")
    print(
        f"  object_pos        = "
        f"{np.round(info.get('object_pos'), 4) if info.get('object_pos') is not None else None}"
    )
    print(f"  object_yaw        = {info.get('object_yaw', 0.0):.4f}")
    print(
        f"  target_pos        = "
        f"{np.round(info.get('target_pos'), 4) if info.get('target_pos') is not None else None}"
    )
    print(f"  target_yaw        = {info.get('target_yaw', 0.0):.4f}")
    print(f"  dist              = {info.get('dist', -1):.4f}")
    print(f"  xy_dist           = {info.get('xy_dist', -1):.4f}")
    print(f"  z_dist            = {info.get('z_dist', -1):.4f}")
    print(f"  yaw_error         = {info.get('yaw_error', 0.0):.4f}")
    print(f"  phase             = {info.get('phase', 'N/A')}")
    print(f"  stable_pose_steps = {info.get('stable_pose_steps', 0)}")

    if info.get("object_pos") is not None:
        delta = np.array(info.get("object_pos")) - np.array(ee_pos_final)
        print(f"  delta_xyz         = {np.round(delta, 4)}")

    episode_success = bool(info.get("success", False))
    print(f"  done_reason       = success={episode_success}, truncated={truncated}")


# ============================================================
# MAIN
# ============================================================
def main():
    substage = choose_substage()
    models = get_available_models_for_substage(substage)
    model_path, mode = choose_model(models, substage)

    if mode == "model" and model_path is not None:
        print(f"\nUsing model: {model_path}")
        model = PPO.load(model_path)
    else:
        print(f"\nUsing RANDOM POLICY for substage {substage}")
        model = None
        model_path = None

    cfg = build_reach_config(substage)
    cfg.sim.use_gui = USE_GUI
    env = ReachEnv(cfg)

    success_count = 0
    rewards = []
    final_dists = []
    final_xy_dists = []
    final_z_dists = []
    final_stable_steps = []
    final_yaw_errors = []

    try:
        for ep in range(N_EPISODES):
            obs, info = env.reset()
            done = False
            truncated = False
            total = 0.0
            step = 0

            last_dist = None
            last_xy = None
            last_z = None
            last_yaw_error = None

            print(f"\n=== Episode {ep + 1} | substage={substage} ===")
            print_reset_info(env, info, substage)

            while not (done or truncated):
                if model is not None:
                    action, _ = model.predict(obs, deterministic=DETERMINISTIC)
                else:
                    action = env.action_space.sample()

                obs, reward, done, truncated, info = env.step(action)
                total += reward
                step += 1

                last_dist = info.get("dist", None)
                last_xy = info.get("xy_dist", None)
                last_z = info.get("z_dist", None)
                last_yaw_error = info.get("yaw_error", None)

                line = (
                    f"step={step:03d} | "
                    f"phase={info.get('phase', 'N/A'):>8s} | "
                    f"reward={reward:+.3f} | "
                    f"xy={info.get('xy_dist', -1):.3f} | "
                    f"zdist={info.get('z_dist', -1):.3f} | "
                    f"grip={info.get('grip_width', -1):.3f} | "
                    f"contact=({int(info.get('left_contact', False))},{int(info.get('right_contact', False))}) | "
                    f"grasp={info.get('grasp_established', False)} | "
                    f"success={info.get('success', False)} | "
                    f"ee_z={info.get('ee_z_world', -1):.3f} | "
                    f"grasp_z={info.get('grasp_z_world', -1):.3f} | "
                    f"cube_z={info.get('cube_center_z_world', -1):.3f} | "
                    f"finger_mid_z={info.get('finger_mid_z_world', -1):.3f} | "
                    f"ee->grasp_dz={info.get('ee_to_grasp_z_error', -1):.3f} | "
                    f"finger->cube_dz={info.get('finger_mid_to_cube_z_error', -1):.3f}"
                )
                print(line)

                if USE_GUI:
                    time.sleep(SLEEP_SEC)

            print_final_info(env, info, truncated, substage)

            episode_success = bool(info.get("success", False))
            if episode_success:
                success_count += 1

            rewards.append(total)

            if last_dist is not None:
                final_dists.append(float(last_dist))
            if last_xy is not None:
                final_xy_dists.append(float(last_xy))
            if last_z is not None:
                final_z_dists.append(float(last_z))
            if last_yaw_error is not None:
                final_yaw_errors.append(float(last_yaw_error))

            final_stable_steps.append(float(info.get("stable_pose_steps", 0)))

            line = (
                f"[EP DONE] total_reward={total:.3f} | "
                f"final_dist={last_dist if last_dist is not None else 'N/A'} | "
                f"final_xy={last_xy if last_xy is not None else 'N/A'} | "
                f"final_z={last_z if last_z is not None else 'N/A'} | "
                f"final_yaw={last_yaw_error if last_yaw_error is not None else 'N/A'} | "
                f"stable={info.get('stable_pose_steps', 0)} | "
                f"success={episode_success} | "
                f"terminated={done} | truncated={truncated}"
            )
            print(line)

    finally:
        env.close()

    print("\n===== RESULT =====")
    print("=" * 70)
    print(f"Substage tested: {substage}")
    print(f"Mode: {'MODEL TEST' if mode == 'model' else 'RANDOM TEST'}")

    if model_path is not None:
        print(f"Model used: {model_path}")

    print(f"Episodes: {N_EPISODES}")
    print(f"Success rate: {success_count / N_EPISODES:.2f}")
    print(f"Mean reward: {np.mean(rewards):.3f}")

    if len(final_dists) > 0:
        print(f"Mean final dist: {np.mean(final_dists):.4f}")
    else:
        print("Mean final dist: N/A")

    if len(final_xy_dists) > 0:
        print(f"Mean final xy_dist: {np.mean(final_xy_dists):.4f}")
    else:
        print("Mean final xy_dist: N/A")

    if len(final_z_dists) > 0:
        print(f"Mean final z_dist: {np.mean(final_z_dists):.4f}")
    else:
        print("Mean final z_dist: N/A")

    if len(final_yaw_errors) > 0:
        print(f"Mean final yaw_error: {np.mean(final_yaw_errors):.4f}")
    else:
        print("Mean final yaw_error: N/A")

    if len(final_stable_steps) > 0:
        print(f"Mean final stable_pose_steps: {np.mean(final_stable_steps):.2f}")
    else:
        print("Mean final stable_pose_steps: N/A")


if __name__ == "__main__":
    main()