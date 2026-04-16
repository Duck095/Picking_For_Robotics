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
USE_REPRODUCIBLE_SEEDS = True
BASE_SEED = 42

VALID_SUBSTAGES = ["2A", "2B", "2C"]


def normalize_path(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def choose_substage() -> str:
    print("\n===== CHỌN SUBSTAGE STAGE 2 MUỐN TEST =====")
    for i, s in enumerate(VALID_SUBSTAGES, start=1):
        print(f"{i}. {s}")

    print("\nEnter số / Enter mặc định = 2A")
    choice = input("> ").strip()

    if choice == "":
        return "2A"

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(VALID_SUBSTAGES):
            return VALID_SUBSTAGES[idx]

    choice = choice.upper()
    if choice in VALID_SUBSTAGES:
        return choice

    print("Input không hợp lệ -> dùng mặc định 2A")
    return "2A"


def substage_to_tag(substage: str) -> str:
    return substage.replace(".", "_")


def get_available_models_for_substage(substage: str) -> List[str]:
    tag = substage_to_tag(substage)
    models: List[str] = []
    seen = set()

    def add_if_exists(path: str):
        norm = normalize_path(path)
        if os.path.exists(path) and norm not in seen:
            seen.add(norm)
            models.append(os.path.normpath(path))

    add_if_exists(f"models/stage2_grasp_mastery_{tag}.zip")
    add_if_exists(f"models/stage2_grasp_mastery_{tag}_latest.zip")
    add_if_exists(f"models/stage2_{tag}.zip")
    add_if_exists(f"models/stage2_{tag}_latest.zip")

    for pattern in [
        f"models/checkpoints_stage2_{tag}/*.zip",
        f"models/checkpoints_grasp_{tag}/*.zip",
    ]:
        for p in sorted(glob.glob(pattern), reverse=True):
            add_if_exists(p)

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


def print_phase_flow(substage: str):
    if substage == "2A":
        print("\n[FLOW 2A] xy_align -> descend -> close")
    elif substage == "2B":
        print("\n[FLOW 2B] xy_align -> descend -> close -> hold")
    elif substage == "2C":
        print("\n[FLOW 2C] xy_align -> descend -> close -> hold -> lift")


def compute_alignment_snapshot(env: GraspEnv) -> dict:
    ee_pos, _, ee_yaw = env._get_ee_pose()
    object_pos, object_yaw = env._get_object_pose()
    finger_mid = env._get_finger_midpoint()

    ee_xy_err = float(np.linalg.norm(ee_pos[:2] - object_pos[:2]))
    finger_xy_err = float(np.linalg.norm(finger_mid[:2] - object_pos[:2]))
    ee_z_to_grasp = abs(float(ee_pos[2] - env.grasp_pos[2]))
    finger_z_to_cube = abs(float(finger_mid[2] - object_pos[2]))
    yaw_error = abs(env._wrap_angle(env.target_yaw - ee_yaw))

    return {
        "ee_pos": ee_pos,
        "finger_mid": finger_mid,
        "object_pos": object_pos,
        "object_yaw": float(object_yaw),
        "ee_xy_err": ee_xy_err,
        "finger_xy_err": finger_xy_err,
        "ee_z_to_grasp": ee_z_to_grasp,
        "finger_z_to_cube": finger_z_to_cube,
        "yaw_error": yaw_error,
        "xy_gap_ee_vs_finger": abs(ee_xy_err - finger_xy_err),
    }


def format_reset_info(info: dict, align: dict, episode_seed: Optional[int]) -> str:
    seed_txt = episode_seed if episode_seed is not None else "random"
    return (
        f"[RESET] seed={seed_txt} | "
        f"object={np.round(info.get('object_pos'), 4)} | "
        f"target={np.round(info.get('target_pos'), 4)} | "
        f"phase={info.get('phase')} | "
        f"pregrasp_z={info.get('pregrasp_z', 'N/A')} | "
        f"grasp_z={info.get('grasp_z', 'N/A')} | "
        f"ee_xy={align['ee_xy_err']:.4f} | "
        f"finger_xy={align['finger_xy_err']:.4f} | "
        f"gap={align['xy_gap_ee_vs_finger']:.4f} | "
        f"yaw_err={align['yaw_error']:.4f}"
    )


def inconsistency_note(info: dict) -> str:
    ee_xy = float(info.get("xy_dist", -1.0))
    finger_xy = float(info.get("finger_mid_to_cube_xy_error", -1.0))
    ee_z = float(info.get("ee_to_grasp_z_error", -1.0))
    finger_z = float(info.get("finger_mid_to_cube_z_error", -1.0))

    notes = []
    if ee_xy >= 0.0 and finger_xy >= 0.0 and abs(ee_xy - finger_xy) > 0.008:
        notes.append("XY(EE!=finger)")
    if ee_z >= 0.0 and finger_z >= 0.0 and abs(ee_z - finger_z) > 0.008:
        notes.append("Z(EE!=finger)")
    if info.get("grasp_established", False) and finger_xy > ee_xy + 0.006:
        notes.append("pass_by_EE")

    return ",".join(notes) if notes else "-"


def format_step_info(step: int, reward: float, info: dict) -> str:
    return (
        f"step={step:03d} | "
        f"phase={info.get('phase', 'N/A'):>8s} | "
        f"reward={reward:+.3f} | "
        f"xy_ee={info.get('xy_dist', -1):.3f} | "
        f"xy_finger={info.get('finger_mid_to_cube_xy_error', -1):.3f} | "
        f"z_ee={info.get('ee_to_grasp_z_error', -1):.3f} | "
        f"z_finger={info.get('finger_mid_to_cube_z_error', -1):.3f} | "
        f"yaw={info.get('yaw_error', -1):.3f} | "
        f"grip={info.get('grip_width', -1):.3f} | "
        f"contact=({int(info.get('left_contact', False))},{int(info.get('right_contact', False))}) | "
        f"grasp={info.get('grasp_established', False)} | "
        f"success={info.get('success', False)} | "
        f"warn={inconsistency_note(info)}"
    )


def summarize_episode(info: dict) -> str:
    return (
        f"[EP SUMMARY] "
        f"success={info.get('success', False)} | "
        f"phase={info.get('phase', 'N/A')} | "
        f"xy_ee={info.get('xy_dist', -1):.4f} | "
        f"xy_finger={info.get('finger_mid_to_cube_xy_error', -1):.4f} | "
        f"z_ee={info.get('ee_to_grasp_z_error', -1):.4f} | "
        f"z_finger={info.get('finger_mid_to_cube_z_error', -1):.4f} | "
        f"grasp={info.get('grasp_established', False)} | "
        f"warn={inconsistency_note(info)}"
    )


def main():
    substage = choose_substage()
    models = get_available_models_for_substage(substage)
    model_path, mode = choose_model(models, substage)

    model = None
    if mode == "model" and model_path is not None:
        print(f"\nUsing model: {model_path}")
        model = PPO.load(model_path)
    else:
        print(f"\nUsing RANDOM POLICY for substage {substage}")

    cfg = build_stage2_grasp_config(substage)
    cfg.sim.use_gui = USE_GUI
    cfg.sim.seed = BASE_SEED

    env = GraspEnv(cfg)
    print_phase_flow(substage)
    print(
        "\n[NOTE] File test này đã chỉnh để: "
        "(1) reset có seed reproducible, "
        "(2) log đồng thời EE error và finger-midpoint error, "
        "(3) cảnh báo khi env pass theo EE nhưng finger midpoint còn lệch."
    )

    try:
        for ep in range(N_EPISODES):
            episode_seed = cfg.sim.seed + ep if USE_REPRODUCIBLE_SEEDS else None
            if episode_seed is None:
                obs, info = env.reset()
            else:
                obs, info = env.reset(seed=episode_seed)

            align = compute_alignment_snapshot(env)
            done = False
            truncated = False
            total_reward = 0.0
            step = 0

            print(f"\n=== Episode {ep + 1} | substage={substage} ===")
            print(format_reset_info(info, align, episode_seed))

            while not (done or truncated):
                if model is not None:
                    action, _ = model.predict(obs, deterministic=DETERMINISTIC)
                else:
                    action = env.action_space.sample()

                obs, reward, done, truncated, info = env.step(action)
                total_reward += float(reward)
                step += 1

                print(format_step_info(step, float(reward), info))

                if USE_GUI:
                    time.sleep(SLEEP_SEC)

            print(
                f"[EP DONE] total_reward={total_reward:.3f} "
                f"success={info.get('success', False)} "
                f"truncated={truncated}"
            )
            print(summarize_episode(info))

    finally:
        env.close()


if __name__ == "__main__":
    main()
