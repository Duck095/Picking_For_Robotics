# test/test_reach_env.py
import sys
import os
import glob
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
import numpy as np
from env.reach_env import ReachEnv
from config.reach_env_config import EnvConfig

# =========================
# CONFIG
# =========================
USE_GUI = True
N_EPISODES = 10
SLEEP_SEC = 1.0 if USE_GUI else 0.0

# Thứ tự ưu tiên mặc định nếu user chỉ bấm Enter
MODEL_PRIORITY = [
    "models/stage1_reach_1C.zip",
    "models/stage1_reach_1C_latest.zip",
    "models/stage1_reach_1B.zip",
    "models/stage1_reach_1B_latest.zip",
    "models/stage1_reach_1A.zip",
    "models/stage1_reach_1A_latest.zip",
]


def get_available_models():
    models = []

    # 1) file chính
    for path in MODEL_PRIORITY:
        if os.path.exists(path):
            models.append(path)

    # 2) checkpoint 1B
    ckpt_1b = sorted(
        glob.glob("models/checkpoints_stage1_1B/*.zip"),
        reverse=True
    )

    # 3) checkpoint 1A
    ckpt_1a = sorted(
        glob.glob("models/checkpoints_stage1_1A/*.zip"),
        reverse=True
    )

    models.extend(ckpt_1b)
    models.extend(ckpt_1a)

    return models


def choose_model(available_models):
    print("\n[INFO] Available trained models:")
    for i, path in enumerate(available_models, start=1):
        print(f"  {i}. {path}")

    print("\nChọn cách test:")
    print("  - Nhập số để chọn model cụ thể")
    print("  - Bấm Enter để dùng model ưu tiên cao nhất")
    print("  - Nhập 'r' để chạy random test")

    choice = input("Lựa chọn của bạn: ").strip().lower()

    if choice == "r":
        return None, "random"

    if choice == "":
        return available_models[0], "model"

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(available_models):
            return available_models[idx], "model"

    print("[WARN] Lựa chọn không hợp lệ, dùng model ưu tiên cao nhất.")
    return available_models[0], "model"

def infer_stage1_substage(model_path: str):
    path = model_path.lower().replace("\\", "/")
    if "1c" in path:
        return "1C"
    elif "1b" in path:
        return "1B"
    elif "1a" in path:
        return "1A"
    else:
        return "1A"

def main():
    available_models = get_available_models()

    if len(available_models) == 0:
        print("[INFO] No trained model found.")
        print("[MODE] Random action test (system/setup check only)")
        model = None
        test_mode = "random"
        model_path = None
    else:
        model_path, test_mode = choose_model(available_models)

        if test_mode == "model":
            from stable_baselines3 import PPO
            print(f"\n[INFO] Selected model: {model_path}")
            print("[MODE] Test trained model")
            model = PPO.load(model_path)
        else:
            print("\n[MODE] Random action test (system/setup check only)")
            model = None

    if test_mode == "model" and model_path is not None:
        selected_substage = infer_stage1_substage(model_path)
    else:
        selected_substage = "1A"

    cfg = EnvConfig(stage1_substage=selected_substage)
    env = ReachEnv(use_gui=USE_GUI, config=cfg)

    print(f"[DEBUG] selected_substage = {selected_substage}")
    print(f"[DEBUG] env.cfg.stage1_substage = {env.cfg.stage1_substage}")
    print(f"[DEBUG] reward success_dist = {env.rewarder.success_dist}")

    success_count = 0
    episode_rewards = []
    final_dists = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        last_dist = None
        step = 0
        info = {}

        print("\n" + "=" * 70)
        print(f"Episode {ep + 1}/{N_EPISODES}")

        while not (done or truncated):
            if test_mode == "model":
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)

            ep_reward += reward
            last_dist = info.get("ee_obj_dist", None)
            success = info.get("success", False)
            step += 1

            if last_dist is not None:
                print(
                    f"step={step:03d} | reward={reward:.3f} | "
                    f"dist={last_dist:.3f} | success={success}"
                )
            else:
                print(
                    f"step={step:03d} | reward={reward:.3f} | success={success}"
                )

            if success and USE_GUI:
                time.sleep(5.0)

            time.sleep(SLEEP_SEC)

        if info.get("success", False):
            success_count += 1

        episode_rewards.append(ep_reward)
        if last_dist is not None:
            final_dists.append(last_dist)

        print(
            f"[EP DONE] total_reward={ep_reward:.3f} | "
            f"final_dist={last_dist if last_dist is not None else 'N/A'} | "
            f"success={info.get('success', False)} | "
            f"terminated={done} | truncated={truncated}"
        )

    env.close()

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print(f"Mode: {'MODEL TEST' if test_mode == 'model' else 'RANDOM TEST'}")

    if model_path is not None:
        print(f"Model used: {model_path}")

    print(f"Episodes: {N_EPISODES}")
    print(f"Success rate: {success_count / N_EPISODES:.2f}")
    print(f"Mean reward: {np.mean(episode_rewards):.3f}")

    if USE_GUI:
        time.sleep(5.0)

    if len(final_dists) > 0:
        print(f"Mean final dist: {np.mean(final_dists):.3f}")
    else:
        print("Mean final dist: N/A")


if __name__ == "__main__":
    main()