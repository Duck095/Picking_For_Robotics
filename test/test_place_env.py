# test/test_place_env.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
import numpy as np
from env.place_env import PlaceEnv

# =========================
# CONFIG
# =========================
USE_GUI = True
N_EPISODES = 10
SLEEP_SEC = 1 / 2

# Thứ tự ưu tiên mặc định nếu user chỉ bấm Enter
MODEL_PRIORITY = [
    "models/stage3_place_latest.zip",
]


def get_available_models():
    models = []
    for path in MODEL_PRIORITY:
        if os.path.exists(path):
            models.append(path)
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


def get_place_dist(info):
    return info.get("ee_goal_dist", info.get("goal_dist", info.get("obj_goal_dist", None)))


def main():
    available_models = get_available_models()

    if len(available_models) == 0:
        print("[INFO] No trained model found.")
        print("[MODE] Random action test (system/setup check only)")
        model = None
        test_mode = "random"
        model_path = None
        env = PlaceEnv(use_gui=USE_GUI, start_held=True)
    else:
        model_path, test_mode = choose_model(available_models)

        if test_mode == "model":
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

            print(f"\n[INFO] Selected model: {model_path}")
            print("[MODE] Test trained model")

            vec_env = DummyVecEnv([
                lambda: PlaceEnv(use_gui=USE_GUI, start_held=True)
            ])
            vec_env = VecTransposeImage(vec_env)

            model = PPO.load(model_path, env=vec_env)
            env = vec_env
        else:
            print("\n[MODE] Random action test (system/setup check only)")
            model = None
            model_path = None
            env = PlaceEnv(use_gui=USE_GUI, start_held=True)

    success_count = 0
    episode_rewards = []
    final_dists = []

    for ep in range(N_EPISODES):
        if test_mode == "model":
            obs = env.reset()
        else:
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
                obs, rewards, dones, infos = env.step(action)

                reward = float(rewards[0])
                done = bool(dones[0])
                truncated = False
                info = infos[0]
            else:
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)

            ep_reward += reward
            last_dist = get_place_dist(info)
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

    if len(final_dists) > 0:
        print(f"Mean final dist: {np.mean(final_dists):.3f}")
    else:
        print("Mean final dist: N/A")


if __name__ == "__main__":
    main()