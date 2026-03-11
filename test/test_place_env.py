import os
import sys
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from env.reach_env import ReachEnv


def make_env(use_gui=True):
    def _thunk():
        return ReachEnv(use_gui=use_gui)
    return _thunk


def find_default_model():
    candidates = [
        "models/stage1_reach_latest.zip",
        "models/stage1_reach_latest",
        "models/stage1_reach_1B.zip",
        "models/stage1_reach_1B",
        "models/stage1_reach_1A.zip",
        "models/stage1_reach_1A",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def run_random(use_gui=True, steps=5000, sleep_time=1 / 60):
    env = ReachEnv(use_gui=use_gui)
    obs, info = env.reset()

    for step in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"[RANDOM] step={step} "
            f"reward={reward:.3f} "
            f"dist={info.get('ee_obj_dist', -1):.3f} "
            f"success={info.get('success', False)}"
        )

        if terminated or truncated:
            print("RESET\n")
            obs, info = env.reset()

        if sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


def run_model(model_path, use_gui=True, steps=5000, sleep_time=1 / 60, deterministic=True):
    vec_env = DummyVecEnv([make_env(use_gui=use_gui)])
    vec_env = VecTransposeImage(vec_env)

    model = PPO.load(model_path, env=vec_env)
    obs = vec_env.reset()

    for step in range(steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = vec_env.step(action)

        reward = float(rewards[0])
        done = bool(dones[0])
        info = infos[0]

        print(
            f"[MODEL] step={step} "
            f"reward={reward:.3f} "
            f"dist={info.get('ee_obj_dist', -1):.3f} "
            f"success={info.get('success', False)}"
        )

        if done:
            print("RESET\n")

        if sleep_time > 0:
            time.sleep(sleep_time)

    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Đường dẫn model PPO")
    parser.add_argument("--steps", type=int, default=5000, help="Số bước chạy test")
    parser.add_argument("--no-gui", action="store_true", help="Chạy không mở GUI")
    parser.add_argument("--stochastic", action="store_true", help="Predict không deterministic")
    parser.add_argument("--sleep", type=float, default=1 / 60, help="Thời gian nghỉ giữa mỗi step")
    parser.add_argument("--random", action="store_true", help="Ép chạy random dù có model")
    args = parser.parse_args()

    use_gui = not args.no_gui
    model_path = args.model if args.model else find_default_model()

    if args.random:
        print("Ép chạy random action.")
        run_random(
            use_gui=use_gui,
            steps=args.steps,
            sleep_time=args.sleep,
        )
    elif model_path and os.path.exists(model_path):
        print(f"Đang chạy với model: {model_path}")
        run_model(
            model_path=model_path,
            use_gui=use_gui,
            steps=args.steps,
            sleep_time=args.sleep,
            deterministic=not args.stochastic,
        )
    else:
        print("Không tìm thấy model, chạy random action.")
        run_random(
            use_gui=use_gui,
            steps=args.steps,
            sleep_time=args.sleep,
        )