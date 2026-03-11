# train_stage3_place_sb3.py
import os
import sys
import json
import argparse
import multiprocessing as mp

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
torch.set_num_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

from env.place_env import PlaceEnv
from script.callbacks import PlaceTensorboardCallback


NUM_ENVS = 8
TIMESTEPS_PER_STAGE = 500_000

MODEL_DIR = "models"
LOG_DIR = "logs_stage3"

MODEL_PATH = os.path.join(MODEL_DIR, "stage3_place_latest")
STATE_PATH = os.path.join(MODEL_DIR, "stage3_place_state.json")

VALID_STAGES = ["3A", "3B", "3C"]


def make_env(substage="3A"):
    def _thunk():
        env = PlaceEnv(use_gui=False, start_held=True, substage=substage)
        return Monitor(env)
    return _thunk


def build_env(substage):
    env = SubprocVecEnv([make_env(substage) for _ in range(NUM_ENVS)])
    env = VecTransposeImage(env)
    return env


def default_state():
    return {
        "done": {
            "3A": False,
            "3B": False,
            "3C": False
        }
    }


def load_state():
    if not os.path.exists(STATE_PATH):
        return default_state()

    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)

    if "done" not in state:
        state["done"] = {"3A": False, "3B": False, "3C": False}

    for s in VALID_STAGES:
        if s not in state["done"]:
            state["done"][s] = False

    return state


def save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def print_stage_status(state):
    done = state.get("done", {})

    print("\n===== TRẠNG THÁI CÁC STAGE =====")
    for s in VALID_STAGES:
        status = "DONE" if done.get(s, False) else "CHƯA DONE"
        print(f"{s}: {status}")
    print("================================\n")


def print_next_action_hint(stage, state):
    done = state.get("done", {})

    print("Gợi ý hiện tại:")
    for s in VALID_STAGES:
        status = "DONE" if done.get(s, False) else "CHƯA DONE"
        marker = " <= stage đang chọn" if s == stage else ""
        print(f"- {s}: {status}{marker}")
    print()


def create_new_model(env):
    print("[NEW] Tạo model mới")
    return PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=LOG_DIR,
    )


def load_or_create_model(env):
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"[LOAD] Load model cũ: {MODEL_PATH}.zip")
        return PPO.load(MODEL_PATH, env=env)
    return create_new_model(env)


if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        type=str,
        choices=VALID_STAGES,
        help="Chọn stage muốn train: 3A, 3B hoặc 3C"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Train lại stage này dù state đang đánh dấu done"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Chỉ xem trạng thái DONE/CHƯA DONE của 3A, 3B, 3C rồi thoát"
    )
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    state = load_state()

    if args.status:
        print_stage_status(state)
        sys.exit(0)

    if not args.stage:
        print("Thiếu --stage. Ví dụ: python train_stage3_place_sb3.py --stage 3A")
        sys.exit(1)

    stage = args.stage

    print_stage_status(state)
    print_next_action_hint(stage, state)

    if state["done"].get(stage, False) and not args.force:
        print(f"[SKIP] Stage {stage} đã train xong 500000 rồi, không train lại.")
        print("Nếu muốn train lại stage này, chạy thêm --force")
        sys.exit(0)

    print("=" * 60)
    print(f"Stage sẽ train: {stage}")
    print(f"Số env: {NUM_ENVS}")
    print(f"Train đúng 1 lần: {TIMESTEPS_PER_STAGE} steps")
    print(f"Model lưu đè tại: {MODEL_PATH}.zip")
    print("=" * 60)

    env = build_env(stage)
    model = load_or_create_model(env)
    model.set_env(env)

    callback = PlaceTensorboardCallback()

    model.learn(
        total_timesteps=TIMESTEPS_PER_STAGE,
        callback=callback,
        reset_num_timesteps=False,
        tb_log_name=f"place_{stage}",
    )

    model.save(MODEL_PATH)
    print(f"[SAVE] Đã ghi đè model tại: {MODEL_PATH}.zip")

    state["done"][stage] = True
    save_state(state)

    env.close()

    print(f"[DONE] Stage {stage} đã train xong đúng 500000 steps.")
    print_stage_status(state)