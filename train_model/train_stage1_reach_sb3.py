# train_model/train_stage1_reach_sb3.py
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from typing import Optional, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from env.reach_env import ReachEnv
from config.reach_env_config import EnvConfig
from script.stage1_reach.tensosbroad_callbacks import ReachTensorboardCallback
from script.stage1_reach.control_callbacks import (
    SaveLatestOnStopCallback,
    StopTrainingGracefully,
)
from script.stage1_reach.debug_callbacks import ReachDebugLoggerCallback


TOTAL_TIMESTEPS_PER_STAGE = 3_000_000
N_ENVS = 8
TARGET_CKPT_STEPS = 100_000  # muốn mỗi 100k total timesteps lưu 1 lần


def make_env(substage="1A"):
    def _thunk():
        cfg = EnvConfig(stage1_substage=substage)
        env = ReachEnv(use_gui=False, config=cfg)
        return Monitor(env)
    return _thunk


def build_env(substage: str):
    env = SubprocVecEnv([make_env(substage) for _ in range(N_ENVS)])
    env = VecTransposeImage(env)
    return env


def build_new_model(env):
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
        tensorboard_log="logs_stage1",
    )


def load_model(model_path: str, env):
    print(f"[RESUME] Loading model: {model_path}")
    model = PPO.load(model_path, env=env)
    return model


def train_one_substage(substage: str, model: Optional[PPO] = None) -> Tuple[PPO, bool]:
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs_stage1", exist_ok=True)
    os.makedirs("debug_logs", exist_ok=True)
    os.makedirs(f"models/checkpoints_stage1_{substage}", exist_ok=True)

    env = build_env(substage)

    if model is None:
        model = build_new_model(env)
    else:
        model.set_env(env)

    latest_path = f"models/stage1_reach_{substage}_latest"
    final_path = f"models/stage1_reach_{substage}"

    ckpt = CheckpointCallback(
        save_freq=max(TARGET_CKPT_STEPS // env.num_envs, 1),
        save_path=f"models/checkpoints_stage1_{substage}",
        name_prefix=f"ppo_stage1_{substage}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    tb = ReachTensorboardCallback()

    latest_on_stop_cb = SaveLatestOnStopCallback(
        save_path=latest_path,
        verbose=1,
    )

    debug_cb = ReachDebugLoggerCallback(
        debug_csv_path=f"debug_logs/stage1_{substage}_debug.csv",
        summary_csv_path=f"debug_logs/stage1_{substage}_summary.csv",
        n_envs=env.num_envs,
        verbose=0,
    )

    callbacks = CallbackList([tb, ckpt, latest_on_stop_cb, debug_cb])

    finished_normally = False

    try:
        print(f"\n===== TRAIN {substage} START =====")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS_PER_STAGE,
            callback=callbacks,
            reset_num_timesteps=False,
        )

        model.save(final_path)
        print(f"[OK] Saved final model: {final_path}.zip")
        finished_normally = True

        # stage đã hoàn tất thì xóa latest cũ để tránh resume nhầm
        latest_zip = latest_path + ".zip"
        if os.path.exists(latest_zip):
            os.remove(latest_zip)
            print(f"[CLEAN] Removed old latest file: {latest_zip}")

    except StopTrainingGracefully:
        print(f"[STOP] Stage {substage} stopped safely after saving latest.")

    except KeyboardInterrupt:
        # fallback nếu tín hiệu không đi qua callback
        print(f"\n[INTERRUPT] Stage {substage} stopped by user.")
        model.save(latest_path)
        print(f"[OK] Saved latest model: {latest_path}.zip")

    finally:
        env.close()

    return model, finished_normally


def resolve_resume_state():
    path_1a_latest = "models/stage1_reach_1A_latest.zip"
    path_1a_final = "models/stage1_reach_1A.zip"
    path_1b_latest = "models/stage1_reach_1B_latest.zip"
    path_1b_final = "models/stage1_reach_1B.zip"
    path_1c_latest = "models/stage1_reach_1C_latest.zip"
    path_1c_final = "models/stage1_reach_1C.zip"

    # 1) đang dở 1C -> resume 1C
    if os.path.exists(path_1c_latest):
        return {
            "mode": "resume_1c",
            "substage": "1C",
            "model_path": path_1c_latest,
        }

    # 2) 1C đã xong -> kết thúc
    if os.path.exists(path_1c_final):
        return {
            "mode": "done",
            "substage": "1C",
            "model_path": path_1c_final,
        }

    # 3) đang dở 1B -> resume 1B
    if os.path.exists(path_1b_latest):
        return {
            "mode": "resume_1b",
            "substage": "1B",
            "model_path": path_1b_latest,
        }

    # 4) 1B đã xong -> bắt đầu 1C từ model 1B final
    if os.path.exists(path_1b_final):
        return {
            "mode": "start_1c_from_1b_final",
            "substage": "1C",
            "model_path": path_1b_final,
        }

    # 5) đang dở 1A -> resume 1A
    if os.path.exists(path_1a_latest):
        return {
            "mode": "resume_1a",
            "substage": "1A",
            "model_path": path_1a_latest,
        }

    # 6) 1A đã xong -> bắt đầu 1B từ model 1A final
    if os.path.exists(path_1a_final):
        return {
            "mode": "start_1b_from_1a_final",
            "substage": "1B",
            "model_path": path_1a_final,
        }

    # 7) chưa có gì -> train mới từ 1A
    return {
        "mode": "fresh_start",
        "substage": "1A",
        "model_path": None,
    }

if __name__ == "__main__":
    state = resolve_resume_state()
    print(f"[STATE] {state['mode']}")

    # ===== CASE 1: đang dở 1C -> resume 1C =====
    if state["mode"] == "resume_1c":
        env = build_env("1C")
        model = load_model(state["model_path"], env)
        env.close()

        model, finished_1c = train_one_substage("1C", model=model)

    # ===== CASE 2: 1C đã xong =====
    elif state["mode"] == "done":
        print("[DONE] Stage 1C already completed. Nothing to train.")

    # ===== CASE 3: đang dở 1B -> resume 1B, xong mới sang 1C =====
    elif state["mode"] == "resume_1b":
        env = build_env("1B")
        model = load_model(state["model_path"], env)
        env.close()

        model, finished_1b = train_one_substage("1B", model=model)

        if finished_1b:
            model, finished_1c = train_one_substage("1C", model=model)

    # ===== CASE 4: 1B đã xong -> bắt đầu 1C từ model 1B final =====
    elif state["mode"] == "start_1c_from_1b_final":
        env = build_env("1C")
        model = load_model(state["model_path"], env)
        env.close()

        model, finished_1c = train_one_substage("1C", model=model)

    # ===== CASE 5: đang dở 1A -> resume 1A, xong mới sang 1B rồi 1C =====
    elif state["mode"] == "resume_1a":
        env = build_env("1A")
        model = load_model(state["model_path"], env)
        env.close()

        model, finished_1a = train_one_substage("1A", model=model)

        if finished_1a:
            model, finished_1b = train_one_substage("1B", model=model)

            if finished_1b:
                model, finished_1c = train_one_substage("1C", model=model)

    # ===== CASE 6: 1A đã xong -> bắt đầu 1B từ model 1A final, rồi sang 1C =====
    elif state["mode"] == "start_1b_from_1a_final":
        env = build_env("1B")
        model = load_model(state["model_path"], env)
        env.close()

        model, finished_1b = train_one_substage("1B", model=model)

        if finished_1b:
            model, finished_1c = train_one_substage("1C", model=model)

    # ===== CASE 7: train mới từ đầu =====
    elif state["mode"] == "fresh_start":
        model, finished_1a = train_one_substage("1A", model=None)

        if finished_1a:
            model, finished_1b = train_one_substage("1B", model=model)

            if finished_1b:
                model, finished_1c = train_one_substage("1C", model=model)