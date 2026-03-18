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

from env.place_env import PlaceEnv

# Boss lưu ý: Import đúng đường dẫn file callback của Stage 3 nhé
# Nếu chưa có file debug_callbacks cho Place thì Boss có thể comment nó lại
from script.stage3_place.tensorboard_callbacks import PlaceTensorboardCallback
from script.stage1_reach.control_callbacks import (
    SaveLatestOnStopCallback,
    StopTrainingGracefully,
)
# from script.stage3_place.debug_callbacks import PlaceDebugLoggerCallback 


TOTAL_TIMESTEPS_PER_STAGE = 500_000
N_ENVS = 8
TARGET_CKPT_STEPS = 50_000  # Lưu checkpoint mỗi 50k timesteps


def make_env(substage="3A"):
    def _thunk():
        # Đảm bảo start_held=True để tay robot cầm sẵn vật ở Home như code môi trường mới nhất
        env = PlaceEnv(use_gui=False, start_held=True, substage=substage)
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
        tensorboard_log="logs_stage3",
    )


def load_model(model_path: str, env):
    print(f"[RESUME] Loading model: {model_path}")
    model = PPO.load(model_path, env=env)
    return model


def train_one_substage(substage: str, model: Optional[PPO] = None) -> Tuple[PPO, bool]:
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs_stage3", exist_ok=True)
    os.makedirs("debug_logs_stage3", exist_ok=True)
    os.makedirs(f"models/checkpoints_stage3_{substage}", exist_ok=True)

    env = build_env(substage)

    if model is None:
        model = build_new_model(env)
    else:
        model.set_env(env)

    latest_path = f"models/stage3_place_{substage}_latest"
    final_path = f"models/stage3_place_{substage}"

    ckpt = CheckpointCallback(
        save_freq=max(TARGET_CKPT_STEPS // env.num_envs, 1),
        save_path=f"models/checkpoints_stage3_{substage}",
        name_prefix=f"ppo_stage3_{substage}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    tb = PlaceTensorboardCallback()

    latest_on_stop_cb = SaveLatestOnStopCallback(
        save_path=latest_path,
        verbose=1,
    )

    # Nếu Boss có file debug cho Place thì mở comment đoạn này và add vào CallbackList
    # debug_cb = PlaceDebugLoggerCallback(
    #     debug_csv_path=f"debug_logs_stage3/stage3_{substage}_debug.csv",
    #     summary_csv_path=f"debug_logs_stage3/stage3_{substage}_summary.csv",
    #     n_envs=env.num_envs,
    #     verbose=0,
    # )

    callbacks = CallbackList([tb, ckpt, latest_on_stop_cb])

    finished_normally = False

    try:
        print(f"\n===== TRAIN {substage} START =====")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS_PER_STAGE,
            callback=callbacks,
            reset_num_timesteps=False,
            tb_log_name=f"place_{substage}"
        )

        model.save(final_path)
        print(f"[OK] Saved final model: {final_path}.zip")
        finished_normally = True

        latest_zip = latest_path + ".zip"
        if os.path.exists(latest_zip):
            os.remove(latest_zip)
            print(f"[CLEAN] Removed old latest file: {latest_zip}")

    except StopTrainingGracefully:
        print(f"[STOP] Stage {substage} stopped safely after saving latest.")

    except KeyboardInterrupt:
        print(f"\n[INTERRUPT] Stage {substage} stopped by user.")
        model.save(latest_path)
        print(f"[OK] Saved latest model: {latest_path}.zip")

    finally:
        env.close()

    return model, finished_normally


def resolve_resume_state():
    path_3a_latest = "models/stage3_place_3A_latest.zip"
    path_3a_final = "models/stage3_place_3A.zip"
    path_3b_latest = "models/stage3_place_3B_latest.zip"
    path_3b_final = "models/stage3_place_3B.zip"
    path_3c_latest = "models/stage3_place_3C_latest.zip"
    path_3c_final = "models/stage3_place_3C.zip"

    if os.path.exists(path_3c_latest): return {"mode": "resume_3c", "substage": "3C", "model_path": path_3c_latest}
    if os.path.exists(path_3c_final): return {"mode": "done", "substage": "3C", "model_path": path_3c_final}
    if os.path.exists(path_3b_latest): return {"mode": "resume_3b", "substage": "3B", "model_path": path_3b_latest}
    if os.path.exists(path_3b_final): return {"mode": "start_3c_from_3b_final", "substage": "3C", "model_path": path_3b_final}
    if os.path.exists(path_3a_latest): return {"mode": "resume_3a", "substage": "3A", "model_path": path_3a_latest}
    if os.path.exists(path_3a_final): return {"mode": "start_3b_from_3a_final", "substage": "3B", "model_path": path_3a_final}
    
    return {"mode": "fresh_start", "substage": "3A", "model_path": None}


if __name__ == "__main__":
    state = resolve_resume_state()
    print(f"[STATE] {state['mode']}")

    if state["mode"] == "resume_3c":
        env = build_env("3C")
        model = load_model(state["model_path"], env)
        env.close()
        model, finished_3c = train_one_substage("3C", model=model)

    elif state["mode"] == "done":
        print("[DONE] Stage 3C already completed. Nothing to train.")

    elif state["mode"] == "resume_3b":
        env = build_env("3B")
        model = load_model(state["model_path"], env)
        env.close()
        model, finished_3b = train_one_substage("3B", model=model)
        if finished_3b:
            model, finished_3c = train_one_substage("3C", model=model)

    elif state["mode"] == "start_3c_from_3b_final":
        env = build_env("3C")
        model = load_model(state["model_path"], env)
        env.close()
        model, finished_3c = train_one_substage("3C", model=model)

    elif state["mode"] == "resume_3a":
        env = build_env("3A")
        model = load_model(state["model_path"], env)
        env.close()
        model, finished_3a = train_one_substage("3A", model=model)
        if finished_3a:
            model, finished_3b = train_one_substage("3B", model=model)
            if finished_3b:
                model, finished_3c = train_one_substage("3C", model=model)

    elif state["mode"] == "start_3b_from_3a_final":
        env = build_env("3B")
        model = load_model(state["model_path"], env)
        env.close()
        model, finished_3b = train_one_substage("3B", model=model)
        if finished_3b:
            model, finished_3c = train_one_substage("3C", model=model)

    elif state["mode"] == "fresh_start":
        model, finished_3a = train_one_substage("3A", model=None)
        if finished_3a:
            model, finished_3b = train_one_substage("3B", model=model)
            if finished_3b:
                model, finished_3c = train_one_substage("3C", model=model)