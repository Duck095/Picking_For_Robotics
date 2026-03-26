from __future__ import annotations

import os
import sys
from typing import Callable, Optional, Tuple

# ============================================================
# THREAD SETTINGS
# ------------------------------------------------------------
# Giảm tranh chấp CPU khi dùng nhiều env/process
# ============================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from script.stage1_reach.control_callbacks import SaveLatestOnStopCallback, request_stop, STOP_REQUESTED
from script.stage1_reach.debug_step_callback import ReachDebugStepCallback
from script.stage1_reach.debug_summary_callback import ReachDebugSummaryCallback
from script.stage1_reach.tensosbroad_callbacks import ReachTensorboardCallback

from config.reach_env_config import build_reach_config
from env.reach_env import ReachEnv


# ============================================================
# TRAIN CONFIG
# ============================================================
TOTAL_TIMESTEPS = 3_000_000
N_ENVS = 8

MODEL_DIR = "models"
DEBUG_DIR = "debug_logs"
TENSORBOARD_DIR = "logs_stage1"

SAVE_FREQ_STEPS = 25_000

USE_SUBPROC = True
USE_GUI = False
SEED = 42


# ============================================================
# PATH HELPERS
# ============================================================
def latest_path(substage: str) -> str:
    return os.path.join(MODEL_DIR, f"stage1_reach_{substage}_latest.zip")


def final_path(substage: str) -> str:
    return os.path.join(MODEL_DIR, f"stage1_reach_{substage}.zip")


def checkpoint_dir(substage: str) -> str:
    return os.path.join(MODEL_DIR, f"checkpoints_stage1_{substage}")


# ============================================================
# ENV BUILDERS
# ============================================================
def make_env(substage: str, rank: int, use_gui: bool = False) -> Callable[[], ReachEnv]:
    """
    Factory tạo env cho vectorized training.
    """
    def _init():
        cfg = build_reach_config(substage)
        cfg.sim.use_gui = use_gui
        cfg.sim.seed = SEED + rank
        env = ReachEnv(cfg)
        return env

    return _init


def build_vec_env(substage: str, n_envs: int, use_subproc: bool = True, use_gui: bool = False):
    """
    Tạo vectorized env.

    Debug:
        USE_SUBPROC = False, N_ENVS = 1

    Train thật:
        USE_SUBPROC = True, N_ENVS > 1
    """
    env_fns = [make_env(substage=substage, rank=i, use_gui=use_gui) for i in range(n_envs)]

    if use_subproc and n_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


# ============================================================
# CALLBACKS
# ============================================================
def build_callbacks(model_dir: str, debug_dir: str, substage: str):
    """
    Callback list:
    - checkpoint_cb: lưu checkpoint định kỳ
    - control_cb: lưu latest khi stop, final khi train xong bình thường
    - debug_step_cb: log step-level gọn
    - debug_summary_cb: log summary-level gọn
    - tensorboard_cb: log tensorboard tối giản
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(checkpoint_dir(substage), exist_ok=True)

    control_cb = SaveLatestOnStopCallback(
        save_dir=model_dir,
        latest_name=f"stage1_reach_{substage}_latest",
        final_name=f"stage1_reach_{substage}",
        save_final_on_training_end=True,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(SAVE_FREQ_STEPS // N_ENVS, 1),
        save_path=checkpoint_dir(substage),
        name_prefix=f"ppo_stage1_{substage}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    debug_step_cb = ReachDebugStepCallback(
        log_dir=debug_dir,
        file_name=f"stage1_{substage}_debug.log",
        print_freq=5000,
        verbose=1,
    )

    debug_summary_cb = ReachDebugSummaryCallback(
        log_dir=debug_dir,
        file_name=f"stage1_{substage}_summary.log",
        window_size=100,
        print_freq=10000,
        verbose=1,
    )

    tensorboard_cb = ReachTensorboardCallback(
        window_size=100,
        verbose=0,
    )

    return CallbackList([
        checkpoint_cb,
        control_cb,
        debug_step_cb,
        debug_summary_cb,
        tensorboard_cb,
    ])


# ============================================================
# RESUME / CURRICULUM LOGIC
# ------------------------------------------------------------
# Logic:
# - latest_1A  -> resume 1A
# - final_1A   -> start 1B
# - latest_1B  -> resume 1B
# - final_1B   -> start 1C
# - latest_1C  -> resume 1C
# - final_1C   -> done
# - nothing    -> fresh start 1A
# ============================================================
def resolve_resume_state():
    path_1a_latest = latest_path("1A")
    path_1a_final = final_path("1A")

    path_1b_latest = latest_path("1B")
    path_1b_final = final_path("1B")

    path_1c_latest = latest_path("1C")
    path_1c_final = final_path("1C")

    # 1) đang dở 1C -> resume 1C
    if os.path.exists(path_1c_latest):
        return {"mode": "resume_1c", "substage": "1C", "model_path": path_1c_latest}

    # 2) 1C đã xong -> done
    if os.path.exists(path_1c_final):
        return {"mode": "done", "substage": "1C", "model_path": path_1c_final}

    # 3) đang dở 1B -> resume 1B
    if os.path.exists(path_1b_latest):
        return {"mode": "resume_1b", "substage": "1B", "model_path": path_1b_latest}

    # 4) 1B đã xong -> start 1C từ final 1B
    if os.path.exists(path_1b_final):
        return {"mode": "start_1c_from_1b_final", "substage": "1C", "model_path": path_1b_final}

    # 5) đang dở 1A -> resume 1A
    if os.path.exists(path_1a_latest):
        return {"mode": "resume_1a", "substage": "1A", "model_path": path_1a_latest}

    # 6) 1A đã xong -> start 1B từ final 1A
    if os.path.exists(path_1a_final):
        return {"mode": "start_1b_from_1a_final", "substage": "1B", "model_path": path_1a_final}

    # 7) chưa có gì -> train mới từ 1A
    return {"mode": "fresh_start", "substage": "1A", "model_path": None}


# ============================================================
# MODEL BUILD / LOAD
# ============================================================
def build_new_model(env):
    """
    Tạo model mới cho Stage 1.
    Vì ReachEnv đang dùng vector observation -> MlpPolicy
    """
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=TENSORBOARD_DIR,
        seed=SEED,
    )


def load_model(model_path: str, env):
    """
    Resume model từ checkpoint.
    """
    print(f"[RESUME] Loading model: {model_path}")
    return PPO.load(model_path, env=env)


# ============================================================
# TRAIN ONE SUBSTAGE
# ============================================================
def train_one_substage(substage: str, model: Optional[PPO] = None) -> Tuple[PPO, bool]:
    """
    Train một substage.

    Returns
    -------
    model:
        model hiện tại
    finished_normally:
        True nếu learn kết thúc bình thường
        False nếu bị Ctrl+C / stop giữa chừng
    """
    print("=" * 80)
    print(f"TRAIN STAGE 1 - REACH - SUBSTAGE {substage}")
    print("=" * 80)
    print(f"TOTAL_TIMESTEPS  : {TOTAL_TIMESTEPS}")
    print(f"N_ENVS           : {N_ENVS}")
    print(f"TENSORBOARD_DIR  : {TENSORBOARD_DIR}")
    print(f"USE_SUBPROC      : {USE_SUBPROC}")
    print(f"USE_GUI          : {USE_GUI}")
    print("=" * 80)

    env = build_vec_env(
        substage=substage,
        n_envs=N_ENVS,
        use_subproc=USE_SUBPROC,
        use_gui=USE_GUI,
    )

    callbacks = build_callbacks(
        model_dir=MODEL_DIR,
        debug_dir=DEBUG_DIR,
        substage=substage,
    )

    if model is None:
        model = build_new_model(env)
    else:
        model.set_env(env)

    finished_normally = False
    latest_save_base = os.path.join(MODEL_DIR, f"stage1_reach_{substage}_latest")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            reset_num_timesteps=False,
            tb_log_name=f"reach_{substage}",
            progress_bar=True,
        )
        finished_normally = True

        # Nếu train kết thúc bình thường, final sẽ được callback lưu.
        # Sau đó xóa latest cũ nếu có để lần sau không resume nhầm.
        lp = latest_path(substage)
        if os.path.exists(lp):
            try:
                os.remove(lp)
                print(f"[CLEAN] Removed stale latest file: {lp}")
            except Exception as e:
                print(f"[WARN] Could not remove latest file {lp}: {e}")

    except (KeyboardInterrupt, InterruptedError, EOFError, BrokenPipeError) as e:
        # Fallback cực quan trọng khi Ctrl+C + SubprocVecEnv trên Windows
        print(f"\n[INTERRUPT] Stage {substage} interrupted by user: type={type(e).__name__} message={e}")
        try:
            model.save(latest_save_base)
            print(f"[OK] Saved latest model: {latest_save_base}.zip")
        except Exception as e:
            print(f"[ERROR] Failed to save latest model: {e}")

    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[WARN] env.close() raised during shutdown: {e}")
        print("[TRAIN] Env closed.")

    return model, finished_normally


# ============================================================
# MAIN
# ============================================================
def main():
    state = resolve_resume_state()
    print(f"[STATE] {state['mode']}")

    # ===== CASE 1: 1C đã xong =====
    if state["mode"] == "done":
        print("[DONE] Stage 1C already completed. Nothing to train.")
        return

    # ===== CASE 2: đang dở 1C -> resume 1C =====
    if state["mode"] == "resume_1c":
        env = build_vec_env("1C", n_envs=N_ENVS, use_subproc=USE_SUBPROC, use_gui=USE_GUI)
        model = load_model(state["model_path"], env)
        env.close()

        train_one_substage("1C", model=model)
        return

    # ===== CASE 3: đang dở 1B -> resume 1B, xong mới sang 1C =====
    if state["mode"] == "resume_1b":
        env = build_vec_env("1B", n_envs=N_ENVS, use_subproc=USE_SUBPROC, use_gui=USE_GUI)
        model = load_model(state["model_path"], env)
        env.close()

        model, finished_1b = train_one_substage("1B", model=model)
        if finished_1b:
            train_one_substage("1C", model=model)
        return

    # ===== CASE 4: 1B đã xong -> bắt đầu 1C từ model 1B final =====
    if state["mode"] == "start_1c_from_1b_final":
        env = build_vec_env("1C", n_envs=N_ENVS, use_subproc=USE_SUBPROC, use_gui=USE_GUI)
        model = load_model(state["model_path"], env)
        env.close()

        train_one_substage("1C", model=model)
        return

    # ===== CASE 5: đang dở 1A -> resume 1A, xong mới sang 1B rồi 1C =====
    if state["mode"] == "resume_1a":
        env = build_vec_env("1A", n_envs=N_ENVS, use_subproc=USE_SUBPROC, use_gui=USE_GUI)
        model = load_model(state["model_path"], env)
        env.close()

        model, finished_1a = train_one_substage("1A", model=model)
        if finished_1a:
            model, finished_1b = train_one_substage("1B", model=model)
            if finished_1b:
                train_one_substage("1C", model=model)
        return

    # ===== CASE 6: 1A đã xong -> bắt đầu 1B từ model 1A final, rồi sang 1C =====
    if state["mode"] == "start_1b_from_1a_final":
        env = build_vec_env("1B", n_envs=N_ENVS, use_subproc=USE_SUBPROC, use_gui=USE_GUI)
        model = load_model(state["model_path"], env)
        env.close()

        model, finished_1b = train_one_substage("1B", model=model)
        if finished_1b:
            train_one_substage("1C", model=model)
        return

    # ===== CASE 7: train mới từ đầu =====
    if state["mode"] == "fresh_start":
        model, finished_1a = train_one_substage("1A", model=None)
        if finished_1a:
            model, finished_1b = train_one_substage("1B", model=model)
            if finished_1b:
                train_one_substage("1C", model=model)
        return


if __name__ == "__main__":
    main()