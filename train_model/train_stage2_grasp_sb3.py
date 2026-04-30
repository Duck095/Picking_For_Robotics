from __future__ import annotations

import os
import sys
from typing import Callable, Optional, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from script.stage2_grasp.control_callbacks import SaveLatestOnStopCallback
from script.stage2_grasp.grasp_debug_step_callback import GraspDebugStepCallback
from script.stage2_grasp.grasp_debug_summary_callback import GraspDebugSummaryCallback
from script.stage2_grasp.grasp_tensorboard_callback import GraspTensorboardCallback

from config.grasp_env_config import build_stage2_grasp_config
from env.grasp_env import GraspEnv


TOTAL_TIMESTEPS_PER_SUBSTAGE = {
    "2A": 500_000,
    "2B": 700_000,
    "2C": 900_000,
    "2D": 900_000,
}
SUBSTAGES = ["2A", "2B", "2C", "2D"]

N_ENVS = 8
USE_SUBPROC = True
USE_GUI = False
SEED = 42
MODEL_DIR = "models"
DEBUG_DIR = "debug_logs"
TENSORBOARD_DIR = "logs_stage2"
SAVE_FREQ_STEPS = 25_000


def substage_to_tag(substage: str) -> str:
    return substage.replace(".", "_")


def final_model_path(substage: str) -> str:
    return os.path.join(MODEL_DIR, f"stage2_grasp_mastery_{substage_to_tag(substage)}.zip")


def latest_model_path(substage: str) -> str:
    return os.path.join(MODEL_DIR, f"stage2_grasp_mastery_{substage_to_tag(substage)}_latest.zip")


def checkpoint_dir(substage: str) -> str:
    return os.path.join(MODEL_DIR, f"checkpoints_stage2_{substage_to_tag(substage)}")


def make_env(substage: str, rank: int, use_gui: bool = False) -> Callable[[], GraspEnv]:
    def _init():
        cfg = build_stage2_grasp_config(substage)
        cfg.sim.use_gui = use_gui
        cfg.sim.seed = SEED + rank
        return GraspEnv(cfg)
    return _init


def build_vec_env(substage: str, n_envs: int, use_subproc: bool = True, use_gui: bool = False):
    env_fns = [make_env(substage, i, use_gui) for i in range(n_envs)]
    if use_subproc and n_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def build_callbacks(substage: str):
    tag = substage_to_tag(substage)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(checkpoint_dir(substage), exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(SAVE_FREQ_STEPS // N_ENVS, 1),
        save_path=checkpoint_dir(substage),
        name_prefix=f"ppo_stage2_{tag}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    control_cb = SaveLatestOnStopCallback(
        save_dir=MODEL_DIR,
        latest_name=f"stage2_grasp_mastery_{tag}_latest",
        final_name=f"stage2_grasp_mastery_{tag}",
        save_final_on_training_end=True,
        verbose=1,
    )

    debug_step_cb = GraspDebugStepCallback(
        log_dir=DEBUG_DIR,
        file_name=f"stage2_{tag}_debug.log",
        print_freq=5000,
        verbose=1,
    )

    debug_summary_cb = GraspDebugSummaryCallback(
        log_dir=DEBUG_DIR,
        file_name=f"stage2_{tag}_summary.log",
        window_size=100,
        print_freq=10000,
        verbose=1,
    )

    tensorboard_cb = GraspTensorboardCallback(window_size=100, verbose=0)

    return CallbackList(
        [checkpoint_cb, control_cb, debug_step_cb, debug_summary_cb, tensorboard_cb]
    )


def build_new_model(env) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.008,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=TENSORBOARD_DIR,
        seed=SEED,
    )


def load_model(model_path: str, env) -> PPO:
    print(f"[LOAD] Loading model: {model_path}")
    return PPO.load(model_path, env=env)


def resolve_training_state():
    for sub in SUBSTAGES:
        lp = latest_model_path(sub)
        if os.path.exists(lp):
            return {"mode": "resume_latest", "substage": sub, "model_path": lp}

    last_finished = None
    for sub in SUBSTAGES:
        fp = final_model_path(sub)
        if os.path.exists(fp):
            last_finished = sub
        else:
            break

    if last_finished is None:
        return {"mode": "fresh_start", "substage": "2A", "model_path": None}

    if last_finished == SUBSTAGES[-1]:
        return {
            "mode": "done",
            "substage": last_finished,
            "model_path": final_model_path(last_finished),
        }

    next_sub = SUBSTAGES[SUBSTAGES.index(last_finished) + 1]
    return {
        "mode": "continue_next",
        "substage": next_sub,
        "model_path": final_model_path(last_finished),
        "prev_substage": last_finished,
    }


def train_one_substage(substage: str, model: Optional[PPO] = None) -> Tuple[PPO, bool]:
    total_timesteps = TOTAL_TIMESTEPS_PER_SUBSTAGE[substage]

    print("=" * 80)
    print(f"TRAIN STAGE 2 - GRASP MASTERY - SUBSTAGE {substage}")
    print("=" * 80)
    print(f"TOTAL_TIMESTEPS : {total_timesteps}")
    print(f"N_ENVS          : {N_ENVS}")
    print(f"TENSORBOARD_DIR : {TENSORBOARD_DIR}")
    print(f"USE_SUBPROC     : {USE_SUBPROC}")
    print(f"USE_GUI         : {USE_GUI}")
    print("=" * 80)

    env = build_vec_env(
        substage=substage,
        n_envs=N_ENVS,
        use_subproc=USE_SUBPROC,
        use_gui=USE_GUI,
    )
    callbacks = build_callbacks(substage)

    if model is None:
        model = build_new_model(env)
    else:
        model.set_env(env)

    finished_normally = False
    latest_save_base = os.path.join(
        MODEL_DIR,
        f"stage2_grasp_mastery_{substage_to_tag(substage)}_latest",
    )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            tb_log_name=f"stage2_grasp_mastery_{substage_to_tag(substage)}",
            progress_bar=True,
        )
        finished_normally = True

        lp = latest_model_path(substage)
        if os.path.exists(lp):
            try:
                os.remove(lp)
                print(f"[CLEAN] Removed stale latest file: {lp}")
            except Exception as e:
                print(f"[WARN] Could not remove latest file {lp}: {e}")

    except (KeyboardInterrupt, InterruptedError, EOFError, BrokenPipeError) as e:
        print(f"\n[INTERRUPT] Substage {substage} interrupted: type={type(e).__name__} message={e}")
        try:
            model.save(latest_save_base)
            print(f"[OK] Saved latest model: {latest_save_base}.zip")
        except Exception as e2:
            print(f"[ERROR] Failed to save latest model: {e2}")

    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[WARN] env.close() raised during shutdown: {e}")
        print("[TRAIN] Env closed.")

    return model, finished_normally


def main():
    state = resolve_training_state()
    print(f"[STATE] {state['mode']}")

    if state["mode"] == "done":
        print("[DONE] Stage 2 (2A -> 2D) đã hoàn tất.")
        print(f"[FINAL MODEL] {state['model_path']}")
        return

    if state["mode"] == "fresh_start":
        start_sub = state["substage"]
        model = None

    elif state["mode"] == "resume_latest":
        start_sub = state["substage"]
        env = build_vec_env(start_sub, N_ENVS, USE_SUBPROC, USE_GUI)
        model = load_model(state["model_path"], env)
        env.close()

    else:
        start_sub = state["substage"]
        print(f"[INFO] Previous substage done: {state['prev_substage']}")
        print(f"[INFO] Continue with: {start_sub}")
        env = build_vec_env(start_sub, N_ENVS, USE_SUBPROC, USE_GUI)
        model = load_model(state["model_path"], env)
        env.close()

    for sub in SUBSTAGES[SUBSTAGES.index(start_sub):]:
        model, finished = train_one_substage(sub, model=model)
        if not finished:
            print(f"[STOP] Training stopped during substage {sub}.")
            return

    print("[DONE] Finished full Stage 2 pipeline: 2A -> 2B -> 2C -> 2D")


if __name__ == "__main__":
    main()