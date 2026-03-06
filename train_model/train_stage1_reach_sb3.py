# scripts/train_stage1_reach_sb3.py
import os
import sys

# --- reduce CPU thread contention (important on Windows) ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

# --- make project root importable ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from env.reach_env import ReachEnv
from config.env_config import EnvConfig
from script.callbacks import ReachTensorboardCallback  # giữ đúng path bạn đang dùng


def make_env(substage="1A"):
    def _thunk():
        cfg = EnvConfig(stage1_substage=substage)
        env = ReachEnv(use_gui=False, config=cfg)
        return Monitor(env)
    return _thunk


def train_one_substage(substage: str, model: Optional[PPO] = None):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs_stage1", exist_ok=True)
    os.makedirs(f"models/checkpoints_stage1_{substage}", exist_ok=True)

    env = SubprocVecEnv([make_env(substage) for _ in range(8)])
    env = VecTransposeImage(env)

    if model is None:
        model = PPO(
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
    else:
        model.set_env(env)

    # ✅ auto checkpoint every N steps
    # NOTE: save_freq is in "callback steps" (which equals 1 step of VecEnv = 8 env steps).
    # SB3 uses num_timesteps (already includes all envs), so this is safe.
    ckpt = CheckpointCallback(
        save_freq=200_000,  # bạn có thể đổi 100_000 hoặc 50_000 nếu muốn lưu dày hơn
        save_path=f"models/checkpoints_stage1_{substage}",
        name_prefix=f"ppo_stage1_{substage}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    tb = ReachTensorboardCallback()
    callbacks = CallbackList([tb, ckpt])

    final_path = f"models/stage1_reach_{substage}_latest"
    try:
        model.learn(total_timesteps=3_000_000, callback=callbacks)
        model.save(f"models/stage1_reach_{substage}")  # save final
        print(f"[OK] Saved final model: models/stage1_reach_{substage}.zip")
    except KeyboardInterrupt:
        # ✅ user stopped training -> save latest
        print("\n[INTERRUPT] Saving latest model before exit...")
        model.save(final_path)
        print(f"[OK] Saved latest model: {final_path}.zip")
    finally:
        env.close()

    return model


if __name__ == "__main__":
    # Train 1A
    model = train_one_substage("1A", model=None)

    # Train 1B continuing from 1A weights
    model = train_one_substage("1B", model=model)