# scripts/train_stage3_place_sb3.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import multiprocessing as mp
import torch
torch.set_num_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

from env.place_env import PlaceEnv
from script.callbacks import PlaceTensorboardCallback


def make_env(substage="3A"):
    def _thunk():
        env = PlaceEnv(use_gui=False, start_held=True, substage=substage)
        return Monitor(env)
    return _thunk


if __name__ == "__main__":
    mp.freeze_support()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs_stage3", exist_ok=True)

    # --- train 3A ---
    env = DummyVecEnv([make_env("3A")])
    env = VecTransposeImage(env)

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
        tensorboard_log="logs_stage3",
    )

    callback = PlaceTensorboardCallback()
    model.learn(total_timesteps=1_000_000, callback=callback, tb_log_name="place_3A")
    model.save("models/stage3_place_3A")
    env.close()

    # --- train 3B ---
    env = DummyVecEnv([make_env("3B")])
    env = VecTransposeImage(env)

    callback = PlaceTensorboardCallback()
    model.set_env(env)
    model.learn(total_timesteps=3_000_000, callback=callback, tb_log_name="place_3B")
    model.save("models/stage3_place_3B")
    env.close()

    # --- train 3C ---
    env = DummyVecEnv([make_env("3C")])
    env = VecTransposeImage(env)

    callback = PlaceTensorboardCallback()
    model.set_env(env)
    model.learn(total_timesteps=5_000_000, callback=callback, tb_log_name="place_3C")
    model.save("models/stage_place_3C")
    env.close()