# scripts/train_stage1_reach_sb3.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys  
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
torch.set_num_threads(1)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

from env.reach_env import ReachEnv
from config.env_config import EnvConfig
from script.callbacks import ReachTensorboardCallback


def make_env(substage="1A"):
    def _thunk():
        cfg = EnvConfig(stage1_substage=substage)
        env = ReachEnv(use_gui=False, config=cfg)
        return Monitor(env)
    return _thunk


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs_stage1", exist_ok=True)

    # train 1A trước cho ổn định
    env = SubprocVecEnv([make_env("1A") for _ in range(8)])
    env = VecTransposeImage(env)  # (H,W,C) -> (C,H,W)

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
    callback = ReachTensorboardCallback()

    model.learn(total_timesteps=3_000_000, callback=callback)
    model.save("models/stage1_reach_1A")
    env.close()

    # (tuỳ chọn) train tiếp 1B để generalize
    env = SubprocVecEnv([make_env("1B") for _ in range(8)])
    env = VecTransposeImage(env)

    callback = ReachTensorboardCallback()

    model.set_env(env)
    model.learn(total_timesteps=3_000_000, callback=callback)
    model.save("models/stage1_reach_1B")
    env.close()