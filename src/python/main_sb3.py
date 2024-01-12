# -*- coding: utf-8 -*-
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from main import load_config, load_env
from pathlib import Path
from datetime import datetime
import argparse


def main(config_path: Path = Path("configs/config_target_env.json")):
    config = load_config(config_path)
    # Initialize wandb
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run = wandb.init(
        project=config["wandb_project"],
        notes=current_time,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    # Create environment
    env = load_env(config)
    env = Monitor(env)

    log_path = "logs/PPO_sb3/" + current_time

    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(
    #     mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    # )

    # Define the models
    model = PPO(
        "MlpPolicy",
        env,
        ent_coef=config["entropy_max_loss"],
        verbose=1,
        tensorboard_log=log_path,
    )

    # Define callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config["save_model_freq"] * env.max_episode_length,
        save_path=log_path,
        name_prefix="model",
    )
    eval_env = Monitor(load_env(config))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_path,
        log_path=log_path,
        eval_freq=config["eval_model_freq"] * env.max_episode_length,
        n_eval_episodes=1,
        deterministic=True,
        verbose=1,
        render=False,
    )
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )
    callback_list = CallbackList([checkpoint_callback, eval_callback, wandb_callback])
    # Train the model
    timesteps = config["max_training_episodes"] * env.max_episode_length
    model.learn(total_timesteps=timesteps, callback=callback_list)

    # Save the model
    model.save(log_path + "/model_final.zip")

    # Close the environment
    env.close()

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--config", type=Path, default="configs/config_direct_redispatch_storage.json"
    )
    args = argparse.parse_args()
    main(args.config)
