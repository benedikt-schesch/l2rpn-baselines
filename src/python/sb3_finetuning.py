# -*- coding: utf-8 -*-
"""
This script demonstrates how to fine-tune a trained model using PPO.
"""

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from environments.Grid2OpRedispatchStorage import Grid2OpRedispatchStorage
import wandb
from wandb.integration.sb3 import WandbCallback
import json
from argparse import ArgumentParser


def fine_tune_model(model_path, env):
    run = wandb.init(
        project="sb3-imitation-finetune",
        sync_tensorboard=True,
        save_code=True,
    )
    policy = torch.load(model_path)
    net_kwargs = {"net_arch": [256, 64]}
    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=net_kwargs,
        gamma=1.0,
        learning_rate=0.0001,
        device="cpu",
        verbose=2,
        tensorboard_log=f"runs/{run.id}",
    )
    model.policy = policy

    callback = WandbCallback(verbose=2)
    model.learn(
        total_timesteps=1000000,
        reset_num_timesteps=False,
        callback=callback,
        progress_bar=True,
    )

    # model.save(fine_tune_config["save_path"])


def main(config_file, model_path):
    config = read_config(config_file)

    env = make_vec_env(
        lambda: Monitor(
            Grid2OpRedispatchStorage(
                episode_ids=config["idx_of_train"], features=config["features"]
            )
        ),
        n_envs=1,
    )

    fine_tune_model(model_path, env)

    # Clean up
    env.close()
    wandb.finish()


def read_config(config_file):
    with open(config_file, "r") as file:
        config = json.load(file)
    return config


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tuning a Trained Model with PPO")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/imitation/multiple_episodes/config_imitation_all_no_idx.json",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="wandb/run-20240123_232919-cwqsysms/files/policy.pt",
    )
    args = parser.parse_args()
    main(args.config, args.model)
