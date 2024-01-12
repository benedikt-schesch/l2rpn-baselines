# -*- coding: utf-8 -*-
import os
from datetime import datetime
import torch
import numpy as np
import imageio
import wandb
from tqdm import tqdm
import json
from pathlib import Path
from PPO import PPO
from environments.Grid2OpResdispatchCurtail import Grid2OpEnvRedispatchCurtail
from environments.Grid2OpResdispatchCurtailFlattened import (
    Grid2OpEnvRedispatchCurtailFlattened,
)
from environments.GeneratorTargetEnv import GeneratorTargetEnv
from environments.Grid2OpRedispatchStorage import Grid2OpRedispatchStorage
from environments.Grid2OpBilevelFlattened import Grid2OpBilevelFlattened
from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)


def load_config(config_path: Path):
    with open(config_path, "r") as file:
        return json.load(file)


def load_env(config):
    env_type = config["env_type"]
    if env_type == "Grid2OpEnvRedispatchCurtail":
        return Grid2OpEnvRedispatchCurtail(env_name=config["env_name"])
    elif env_type == "Grid2OpEnvRedispatchCurtailFlattened":
        return Grid2OpEnvRedispatchCurtailFlattened(env_name=config["env_name"])
    elif env_type == "Grid2OpBilevelFlattened":
        return Grid2OpBilevelFlattened(env_name=config["env_name"])
    elif env_type == "GeneratorTargetEnv":
        return GeneratorTargetEnv(env_name=config["env_name"])
    elif env_type == "Grid2OpRedispatchStorage":
        return Grid2OpRedispatchStorage(env_name=config["env_name"])
    else:
        raise ValueError(f"Unknown env type: {env_type}")


def train(config_path=Path("configs/config_target_env.json")):
    print(
        "============================================================================================"
    )
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    config = load_config(config_path)
    wandb.init(
        project=config["wandb_project"],
        notes=current_time,
    )

    ####### initialize environment hyperparameters from config file ######
    env_name = config["env_name"]
    max_training_episodes = config["max_training_episodes"]
    save_model_freq = config["save_model_freq"]
    K_epochs = config["K_epochs"]
    eps_clip = config["eps_clip"]
    gamma = config["gamma"]
    entropy_max_loss = config.get("entropy_max_loss")
    entropy_min_loss = config.get("entropy_min_loss")
    lr_actor = config["lr_actor"]
    lr_critic = config["lr_critic"]
    random_seed = config.get("random_seed", None)  # Default to None if not provided

    ################# logging variables #################
    wandb.config = config

    print("training environment name : " + env_name)

    env = load_env(config)

    ################### checkpointing ###################
    directory = "logs/PPO"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + "/" + config["env_type"]
    checkpoint_dir = directory + "/" + current_time
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print("save checkpoint path : " + checkpoint_dir)
    #####################################################

    ############# print all hyperparameters #############
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("max training timesteps : ", max_training_episodes)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print(
        "============================================================================================"
    )

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(
        env.observation_space,
        env.action_space,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
    )

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print(
        "============================================================================================"
    )

    time_step = 0
    n_episode = 0
    max_reward = -np.inf
    with Progress(
        BarColumn(), TimeElapsedColumn(), TimeRemainingColumn(), MofNCompleteColumn()
    ) as progress:
        task_episodes = progress.add_task(
            description="[magenta]Episodes...", total=max_training_episodes
        )
        # training loop
        while not progress.finished:
            task_steps = progress.add_task(
                description="[cyan]Training...",
                total=env.max_episode_length,
            )
            state, _ = env.reset()
            current_ep_reward = 0
            done = False
            episode_time_step = 0

            while not done:
                # select action with policy
                action = ppo_agent.select_action(state)
                state, reward, done, terminated, info = env.step(action)

                # saving reward and is_terminals
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                time_step += 1
                episode_time_step += 1
                progress.update(task_steps, completed=episode_time_step)
                current_ep_reward += reward
                # break; if the episode is over
                if terminated:
                    break

            if current_ep_reward > max_reward:
                # Save best model
                max_reward = current_ep_reward
                checkpoint_path = (
                    f"{checkpoint_dir}/PPO_{env_name}_{random_seed}_best.pth"
                )
                print("Saving best model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("Best model saved")

            # save model weights
            if n_episode % save_model_freq == 0:
                print(
                    "--------------------------------------------------------------------------------------------"
                )
                curr_dir = checkpoint_dir + "/time_step_" + str(time_step)
                if not os.path.exists(curr_dir):
                    os.makedirs(curr_dir)
                checkpoint_path = curr_dir + "/PPO_{}_{}_{}.pth".format(
                    env_name, random_seed, time_step
                )
                print("Saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                # draw_agent(env, ppo_agent, curr_dir)
                print("Model saved")
                print(
                    "Elapsed Time  : ",
                    datetime.now().replace(microsecond=0) - start_time,
                )
                print(
                    "--------------------------------------------------------------------------------------------"
                )

            n_episode += 1

            # Draw agent
            if (
                "draw_agent_freq" in config
                and n_episode % config["draw_agent_freq"] == 0
            ):
                curr_dir = Path(checkpoint_dir) / ("time_step_" + str(time_step))
                curr_dir.mkdir(parents=True, exist_ok=True)
                draw_agent(env, ppo_agent, curr_dir)

            print(
                "Episode : {} \t\t Timestep : {} \t\t Epsiode Reward : {} Epsiode length: {} Entropy Loss weight: {}".format(
                    n_episode,
                    time_step,
                    current_ep_reward,
                    episode_time_step,
                    ppo_agent.entropy_loss_weight,
                )
            )
            wandb.log(
                {
                    "reward": current_ep_reward,
                    "timestep": time_step,
                    "episode length": episode_time_step,
                }
            )

            if n_episode % 16 == 15:
                ppo_agent.update(time_step)
                ppo_agent.entropy_loss_weight = max(
                    entropy_min_loss,
                    entropy_max_loss
                    - (entropy_max_loss - entropy_min_loss)
                    * time_step
                    / max_training_episodes,
                )
            progress.update(task_episodes, advance=1)

            progress.remove_task(task_steps)

    env.close()

    # print total training time
    print(
        "============================================================================================"
    )
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print(
        "============================================================================================"
    )

    agent_final_checkpoint = f"PPO_{env_name}_{random_seed}_{time_step}.pth"
    ppo_agent.save(os.path.join(wandb.run.dir, agent_final_checkpoint))  # type: ignore
    wandb.finish()

    print("Training Done!")


def draw_agent(env, ppo_agent: PPO, output_path: Path, frame_rate: int = 2):
    print("Start drawing agent")
    obs, _ = env.reset(seed=1)
    with imageio.get_writer(output_path / "movie.gif", mode="I") as writer:
        rewards = []
        done = False
        for i in tqdm(range(env.max_episode_length)):
            action = ppo_agent.select_action_eval(obs)
            obs, reward, done, terminated, _ = env.step(action)
            rewards.append(reward)

            if i % frame_rate == 0:
                frame = env.render(mode="rgb_array")
                writer.append_data(frame)

            done = done or terminated

            if done:
                break


if __name__ == "__main__":
    train()
