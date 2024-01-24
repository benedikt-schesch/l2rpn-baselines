# -*- coding: utf-8 -*-
"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
from calendar import c
import numpy as np
from imitation.algorithms import bc
from imitation.data import types
from environments.Grid2OpRedispatchStorage import Grid2OpRedispatchStorage
from imitation.util import logger as sb_logger
from stable_baselines3.common.monitor import Monitor
from agents.OptimCVXPY import OptimCVXPY
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
import pickle
import os
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import wandb
import json
from typing import Union
import matplotlib.pyplot as plt


class ExpertAgent:
    def __init__(self, env: Grid2OpRedispatchStorage):
        self.expert = OptimCVXPY(
            env.get_grid2op_env().action_space,
            env.get_grid2op_env(),
            penalty_redispatching_unsafe=0.0,
            penalty_storage_unsafe=0.04,
            penalty_curtailment_unsafe=0.01,
            rho_safe=0.95,
            rho_danger=0.97,
            margin_th_limit=0.93,
            alpha_por_error=0.5,
            weight_redisp_target=0.3,
        )
        self.expert.storage_setpoint = env.get_grid2op_env().storage_Emax  # type: ignore

    def predict(self, obs, info):
        grid2op_action = self.expert.act(info["grid2op_obs"])
        action = np.concatenate([grid2op_action.storage_p, grid2op_action.redispatch])
        return action

    def reset(self, info):
        self.expert.reset(info["grid2op_obs"])


def plot_agent_actions(
    agent_result,
    expert_result,
    value_extractor,
    save_dir: Path,
    plot_title: str,
):
    num_items = len(value_extractor(agent_result[0]))
    fig, axs = plt.subplots(num_items, 1, figsize=(10, 5 * num_items))
    fig.suptitle(plot_title)

    agent_actions = [value_extractor(info) for info in agent_result]
    expert_actions = [value_extractor(info) for info in expert_result]

    for idx in range(num_items):
        axs[idx].plot([value[idx] for value in agent_actions], label="Agent")
        axs[idx].plot([value[idx] for value in expert_actions], label="Expert")
        axs[idx].set_title(f"Dimension {idx}")
        axs[idx].legend()
        axs[idx].set_xlabel("Timestep")
        axs[idx].set_ylabel("Value")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(
        save_dir, f"{plot_title.lower().replace(' ', '_')}_comparison.png"
    )
    plt.savefig(fig_path)
    wandb.log({plot_title: wandb.Image(fig_path)})


def sample_expert_transitions(
    expert: ExpertAgent, env: Grid2OpRedispatchStorage, cache_folder=Path("cache")
):
    # Check if cache exists
    cache_file = cache_folder / (
        "_".join(env.features + [str(i) for i in env.episode_ids]) + ".pkl"
    )
    if os.path.exists(cache_file):
        print("Loading transitions from cache")
        with open(cache_file, "rb") as f:
            episode_infos = pickle.load(f)
    else:
        episode_infos = {}
        for episode_id in tqdm(env.episode_ids):
            episode_info = play_episode(expert, env, episode_id)
            episode_infos[episode_id] = episode_info
            print(f"Episode {episode_id} finished after {len(episode_info)} timesteps")

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # Serialize and store the episode_infos
        with open(cache_file, "wb") as f:
            pickle.dump(episode_infos, f)
            print(f"Episode infos stored in cache: {cache_file}")

    # Construct transitions from episode_infos
    keys = ["obs", "next_obs", "acts", "dones", "infos"]
    trajectories = {k: [] for k in keys}
    for episode_info in episode_infos.values():
        for i, info in enumerate(episode_info):
            if i + 1 >= len(episode_info):
                continue
            obs = info["obs"]
            next_obs = episode_info[i + 1]["obs"]
            trajectories["obs"].append(obs)
            trajectories["next_obs"].append(next_obs)
            trajectories["acts"].append(info["action"])
            trajectories["dones"].append(info["done"])
            trajectories["infos"].append({})

    for k in keys:
        trajectories[k] = np.array(trajectories[k])

    result = types.Transitions(**trajectories)
    return result, episode_infos


def main(config):
    wandb.init(project="grid2op-imitation", config=config, tags=[config["config_path"]])
    log_folder = Path(wandb.run.dir) / "logs"
    print("Logging to", log_folder)

    env = Grid2OpRedispatchStorage(
        episode_ids=config["idx_of_train"], features=config["features"]
    )
    expert = ExpertAgent(env)
    policy = ActorCriticPolicy(
        env.observation_space,
        env.action_space,
        net_arch=[256, 64],
        lr_schedule=lambda _: config["lr"],
    )

    transitions, expert_infos = sample_expert_transitions(expert, env)
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=policy,
        demonstrations=transitions,
        ent_weight=config["ent_weight"],
        l2_weight=config["l2_weight"],
        rng=np.random.default_rng(0),
        optimizer_kwargs=dict(lr=config["lr"]),
        device="cpu",
        custom_logger=sb_logger.configure(
            folder=log_folder,
            format_strs=["stdout", "csv", "wandb"],
        ),
    )

    evaluation_env = Monitor(
        Grid2OpRedispatchStorage(
            episode_ids=config["idx_of_eval"], features=config["features"]
        )
    )

    # print("Evaluating the untrained policy.")
    # rewards, episode_lengths = evaluate_policy(
    #     bc_trainer.policy,  # type: ignore[arg-type]
    #     evaluation_env,
    #     n_eval_episodes=3,
    #     return_episode_rewards=True,
    # )
    # print(f"Testing Episode lengths before training: {episode_lengths}")

    print("Training a policy using Behavior Cloning")
    bc_trainer.train(n_epochs=config["n_epochs"])

    print("Evaluating the trained policy.")
    infos = {}
    for episode_id in env.episode_ids:
        infos[episode_id] = play_episode(bc_trainer.policy, env, episode_id)
        print(
            f"Training episode {episode_id} finished after {len(infos[episode_id])} timesteps"
        )
        plot_agent_actions(
            infos[episode_id],
            expert_infos[episode_id],
            lambda info: info["grid2op_action"].storage_p,
            log_folder,
            f"Storage Actions Episode {episode_id}",
        )
        plot_agent_actions(
            infos[episode_id],
            expert_infos[episode_id],
            lambda info: info["grid2op_action"].redispatch,
            log_folder,
            f"Redispatch Actions Episode {episode_id}",
        )
        # Plot the gen_p
        plot_agent_actions(
            infos[episode_id],
            expert_infos[episode_id],
            lambda info: info["grid2op_obs"].gen_p,
            log_folder,
            f"Generation Power Episode {episode_id}",
        )
        # Plot the storage
        plot_agent_actions(
            infos[episode_id],
            expert_infos[episode_id],
            lambda info: info["grid2op_obs"].storage_power,
            log_folder,
            f"Storage power Episode {episode_id}",
        )

    # rewards, episode_lengths = evaluate_policy(
    #     bc_trainer.policy,  # type: ignore[arg-type]
    #     evaluation_env,
    #     n_eval_episodes=10,
    #     return_episode_rewards=True,
    #     deterministic=True,
    # )
    # print(f"Testing Episode lengths after training: {episode_lengths}")

    # Save the policy
    torch.save(bc_trainer.policy, log_folder / "policy.pt")
    wandb.save(str(log_folder / "policy.pt"))
    # Clean up
    env.close()
    evaluation_env.close()
    wandb.finish()
    print("Done logged to", log_folder)


def read_config(config_file):
    with open(config_file, "r") as file:
        config = json.load(file)
    return config


def play_episode(
    agent: Union[ActorCriticPolicy, ExpertAgent],
    env: Grid2OpRedispatchStorage,
    episode_id: int,
):
    obs, info = env.reset(episode_id)
    if isinstance(agent, ExpertAgent):
        agent.reset(info)
    done = False
    infos = []
    for i in tqdm(range(env.max_episode_length)):
        if isinstance(agent, ActorCriticPolicy):
            action = agent.predict(obs, deterministic=True)[0]
        else:
            action = agent.predict(obs, info)
        info = {}
        info["obs"] = obs
        info["action"] = action
        obs, reward, done, _, step_info = env.step(action)
        info.update(step_info)
        info["reward"] = reward
        info["done"] = done
        infos.append(info)
        if done:
            break
    return infos


if __name__ == "__main__":
    parser = ArgumentParser(description="Behavior Cloning with Grid2Op")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/imitation/multiple_episodes/config_imitation_all_episode_timestep_idx.json",
    )
    args = parser.parse_args()
    config = read_config(args.config)
    config["config_path"] = args.config
    main(config)
