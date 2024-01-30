# -*- coding: utf-8 -*-
"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
import numpy as np
from environments.Grid2OpRedispatchStorage import Grid2OpRedispatchStorage
from agents.OptimCVXPY import OptimCVXPY
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
import pickle
import os
import wandb
import json
from typing import Union
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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
            episode_info, _ = play_episode(expert, env, episode_id)
            episode_infos[episode_id] = episode_info
            print(f"Episode {episode_id} finished after {len(episode_info)} timesteps")

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # Serialize and store the episode_infos
        with open(cache_file, "wb") as f:
            pickle.dump(episode_infos, f)
            print(f"Episode infos stored in cache: {cache_file}")

    observations = [
        info["obs"] for episode in episode_infos.values() for info in episode
    ]
    actions = [info["action"] for episode in episode_infos.values() for info in episode]
    obs_tensor = torch.FloatTensor(observations)
    acts_tensor = torch.FloatTensor(actions)
    return TensorDataset(obs_tensor, acts_tensor), episode_infos


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_space, action_space):
        super(ActorCriticNetwork, self).__init__()
        # Define network architecture here
        self.fc = nn.Sequential(
            nn.Linear(obs_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.shape[0]),
        )

    def forward(self, x):
        return self.fc(x)


def train_bc_model(demonstrations, policy_net, optimizer, loss_fn, n_epochs):
    for epoch in tqdm(range(n_epochs)):
        for obs, acts in DataLoader(demonstrations, batch_size=32, shuffle=True):
            optimizer.zero_grad()
            predicted_actions = policy_net(obs)
            loss = loss_fn(predicted_actions, acts)
            loss.backward()
            optimizer.step()


def main(config):
    wandb.init(
        project="grid2op-imitation",
        config=config,
        tags=config["config_path"].split("/"),
    )
    log_folder = Path(wandb.run.dir) / "logs"
    print("Logging to", log_folder)

    env = Grid2OpRedispatchStorage(
        episode_ids=config["idx_of_train"], features=config["features"]
    )
    expert = ExpertAgent(env)

    demonstrations, expert_infos = sample_expert_transitions(expert, env)

    evaluation_env = Grid2OpRedispatchStorage(
        episode_ids=config["idx_of_eval"], features=config["features"]
    )

    # Initialize policy network, optimizer, and loss function
    policy_net = ActorCriticNetwork(env.observation_space, env.action_space)
    optimizer = optim.Adam(policy_net.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()  # or nn.CrossEntropyLoss() for discrete actions

    train_bc_model(demonstrations, policy_net, optimizer, loss_fn, config["n_epochs"])

    # print("Evaluating the untrained policy.")
    # rewards, episode_lengths = evaluate_policy(
    #     bc_trainer.policy,  # type: ignore[arg-type]
    #     evaluation_env,
    #     n_eval_episodes=3,
    #     return_episode_rewards=True,
    # )
    # print(f"Testing Episode lengths before training: {episode_lengths}")

    print("Training a policy using Behavior Cloning")

    print("Evaluating the trained policy.")
    infos = {}
    for episode_id in env.episode_ids:
        infos[episode_id], episode_length = play_episode(policy_net, env, episode_id)
        print(
            f"Training episode {episode_id} finished after {episode_length} timesteps"
        )
        assert len(infos[episode_id]) == episode_length + 1
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
    torch.save(policy_net, log_folder / "policy.pt")
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
    agent: Union[ActorCriticNetwork, ExpertAgent],
    env: Grid2OpRedispatchStorage,
    episode_id: int,
):
    obs, info = env.reset(episode_id)
    if isinstance(agent, ExpertAgent):
        agent.reset(info)
    done = False
    infos = []
    i = 0
    for i in tqdm(range(env.max_episode_length)):
        if isinstance(agent, ActorCriticNetwork):
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
    return infos, i


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
