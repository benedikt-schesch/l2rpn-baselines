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
from typing import Union, Tuple, List
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    loss_fn,  # Loss function
    save_dir: Path,
    plot_title: str,
):
    num_items = len(value_extractor(agent_result[0]))
    fig, axs = plt.subplots(num_items, 1, figsize=(10, 5 * num_items))
    fig.suptitle(plot_title)

    agent_actions = [value_extractor(info) for info in agent_result]
    expert_actions = [value_extractor(info) for info in expert_result]

    # Initialize list to store losses
    losses = []

    for idx in range(num_items):
        axs[idx].plot([value[idx] for value in agent_actions], label="Agent")
        axs[idx].plot([value[idx] for value in expert_actions], label="Expert")
        axs[idx].set_title(f"Dimension {idx}")
        axs[idx].legend()
        axs[idx].set_xlabel("Timestep")
        axs[idx].set_ylabel("Value")

    # Calculate loss for this dimension at each timestep and add to losses list
    for agent_action, expert_action in zip(agent_result, expert_result):
        loss = loss_fn(
            torch.tensor(agent_action["action"]), torch.tensor(expert_action["action"])
        )
        losses.append(loss.item())

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(
        save_dir, f"{plot_title.lower().replace(' ', '_')}_comparison.png"
    )
    plt.savefig(fig_path)
    wandb.log({plot_title: wandb.Image(fig_path)})

    # Compute and print the average loss
    avg_loss = sum(losses) / len(losses) if losses else 0
    print(f"Average loss for {plot_title}: {avg_loss:.4f}")


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
            nn.Linear(obs_space.shape[0], 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_space.shape[0]),
            nn.Tanh(),
        )
        self.action_space = action_space
        self.th_high_action = nn.Parameter(
            torch.tensor(self.action_space.high, dtype=torch.float32),
            requires_grad=False,
        )
        self.th_low_action = nn.Parameter(
            torch.tensor(self.action_space.low, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, x):
        res = self.fc(x)
        # Rescale the output to match the action space
        return (
            res * (self.th_high_action - self.th_low_action) / 2.0
            + (self.th_high_action + self.th_low_action) / 2.0
        )


def train_bc_model(demonstrations, policy_net, optimizer, loss_fn, n_epochs):
    # Log hyperparameters (if any)
    wandb.config.update(
        {
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epochs": n_epochs,
            # add other hyperparameters here
        }
    )
    policy_net.to(device)
    for epoch in tqdm(range(n_epochs)):
        total_loss = 0
        for obs, acts in DataLoader(demonstrations, batch_size=32, shuffle=True):
            obs, acts = obs.to(device), acts.to(device)
            optimizer.zero_grad()
            predicted_actions = policy_net(obs)
            loss = loss_fn(predicted_actions, acts)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Calculate and log average loss for the epoch
        avg_loss = total_loss / len(demonstrations)
        wandb.log({"epoch": epoch, "loss": avg_loss})

    # Save and log the final model
    model_path = Path(wandb.run.dir) / "policy.pt"
    torch.save(policy_net.state_dict(), model_path)
    wandb.save(str(model_path))


@torch.no_grad()
def collect_and_label_with_expert(
    policy_net: ActorCriticNetwork,
    expert: ExpertAgent,
    env: Grid2OpRedispatchStorage,
) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
    """
    Collects data from the policy and labels it using the expert's actions.

    Parameters:
    - policy_net: The current policy network being trained.
    - expert: The expert agent used for querying the correct actions.
    - env: The environment to interact with.
    - num_episodes: Number of episodes to run for data collection.

    Returns:
    - An aggregated dataset of observations and expert-labeled actions.
    """
    aggregated_observations = []
    aggregated_actions = []
    policy_net.to("cpu")

    for episode in env.episode_ids:
        obs, info = env.reset(episode)
        if isinstance(expert, ExpertAgent):
            expert.reset(info)
        done = False

        while not done:
            obs = torch.FloatTensor(obs)
            # Use the policy network to decide on an action
            policy_action = policy_net(obs)
            # Query the expert for the correct action based on the current observation
            expert_action = expert.predict(obs, info)

            # Store the observation and expert's action
            aggregated_observations.append(obs)
            aggregated_actions.append(expert_action)

            # Step the environment
            obs, reward, done, _, info = env.step(policy_action.numpy())

    # Convert lists to appropriate tensor or array formats as required for training
    return aggregated_observations, aggregated_actions


def aggregate_datasets(
    existing_dataset: TensorDataset,
    new_observations: List[torch.Tensor],
    new_actions: List[np.ndarray],
) -> TensorDataset:
    """
    Aggregates the new observations and actions with the existing dataset.
    """
    existing_observations, existing_actions = existing_dataset.tensors
    new_observations_tensor = torch.stack(new_observations)
    new_actions_tensor = torch.tensor(np.stack(new_actions))

    aggregated_observations = torch.cat(
        [existing_observations, new_observations_tensor], dim=0
    )
    aggregated_actions = torch.cat([existing_actions, new_actions_tensor], dim=0)

    return TensorDataset(aggregated_observations, aggregated_actions)


def main(config):
    wandb.init(
        project="grid2op-imitation-dagger",
        config=config,
        tags=config["config_path"].split("/"),
    )
    env = Grid2OpRedispatchStorage(
        episode_ids=config["idx_of_train"], features=config["features"]
    )
    expert = ExpertAgent(env)

    # Start with expert demonstrations
    demonstrations, expert_infos = sample_expert_transitions(expert, env)
    loss_fn = nn.MSELoss()

    for i in range(config["dagger_iterations"]):
        print(f"DAgger Iteration {i+1}/{config['dagger_iterations']}")

        # Train policy on current dataset
        policy_net = ActorCriticNetwork(env.observation_space, env.action_space)
        optimizer = optim.Adam(policy_net.parameters(), lr=config["lr"])

        train_bc_model(
            demonstrations, policy_net, optimizer, loss_fn, config["n_epochs"]
        )

        new_observations, new_actions = collect_and_label_with_expert(
            policy_net, expert, env
        )

        # Aggregate datasets
        demonstrations = aggregate_datasets(
            demonstrations, new_observations, new_actions
        )

        policy_net.eval()
        policy_net.to(torch.device("cpu"))

        # Save the model
        model_path = Path(wandb.run.dir) / f"policy_{i+1}.pt"
        torch.save(policy_net, model_path)

        print("Evaluating the trained policy.")
        img_folder = Path(wandb.run.dir) / f"images_{i+1}"
        img_folder.mkdir(exist_ok=True)
        infos = {}
        for episode_id in env.episode_ids:
            infos[episode_id], episode_length = play_episode(
                policy_net, env, episode_id
            )
            print(
                f"Training episode {episode_id} finished after {episode_length} timesteps"
            )
            assert len(infos[episode_id]) == episode_length + 1
            plot_agent_actions(
                infos[episode_id],
                expert_infos[episode_id],
                lambda info: info["grid2op_action"].storage_p,
                loss_fn,
                img_folder,
                f"Storage Actions Episode {episode_id}",
            )
            plot_agent_actions(
                infos[episode_id],
                expert_infos[episode_id],
                lambda info: info["grid2op_action"].redispatch,
                loss_fn,
                img_folder,
                f"Redispatch Actions Episode {episode_id}",
            )
            # Plot the gen_p
            plot_agent_actions(
                infos[episode_id],
                expert_infos[episode_id],
                lambda info: info["grid2op_obs"].gen_p,
                loss_fn,
                img_folder,
                f"Generation Power Episode {episode_id}",
            )
            # Plot the storage
            plot_agent_actions(
                infos[episode_id],
                expert_infos[episode_id],
                lambda info: info["grid2op_obs"].storage_power,
                loss_fn,
                img_folder,
                f"Storage power Episode {episode_id}",
            )

    # Clean up
    env.close()
    print("Done logged to", str(wandb.run.dir))
    wandb.finish()


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
            action = agent(torch.FloatTensor(obs)).detach().numpy()
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
