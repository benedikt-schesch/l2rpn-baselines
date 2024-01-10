# -*- coding: utf-8 -*-
from typing import Union, Tuple
import torch
from gymnasium import Env
from gymnasium import spaces
from collections import OrderedDict
from .OptimizerNoCurtailement import OptimCVXPY
from pathlib import Path
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from grid2op.Chronics import GridStateFromFileWithForecastsWithoutMaintenance
from grid2op.Action import DontAct, PlayableAction
from grid2op.Opponent import BaseOpponent, NeverAttackBudget
import grid2op
from grid2op.Reward import LinesCapacityReward
from lightsim2grid import LightSimBackend
import numpy as np
import matplotlib.pyplot as plt


class Grid2OpBilevelFlattened(Env):
    def __init__(self, env_name: str = "l2rpn_case14_sandbox") -> None:
        super().__init__()
        # Initialize the Grid2OpEnvRedispatchCurtailFlattened environment
        self.grid2op_env = grid2op.make(
            env_name,
            reward_class=LinesCapacityReward,
            backend=LightSimBackend(),
            data_feeding_kwargs={
                "gridvalueClass": GridStateFromFileWithForecastsWithoutMaintenance
            },
            opponent_attack_cooldown=999999,
            opponent_attack_duration=0,
            opponent_budget_per_ts=0,
            opponent_init_budget=0,
            opponent_action_class=DontAct,
            opponent_class=BaseOpponent,
            opponent_budget_class=NeverAttackBudget,
            action_class=PlayableAction,
            test=True,
        )
        obs = self.grid2op_env.reset()
        flat_features = self.flatten_features(obs)
        self.observation_space = spaces.Box(
            low=-float("inf"), high=float("inf"), shape=flat_features.shape
        )
        self.max_nb_bus = self.grid2op_env.n_sub * 2
        self.action_space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(self.max_nb_bus,),
        )
        self.delta = 0.01
        self.optimizer = OptimCVXPY(
            self.get_grid2op_env().action_space,
            self.get_grid2op_env(),
            lines_x_pu=None,
            margin_th_limit=0.9,
            alpha_por_error=0.5,
            rho_danger=0.0,
            margin_rounding=0.01,
            margin_sparse=5e-3,
            delta=self.delta,
        )

    def flatten_features(self, obs: BaseObservation) -> torch.Tensor:
        features = [obs.rho, obs.load_p, obs.load_q, obs.gen_p, obs.gen_q]
        features = torch.tensor(np.concatenate(features))
        return features

    def reset(self, seed: Union[None, int] = None) -> Tuple[torch.Tensor, dict]:
        self.grid2op_env.set_id(6)
        grid2op_obs = self.grid2op_env.reset()
        self.latest_obs = grid2op_obs
        self.done = False
        self.time_step = 0
        self.play_until_unsafe()
        obs = self.flatten_features(self.latest_obs)
        return obs, {}

    def play_until_unsafe(self):
        pass
        # while self.latest_obs.rho.max() < self.optimizer.rho_danger and not self.done:
        #     grid2op_act, optim_info = self.optimizer.act(self.latest_obs)
        #     self.latest_obs, reward, self.done, info = self.grid2op_env.step(
        #         grid2op_act
        #     )
        #     self.time_step += 1

    def step(
        self, action: Union[None, torch.Tensor]
    ) -> Tuple[torch.Tensor, float, bool, bool, dict]:
        self.optimizer.set_target_injected_power(action)
        grid2op_act, optimizer_info = self.optimizer.act(self.latest_obs)
        # print("Storage act:", grid2op_act.storage_p)
        # grid2op_act.storage_p = np.zeros(self.grid2op_env.n_storage)
        # grid2op_act.redispatch = np.ones(self.grid2op_env.n_gen)
        # grid2op_act.redispatch = np.clip(grid2op_act.redispatch, -self.grid2op_env.gen_max_ramp_down, self.grid2op_env.gen_max_ramp_up)
        self.latest_obs, reward, self.done, info = self.grid2op_env.step(grid2op_act)
        self.time_step += 1
        self.play_until_unsafe()
        obs = self.flatten_features(self.latest_obs)
        info.update(optimizer_info)
        info["grid2op_redispatch"] = grid2op_act.redispatch
        info["grid2op_storage_p"] = grid2op_act.storage_p
        info["action"] = action
        return obs, 3 + reward, self.done, False, info

    def render(self, mode="rgb_array"):
        return self.grid2op_env.render(mode)

    def get_time_step(self) -> int:
        return self.time_step

    def get_grid2op_env(self) -> Environment:
        return self.grid2op_env

    def get_grid2op_obs(self):
        return self.latest_obs


node_data_fields = OrderedDict(
    {
        "substation": [
            # "id"
        ],  # All fields: {'id': 1, 'type': 'substation', 'name': 'sub_1', 'cooldown': 0}
        "bus": [
            "v",
            "theta",
            # "id",
        ],  # All fields: {'id': 2, 'global_id': 2, 'local_id': 1, 'type': 'bus', 'connected': True, 'v': 142.1, 'theta': -4.190119}
        "load": [
            # "id"
        ],  # {'id': 8, 'type': 'load', 'name': 'load_11_8', 'connected': True}
        "gen": [
            # "id",
            "target_dispatch",
            "actual_dispath",
            "gen_p_before_curtal",
            # "curtalment_mw",
            # "curtailement",
            # "curtailment_limit",
            "gen_margin_up",
            "gen_margin_down",
            # "difference_dispatch",
            # "target_dispatch",
            # "actual_dispatch",
        ],  # {'id': 3, 'type': 'gen', 'name': 'gen_5_3', 'connected': True, 'target_dispatch': 0.0, 'actual_dispatch': 0.0, 'gen_p_before_curtail': 0.0, 'curtailment_mw': 0.0, 'curtailment': 0.0, 'curtailment_limit': 1.0, 'gen_margin_up': 0.0, 'gen_margin_down': 0.0}
        "line": [
            # "id",
            "rho",
        ],  # {'id': 1, 'type': 'line', 'name': '0_4_1', 'rho': 0.36042336, 'connected': True, 'timestep_overflow': 0, 'time_before_cooldown_line': 0, 'time_next_maintenance': -1, 'duration_next_maintenance': 0}
        "shunt": [
            # "id"
        ],  # {'id': 0, 'type': 'shunt', 'name': 'shunt_8_0', 'connected': True}
        "storage": [
            "storage_charge",
            "storage_power_target",
        ],
    }
)


def plot_cost(infos, delta, plot_dir):
    num_episodes = len(infos)
    fig, axs = plt.subplots(
        num_episodes, 1, figsize=(10, 5 * num_episodes), constrained_layout=True
    )

    for i, episode_infos in enumerate(infos):
        ax1 = axs[i] if num_episodes > 1 else axs

        # Prepare data for this episode
        deviation_values = [info["deviation_component"] for info in episode_infos]
        flow_limit_values = [
            info["flow_limit_violation_component"] for info in episode_infos
        ]

        # Plot the deviation component for this episode
        color = "tab:blue"
        ax1.set_ylabel("Deviation Component", color=color)
        ax1.plot(deviation_values, color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        # Create a second y-axis for the flow limit violation component
        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("Flow Limit Violation", color=color)
        ax2.plot(flow_limit_values, color=color, linestyle="dashed")
        ax2.tick_params(axis="y", labelcolor=color)

        ax1.set_xlabel("Iteration")
        ax1.set_title(f"Episode {i} - Cost Components (delta={delta})")

    fig.suptitle("Cost Components Across Episodes")
    plt.savefig(plot_dir / "cost_components_separate_episodes.pdf")
    plt.close(fig)  # Close the figure to free memory


def play_episode(env: Grid2OpBilevelFlattened, action_baseline: np.ndarray):
    obs, _ = env.reset()
    done = False
    cum_reward = 0
    infos = []
    for i in range(env.get_grid2op_env().chronics_handler.max_timestep()):
        action = (
            action_baseline[i]
            - (2 * np.random.randn(env.action_space.shape[0]) - 1) * 10
        )
        obs, reward, done, _, info = env.step(action)
        info["obs"] = env.get_grid2op_obs()
        infos.append(info)
        cum_reward += reward
        if done:
            break
    print("Cumulative reward", cum_reward)
    print("Done", i)
    return infos


def plot_action_difference(infos, plot_dir):
    # Number of episodes
    num_episodes = len(infos)

    # Create a figure with subplots
    fig, axs = plt.subplots(num_episodes, 1, figsize=(10, 5 * num_episodes))

    # Base actions for comparison (using the first episode as baseline)
    baseline_redispatch = np.array([info["grid2op_redispatch"] for info in infos[0]])
    baseline_storage_p = np.array([info["grid2op_storage_p"] for info in infos[0]])

    for i in range(num_episodes):
        current_redispatch = np.array([info["grid2op_redispatch"] for info in infos[i]])
        current_storage_p = np.array([info["grid2op_storage_p"] for info in infos[i]])

        # Calculate differences up to the length of the shorter episode
        min_length = min(len(baseline_redispatch), len(current_redispatch))
        redispatch_diff = np.linalg.norm(
            current_redispatch[:min_length] - baseline_redispatch[:min_length], axis=1
        )
        storage_diff = np.linalg.norm(
            current_storage_p[:min_length] - baseline_storage_p[:min_length], axis=1
        )

        # Plot on the i-th subplot
        ax = axs[i] if num_episodes > 1 else axs
        ax.plot(redispatch_diff, label="Redispatch Difference")
        ax.plot(storage_diff, label="Storage Difference")
        ax.set_title(f"Difference in Action for Run {i}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_dir / "action_differences_combined.pdf")
    plt.close(fig)  # Close the figure to free memory


# Plot the norm of the actions taken by the agent
def plot_actions(infos, plot_dir, labels):
    # Number of episodes
    num_infos = len(infos)

    # Create a figure with subplots
    fig, axs = plt.subplots(num_infos, 1, figsize=(10, 5 * num_infos))

    for info_idx in range(len(infos)):
        current_redispatch = np.array(
            [np.linalg.norm(info["grid2op_redispatch"]) for info in infos[info_idx]]
        )
        current_storage_p = np.array(
            [np.linalg.norm(info["grid2op_storage_p"]) for info in infos[info_idx]]
        )

        # Plot on the i-th subplot
        ax = axs[info_idx] if num_infos > 1 else axs
        ax.plot(current_redispatch, label="Redispatch Norm")
        ax.plot(current_storage_p, label="Storage Norm")
        ax.set_title(labels[info_idx])
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_dir / "actions_combined.pdf")
    plt.close(fig)  # Close the figure to free memory


def plot_generator_power(infos, plot_dir, labels):
    # Determine the number of generators from the first observation of the first episode
    num_generators = infos[0][0]["obs"].gen_p.shape[0]
    num_episodes = len(infos)

    # Create a subplot for each generator
    fig, axs = plt.subplots(num_generators, 1, figsize=(10, 5 * num_generators))

    for gen_index in range(num_generators):
        ax = axs[gen_index] if num_generators > 1 else axs

        for infos_idx in range(num_episodes):
            # Extract the power level for the current generator across all time steps in this episode
            gen_power_levels = np.array(
                [info["obs"].gen_p[gen_index] for info in infos[infos_idx]]
            )

            ax.plot(gen_power_levels, label=labels[infos_idx])

        ax.set_title(f"Generator {gen_index} Power Across Episodes")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Power Level")
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_dir / "generator_power_across_episodes.pdf")
    plt.close(fig)  # Close the figure to free memory


def plot_storage_charge(infos, plot_dir, labels):
    # Determine the number of storage units from the first observation of the first episode
    num_storage_units = infos[0][0]["obs"].storage_charge.shape[0]
    num_episodes = len(infos)

    # Create a subplot for each storage unit
    fig, axs = plt.subplots(num_storage_units, 1, figsize=(10, 5 * num_storage_units))

    for storage_index in range(num_storage_units):
        ax = axs[storage_index] if num_storage_units > 1 else axs

        for infos_idx in range(num_episodes):
            # Extract the storage charge for the current storage unit across all time steps in this episode
            storage_charge_levels = np.array(
                [info["obs"].storage_charge[storage_index] for info in infos[infos_idx]]
            )

            ax.plot(storage_charge_levels, label=labels[infos_idx])

        ax.set_title(f"Storage Unit {storage_index} Charge Across Episodes")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Storage Charge")
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_dir / "storage_charge_across_episodes.pdf")
    plt.close(fig)  # Close the figure to free memory


def plot_cost_difference(infos, delta, plot_dir):
    num_episodes = len(infos)

    # Choose the first episode as the baseline for comparison
    baseline_deviation = np.array([info["deviation_component"] for info in infos[0]])
    baseline_flow_limit = np.array(
        [info["flow_limit_violation_component"] for info in infos[0]]
    )

    # Create a figure for the cost differences
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)

    for i in range(1, num_episodes):  # Start from 1 since 0 is the baseline
        episode_infos = infos[i]

        # Extract the cost components for this episode
        episode_deviation = np.array(
            [info["deviation_component"] for info in episode_infos]
        )
        episode_flow_limit = np.array(
            [info["flow_limit_violation_component"] for info in episode_infos]
        )

        # Calculate the differences in cost components
        # Adjust to the length of the shorter of the two episodes
        min_length = min(len(baseline_deviation), len(episode_deviation))
        deviation_diff = (
            episode_deviation[:min_length] - baseline_deviation[:min_length]
        )
        flow_limit_diff = (
            episode_flow_limit[:min_length] - baseline_flow_limit[:min_length]
        )

        # Plot the differences
        ax1.plot(deviation_diff, label=f"Episode {i} Deviation Diff")
        ax2.plot(flow_limit_diff, label=f"Episode {i} Flow Limit Diff")

    ax1.set_title("Deviation Component Difference from Baseline")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Difference")
    ax1.legend()

    ax2.set_title("Flow Limit Violation Component Difference from Baseline")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Difference")
    ax2.legend()

    fig.suptitle(f"Cost Component Differences Across Episodes (delta={delta})")
    plt.savefig(plot_dir / "cost_component_differences.pdf")
    plt.close(fig)  # Close the figure to free memory


if __name__ == "__main__":
    plot_dir = Path("experiment")
    plot_dir.mkdir(exist_ok=True)
    np.random.seed(0)
    env = Grid2OpBilevelFlattened("educ_case14_storage")
    action_baseline = (2 * np.random.randn(1000, env.action_space.shape[0]) - 1) * 100
    infos = []
    for i in range(4):
        infos.append(play_episode(env, action_baseline))
    plot_action_difference(infos, plot_dir)
    plot_generator_power(infos, plot_dir)
    plot_storage_charge(infos, plot_dir)
    plot_cost(infos, env.delta, plot_dir)
    plot_cost_difference(infos, env.delta, plot_dir)
    # action_baseline = (2*np.random.randn(1000, env.action_space.shape[0])-1) * 10000
    # infos = []
    # for i in range(1):
    #     infos.append(play_episode(action_baseline))
