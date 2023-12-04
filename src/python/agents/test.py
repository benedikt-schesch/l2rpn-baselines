# -*- coding: utf-8 -*-
from grid2op.Environment import Environment
import matplotlib.pyplot as plt
import grid2op
from grid2op.Agent import RecoPowerlineAgent, DoNothingAgent
from OptimCVXPY import OptimCVXPY
from lightsim2grid import LightSimBackend
from tqdm import tqdm
import numpy as np
import pandas as pd
from numpy.random import default_rng
from pathlib import Path

prng = default_rng(0)

env_name = "l2rpn_wcci_2022"
is_test = False

env = grid2op.make(env_name, test=is_test, backend=LightSimBackend())
logger = None
reco_agent = RecoPowerlineAgent(env.action_space)
do_nothing_agent = DoNothingAgent(env.action_space)

agent = OptimCVXPY(
    env.action_space,
    env,
    rho_safe=0,
    rho_danger=0.0,
    penalty_redispatching_unsafe=0.0,
    penalty_storage_unsafe=0.01,
    penalty_curtailment_unsafe=0.01,
    penalty_curtailment_safe=0.1,
    penalty_redispatching_safe=0.1,
    logger=logger,
)


scen_test = [
    "2050-01-03_31",
    "2050-02-21_31",
    "2050-03-07_31",
    "2050-04-18_31",
    "2050-05-09_31",
    "2050-06-27_31",
    "2050-07-25_31",
    "2050-08-01_31",
    "2050-09-26_31",
    "2050-10-03_31",
    "2050-11-14_31",
    "2050-12-19_31",
]

seeds = prng.integers(0, np.iinfo(np.int32).max, size=len(scen_test))
seeds = {scen: int(seed) for scen, seed in zip(scen_test, seeds)}


# Function to plot redispatching for all agents
def plot(value, name: str, directory: Path, scenario_name: str, agents):
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(value[0]), 1, figsize=(10, int(2.5 * len(value[0]))), tight_layout=True
    )

    for i in range(len(value[0])):
        for agent_idx, agent in enumerate(agents):
            axes[i].plot(value[agent_idx][i], label=f"{agent.__class__.__name__}")
        axes[i].set_title(f"{name} Generator {i+1} in {scenario_name}")
        axes[i].legend()
        axes[i].grid(True)

    plt.savefig(directory / f"{scenario_name}_{name}_plot.pdf")
    plt.close(fig)


# Modified main simulation loop with data collection for redispatching
def run_simulation(scen_test, env: Environment, agents):
    directory = Path("./plots")
    results_rewards = {agent.__class__.__name__: [] for agent in agents}
    results_timesteps = {agent.__class__.__name__: [] for agent in agents}

    for idx, scen_id in enumerate(scen_test):
        env.set_id(scen_id)
        env.seed(seeds[scen_id])

        all_power_levels = [[[] for _ in range(env.n_gen)] for _ in agents]
        all_redispatching = [
            [[] for _ in range(env.n_gen)] for _ in agents
        ]  # Data structure for redispatching
        all_curtailment = [
            [[] for _ in range(env.n_gen)] for _ in agents
        ]  # Data structure for curtailment
        all_storage = [
            [[] for _ in range(env.n_storage)] for _ in agents
        ]  # Data structure for storage

        for agent_idx, agent in enumerate(agents):
            print(f"Running scenario {scen_id} with agent {agent.__class__.__name__}")
            obs = env.reset()
            agent.reset(obs)
            done = False
            reward = env.reward_range[0]
            tot_reward = 0.0
            for nb_step in tqdm(range(obs.max_step)):
                act = agent.act(obs, reward)
                obs, reward, done, _ = env.step(act)
                tot_reward += reward

                if done:
                    break
                for i in range(env.n_gen):
                    all_power_levels[agent_idx][i].append(obs.prod_p[i])
                    all_redispatching[agent_idx][i].append(
                        act._redispatch[i]
                    )  # Collecting redispatching data
                    all_curtailment[agent_idx][i].append(act._curtail[i])
                for i in range(env.n_storage):
                    all_storage[agent_idx][i].append(act.storage_p[i])
            print(
                f"scenario: {scen_id} {agent.__class__.__name__}: {nb_step + 1} / {obs.max_step}"
            )
            results_rewards[agent.__class__.__name__].append(tot_reward)
            results_timesteps[agent.__class__.__name__].append(nb_step + 1)
        # Call the plotting functions after each scenario
        plot(all_power_levels, "power levels", directory, scen_id, agents)
        plot(all_redispatching, "redispatching", directory, scen_id, agents)
        plot(all_curtailment, "curtailment", directory, scen_id, agents)
        plot(all_storage, "storage", directory, scen_id, agents)

    # Convert results to DataFrame
    df_rewards = pd.DataFrame(results_rewards, index=scen_test)
    df_timesteps = pd.DataFrame(results_timesteps, index=scen_test)

    # Combine rewards and timesteps into a single DataFrame
    df_combined = pd.concat(
        [df_rewards.add_suffix(" Reward"), df_timesteps.add_suffix(" Timesteps")],
        axis=1,
    )
    return df_combined


# Run the simulation with all agents
agents = [do_nothing_agent, agent]
df = run_simulation(scen_test, env, agents)
print(df)
df.to_csv("results.csv", index=False)
