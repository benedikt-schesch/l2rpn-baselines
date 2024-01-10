# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import torch
from environments.Grid2OpBilevelFlattened import (
    Grid2OpBilevelFlattened,
    plot_generator_power,
    plot_storage_charge,
    plot_actions,
)
from agents.GraphNetAgent import ActorCritic


def play_episode(env, agent):
    obs, _ = env.reset()
    done = False
    cum_reward = 0
    infos = []
    for i in range(env.get_grid2op_env().chronics_handler.max_timestep()):
        action = agent.act_eval(obs)
        obs, reward, done, _, info = env.step(action.detach().cpu().numpy().flatten())
        info["obs"] = env.get_grid2op_obs()
        infos.append(info)
        cum_reward += reward
        if done:
            break
    print("Cumulative reward", cum_reward)
    print("Done", i)
    return infos


class NoAction:
    def __init__(self, env):
        self.n = env.action_space.shape[0]

    def act_eval(self, state):
        return torch.zeros(self.n)


if __name__ == "__main__":
    plot_dir = Path("action_plots")
    plot_dir.mkdir(exist_ok=True)
    np.random.seed(0)
    env = Grid2OpBilevelFlattened("educ_case14_storage")
    agent = ActorCritic(env.observation_space, env.action_space)
    agent.load_state_dict(
        torch.load(
            "logs/PPO/educ_case14_storage/2023-12-23-03-27-22/PPO_educ_case14_storage_42_best.pth"
        )
    )
    infos = [play_episode(env, NoAction(env)), play_episode(env, agent)]
    plot_generator_power(infos, plot_dir, ["No Action", "PPO"])
    plot_storage_charge(infos, plot_dir, ["No Action", "PPO"])
    plot_actions(infos, plot_dir, ["No Action", "PPO"])
