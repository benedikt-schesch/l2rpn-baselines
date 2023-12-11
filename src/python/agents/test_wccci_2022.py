# -*- coding: utf-8 -*-
# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import grid2op
from grid2op.Agent import RecoPowerlineAgent, DoNothingAgent
from OptimCVXPY import OptimCVXPY
from lightsim2grid import LightSimBackend
from tqdm import tqdm
import numpy as np
from numpy.random import default_rng

prng = default_rng(0)

env_name = "l2rpn_wcci_2022"
is_test = False

env = grid2op.make(env_name, test=is_test, backend=LightSimBackend())
logger = None

reco_agent = RecoPowerlineAgent(env.action_space)

agent = OptimCVXPY(
    env.action_space,
    env,
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


# Modified main simulation loop with data collection for redispatching
def run_simulation(scen_test, env, agents):
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

            for nb_step in tqdm(range(obs.max_step)):
                act = agent.act(obs, reward)
                obs, reward, done, _ = env.step(act)

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
        # # Call the plotting functions after each scenario
        # plot(all_power_levels, "power levels",directory, scen_id)
        # plot(all_redispatching, "redispatching",directory, scen_id)
        # plot(all_curtailment, "curtailment",directory, scen_id)
        # plot(all_storage, "storage",directory, scen_id)


run_simulation(scen_test, env, [DoNothingAgent(env.action_space)])


print("For do nothing: ")
dn_act = env.action_space()
for i, scen_id in enumerate(scen_test):
    env.set_id(scen_id)
    env.seed(seeds[scen_id])
    obs = env.reset()
    done = False
    for nb_step in tqdm(range(obs.max_step)):
        prev_obs = obs
        act = reco_agent.act(obs, None, None)
        obs, reward, done, info = env.step(dn_act)
        if done and (nb_step != prev_obs.max_step - 1):
            break
    print(
        f"\t scenario: {os.path.split(env.chronics_handler.get_id())[-1]}: {nb_step + 1} / {obs.max_step}"
    )

# print("For the optimizer: ")
# for i, scen_id in enumerate(scen_test):
#     act = None
#     env.set_id(scen_id)
#     env.seed(seeds[scen_id])
#     obs = env.reset()
#     agent.reset(obs)
#     done = False
#     for nb_step in tqdm(range(obs.max_step)):
#         prev_obs = obs
#         prev_act = act
#         act = agent.act(obs)
#         obs, reward, done, info = env.step(act)
#         if done and (nb_step != prev_obs.max_step - 1):
#             break
#     print(f"\t scenario: {os.path.split(env.chronics_handler.get_id())[-1]}: {nb_step + 1} / {obs.max_step}")
