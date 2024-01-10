# -*- coding: utf-8 -*-
# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm
import grid2op
from grid2op.Action import PlayableAction
from OptimCVXPY_no_storage import OptimCVXPY as OptimCVXPY_no_storage
from OptimCVXPY_no_redispatch import OptimCVXPY as OptimCVXPY_no_redispatch
from OptimCVXPY_no_curtailment import OptimCVXPY as OptimCVXPY_no_curtailment
from OptimCVXPY_custom import OptimCVXPY as OptimCVXPY_custom
from OptimCVXPY import OptimCVXPY
from lightsim2grid import LightSimBackend

max_step = 288


def get_model(model_name):
    if model_name == "full":
        return OptimCVXPY
    elif model_name == "no_storage":
        return OptimCVXPY_no_storage
    elif model_name == "no_redispatch":
        return OptimCVXPY_no_redispatch
    elif model_name == "no_curtailment":
        return OptimCVXPY_no_curtailment
    elif model_name == "custom":
        return OptimCVXPY_custom
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--rho_safe", type=float, default=0.85, help="rho_safe parameter"
    )
    argparser.add_argument(
        "--rho_danger", type=float, default=0.95, help="rho_danger parameter"
    )
    argparser.add_argument(
        "--model",
        type=str,
        default="full",
        help="model to test: full, custom, no_storage, no_redispatch, no_curtailment",
    )
    argparser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results_optimcvxpy"),
        help="directory to save results",
    )
    args = argparser.parse_args()

    env = grid2op.make(
        "educ_case14_storage",
        test=True,
        backend=LightSimBackend(),
        action_class=PlayableAction,
    )

    model_class = get_model(args.model)
    agent = model_class(
        env.action_space,
        env,
        penalty_redispatching_unsafe=0.0,
        penalty_storage_unsafe=0.04,
        penalty_curtailment_unsafe=0.01,
        rho_safe=args.rho_safe,
        rho_danger=args.rho_danger,
        margin_th_limit=0.93,
        alpha_por_error=0.5,
        weight_redisp_target=0.3,
    )

    # in safe / recovery mode agent tries to fill the storage units as much as possible
    agent.storage_setpoint = env.storage_Emax

    print("For do nothing: ")
    dn_act = env.action_space()
    results_df = pd.DataFrame(
        columns=[
            "baseline episode length",
            "basline reward",
            "agent episode length",
            "agent reward",
        ]
    )
    for scen_id in range(7):
        env.set_id(scen_id)
        obs = env.reset()
        done = False
        cum_reward = 0
        for nb_step in tqdm(range(max_step)):
            obs, reward, done, info = env.step(dn_act)
            cum_reward += reward
            if done and nb_step != (max_step - 1):
                break
        print(
            f"\t scenario: {os.path.split(env.chronics_handler.get_id())[-1]}: {nb_step + 1} / {max_step}"
        )
        results_df.loc[scen_id, "baseline episode length"] = nb_step + 1
        results_df.loc[scen_id, "basline reward"] = cum_reward
    # Average over all scenarios
    results_df.loc["mean", "baseline episode length"] = results_df[
        "baseline episode length"
    ].mean()
    results_df.loc["mean", "basline reward"] = results_df["basline reward"].mean()
    results_df.loc["std", "baseline episode length"] = results_df[
        "baseline episode length"
    ].std()
    results_df.loc["std", "basline reward"] = results_df["basline reward"].std()

    print("For the optimizer: ")
    for scen_id in range(7):
        env.set_id(scen_id)
        obs = env.reset()
        agent.reset(obs)
        done = False
        cum_reward = 0
        for nb_step in tqdm(range(max_step)):
            prev_obs = obs
            act = agent.act(obs)
            obs, reward, done, info = env.step(act)
            cum_reward += reward
            if done and nb_step != (max_step - 1):
                # there is a game over before the end
                break
        print(
            f"\t scenario: {os.path.split(env.chronics_handler.get_id())[-1]}: {nb_step + 1} / {max_step}"
        )
        results_df.loc[scen_id, "agent episode length"] = nb_step + 1
        results_df.loc[scen_id, "agent reward"] = cum_reward
    # Average over all scenarios
    results_df.loc["mean", "agent episode length"] = results_df[
        "agent episode length"
    ].mean()
    results_df.loc["mean", "agent reward"] = results_df["agent reward"].mean()
    results_df.loc["std", "agent episode length"] = results_df[
        "agent episode length"
    ].std()
    results_df.loc["std", "agent reward"] = results_df["agent reward"].std()

    args.output_dir.mkdir(exist_ok=True)
    results_df.to_csv(
        args.output_dir
        / f"results_{args.model}_rho_safe_{args.rho_safe}_rho_danger_{args.rho_danger}.csv",
        index=True,
    )
