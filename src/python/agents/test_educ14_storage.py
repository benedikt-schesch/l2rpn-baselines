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
from grid2op.Action import PlayableAction
from OptimCVXPY_no_storage import OptimCVXPY as OptimCVXPY_no_storage
from OptimCVXPY_no_redispatch import OptimCVXPY as OptimCVXPY_no_redispatch
from OptimCVXPY_no_curtailment import OptimCVXPY as OptimCVXPY_no_curtailment
from OptimCVXPY import OptimCVXPY
from lightsim2grid import LightSimBackend
from tqdm import tqdm
import argparse

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
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--rho_safe", type=float, default=0.0, help="rho_safe parameter"
    )
    argparser.add_argument(
        "--rho_danger", type=float, default=0.0, help="rho_danger parameter"
    )
    argparser.add_argument(
        "--model",
        type=str,
        default="full",
        help="model to test: full, no_storage, no_redispatch, no_curtailment",
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
    for scen_id in range(7):
        env.set_id(scen_id)
        obs = env.reset()
        done = False
        for nb_step in tqdm(range(max_step)):
            obs, reward, done, info = env.step(dn_act)
            if done and nb_step != (max_step - 1):
                break
        print(
            f"\t scenario: {os.path.split(env.chronics_handler.get_id())[-1]}: {nb_step + 1} / {max_step}"
        )

    print("For the optimizer: ")
    for scen_id in range(7):
        env.set_id(scen_id)
        obs = env.reset()
        agent.reset(obs)
        done = False
        for nb_step in tqdm(range(max_step)):
            prev_obs = obs
            act = agent.act(obs)
            obs, reward, done, info = env.step(act)
            if done and nb_step != (max_step - 1):
                # there is a game over before the end
                break
        print(
            f"\t scenario: {os.path.split(env.chronics_handler.get_id())[-1]}: {nb_step + 1} / {max_step}"
        )
