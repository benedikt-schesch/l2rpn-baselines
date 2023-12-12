# -*- coding: utf-8 -*-
from typing import Union, Tuple
import torch
from gymnasium import Env
from gymnasium import spaces
from collections import OrderedDict
from OptimizerNoCurtailement import OptimCVXPY
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from grid2op.Chronics import GridStateFromFileWithForecastsWithoutMaintenance
from grid2op.Action import DontAct, PlayableAction
from grid2op.Opponent import BaseOpponent, NeverAttackBudget
import grid2op
from grid2op.Reward import LinesCapacityReward
from lightsim2grid import LightSimBackend
import numpy as np


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
        self.optimizer = OptimCVXPY(
            self.get_grid2op_env().action_space,
            self.get_grid2op_env(),
            lines_x_pu=None,
            margin_th_limit=0.9,
            alpha_por_error=0.5,
            rho_danger=0.9,
            margin_rounding=0.01,
            margin_sparse=5e-3,
        )

    def flatten_features(self, obs: BaseObservation) -> torch.Tensor:
        features = [obs.rho, obs.load_p, obs.load_q, obs.gen_p, obs.gen_q]
        features = torch.tensor(np.concatenate(features))
        return features

    def reset(self, seed: Union[None, int] = None) -> Tuple[torch.Tensor, dict]:
        self.grid2op_env.set_id(2)
        grid2op_obs = self.grid2op_env.reset()
        self.latest_obs = grid2op_obs
        self.done = False
        self.time_step = 0
        self.play_until_unsafe()
        obs = self.flatten_features(self.latest_obs)
        return obs, {}

    def play_until_unsafe(self):
        while self.latest_obs.rho.max() < self.optimizer.rho_danger and not self.done:
            grid2op_act = self.optimizer.act(self.latest_obs)
            self.latest_obs, reward, self.done, info = self.grid2op_env.step(
                grid2op_act
            )
            self.time_step += 1

    def step(
        self, action: Union[None, torch.Tensor]
    ) -> Tuple[torch.Tensor, float, bool, bool, dict]:
        self.optimizer.set_target_injected_power(action)
        grid2op_act = self.optimizer.act(self.latest_obs)
        self.latest_obs, reward, self.done, info = self.grid2op_env.step(grid2op_act)
        self.play_until_unsafe()
        obs = self.flatten_features(self.latest_obs)
        return obs, reward, self.done, False, info

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


if __name__ == "__main__":
    env = Grid2OpBilevelFlattened("educ_case14_storage")
    for i in range(1):
        obs, _ = env.reset()
        done = False
        cum_reward = 0
        while not done:
            action = np.ones(env.max_nb_bus) * 1000
            obs, reward, done, _, info = env.step(action)
            cum_reward += reward
        print("Cumulative reward", cum_reward)
        print("Done", env.get_time_step())
