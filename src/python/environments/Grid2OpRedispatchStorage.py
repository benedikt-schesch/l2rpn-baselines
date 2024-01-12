# -*- coding: utf-8 -*-
from typing import Union, Tuple
from gymnasium import Env
from gymnasium import spaces
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from grid2op.Chronics import GridStateFromFileWithForecastsWithoutMaintenance
from grid2op.Action import DontAct, PlayableAction
from grid2op.Opponent import BaseOpponent, NeverAttackBudget
import grid2op
from grid2op.Reward import LinesCapacityReward
from lightsim2grid import LightSimBackend
import numpy as np


class Grid2OpRedispatchStorage(Env):
    def __init__(self, env_name: str = "educ_case14_storage") -> None:
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
        self.max_episode_length = self.grid2op_env.chronics_handler.max_timestep()
        self.time_step = 0
        obs = self.grid2op_env.reset()
        flat_features = self.flatten_features(obs)
        self.observation_space = spaces.Box(
            low=-float("inf"), high=float("inf"), shape=flat_features.shape
        )
        self.action_space = spaces.Box(
            low=-np.concatenate(
                [
                    self.grid2op_env.storage_max_p_absorb,
                    self.grid2op_env.gen_max_ramp_down,
                ]
            ),
            high=np.concatenate(
                [self.grid2op_env.storage_max_p_prod, self.grid2op_env.gen_max_ramp_up]
            ),
            shape=(self.grid2op_env.n_storage + self.grid2op_env.n_gen,),
        )

    def flatten_features(self, obs: BaseObservation) -> np.ndarray:
        # One hot encoding of the time step
        features = np.zeros(self.max_episode_length)
        features[self.time_step] = 1
        return features

    def reset(self, seed: Union[None, int] = None) -> Tuple[np.ndarray, dict]:
        self.grid2op_env.set_id(6)
        grid2op_obs = self.grid2op_env.reset()
        self.latest_obs = grid2op_obs
        self.done = False
        self.time_step = 0
        obs = self.flatten_features(self.latest_obs)
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        grid2op_act = self.grid2op_env.action_space({})
        grid2op_act.storage_p = action[: self.grid2op_env.n_storage]
        grid2op_act.redispatch = action[self.grid2op_env.n_storage :]
        self.latest_obs, reward, self.done, info = self.grid2op_env.step(grid2op_act)
        self.time_step += 1
        obs = self.flatten_features(self.latest_obs)
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


if __name__ == "__main__":
    env = Grid2OpRedispatchStorage()
    env.reset()
    env.step(env.action_space.sample())
