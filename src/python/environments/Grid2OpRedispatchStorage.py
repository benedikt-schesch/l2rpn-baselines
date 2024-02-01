# -*- coding: utf-8 -*-
from typing import Union, Tuple
from gymnasium import Env
from gymnasium import spaces
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from grid2op.Action import PlayableAction
import grid2op
from lightsim2grid import LightSimBackend
import numpy as np


class Grid2OpRedispatchStorage(Env):
    def __init__(
        self,
        env_name: str = "educ_case14_storage",
        episode_ids=[6],
        features=["timestep_idx"],
    ) -> None:
        super().__init__()
        # Initialize the Grid2OpEnvRedispatchCurtailFlattened environment
        self.grid2op_env = grid2op.make(
            env_name,
            test=True,
            backend=LightSimBackend(),
            action_class=PlayableAction,
        )
        self.features = features
        assert len(self.features) > 0, "At least one feature must be selected"
        self.max_episode_length = self.grid2op_env.chronics_handler.max_timestep()
        self.time_step = 0
        self.episode_ids = episode_ids
        self.number_of_episodes = len(self.grid2op_env.chronics_handler.subpaths)
        flat_features, _ = self.reset()
        self.observation_space = spaces.Box(
            low=-float("inf"), high=float("inf"), shape=flat_features.shape
        )
        self.action_space = spaces.Box(
            low=-np.concatenate(
                [
                    self.grid2op_env.storage_max_p_absorb,  # type: ignore
                    self.grid2op_env.gen_max_ramp_down,
                ]
            ),
            high=np.concatenate(
                [self.grid2op_env.storage_max_p_prod, self.grid2op_env.gen_max_ramp_up]  # type: ignore
            ),
            shape=(self.grid2op_env.n_storage + self.grid2op_env.n_gen,),
        )

    def flatten_features(self, obs: BaseObservation) -> np.ndarray:
        # One hot encoding of the time step
        features = []
        if "gen_p" in self.features:
            features.append(obs.gen_p)
        if "gen_v" in self.features:
            features.append(obs.gen_v)
        if "load_p" in self.features:
            features.append(obs.load_p)
        if "load_q" in self.features:
            features.append(obs.load_q)
        if "rho" in self.features:
            features.append(obs.rho)
        if "episode_idx" in self.features:
            episode_idx = np.zeros(self.number_of_episodes)
            episode_idx[self.episode_ids.index(self.episode_id)] = 1
            features.append(episode_idx)
        if "timestep_idx" in self.features:
            timestep_idx = np.zeros(self.max_episode_length + 1)
            timestep_idx[self.time_step] = 1
            features.append(timestep_idx)
        return np.concatenate(features)

    def reset(self, seed: Union[None, int] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None and seed not in self.episode_ids:
            np.random.seed(seed)
            self.episode_id = np.random.choice(self.episode_ids)
        elif seed is not None and seed in self.episode_ids:
            self.episode_id = seed
        else:
            self.episode_id = np.random.choice(self.episode_ids)
        self.grid2op_env.set_id(self.episode_id)
        self.latest_obs = self.grid2op_env.reset()
        self.done = False
        self.time_step = 0
        obs = self.flatten_features(self.latest_obs)
        info = {"grid2op_obs": self.latest_obs}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        grid2op_act = self.grid2op_env.action_space({})
        grid2op_act.storage_p = action[: self.grid2op_env.n_storage]
        grid2op_act.redispatch = action[self.grid2op_env.n_storage :]
        self.latest_obs, reward, self.done, info = self.grid2op_env.step(grid2op_act)
        self.time_step += 1
        obs = self.flatten_features(self.latest_obs)
        info["grid2op_action"] = grid2op_act
        info["grid2op_obs"] = self.latest_obs
        return obs, (3 + reward) / (3.0 * 288), self.done, False, info

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
