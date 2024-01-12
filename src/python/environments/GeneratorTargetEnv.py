# -*- coding: utf-8 -*-
from typing import Any
import grid2op
from grid2op.Reward import LinesCapacityReward
from lightsim2grid import LightSimBackend
from gymnasium import Env
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image
from gymnasium import spaces
import torch


class GeneratorTargetEnv(Env):
    def __init__(
        self, env_name: str = "l2rpn_case14_sandbox", render_mode: str = "rgb_array"
    ) -> None:
        super().__init__()
        self.env_name = env_name
        self.env = grid2op.make(
            env_name,
            reward_class=LinesCapacityReward,
            backend=LightSimBackend(),
            experimental_read_from_local_dir=True,
        )
        self.n_gen = self.env.n_gen

        # Observation space normalization factors
        self.gen_pmax = torch.tensor(self.env.observation_space.gen_pmax)
        self.gen_pmin = torch.tensor(self.env.observation_space.gen_pmin)
        assert torch.all(self.gen_pmax >= self.gen_pmin) and torch.all(
            self.gen_pmin >= 0
        )  # type: ignore

        # Observation space observation
        self.observation_space = spaces.Box(
            low=-self.gen_pmax.max().item(),
            high=self.gen_pmax.max().item(),
            shape=(3 * self.n_gen,),
            dtype=np.float32,
        )

        # Action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_gen,), dtype=np.float32
        )

        # Action space normalization factor
        self.action_norm_factor = np.maximum(
            self.env.observation_space.gen_max_ramp_up,  # type: ignore
            -self.env.observation_space.gen_max_ramp_down,  # type: ignore
        )
        self.max_episode_length = 100
        self.render_mode = render_mode

    def denormalize_action(self, action):
        action = np.tanh(action)
        action = action * self.action_norm_factor
        return action

    def set_target_state(self):
        self.target_state = torch.tensor(
            [
                np.random.uniform(low=pmi, high=pmx)
                for pmi, pmx in zip(self.gen_pmin, self.gen_pmax)
            ]
        )

    def observe(self):
        obs = torch.stack(
            [
                self.curr_state,
                self.target_state,
                self.curr_state - self.target_state,
            ]
        )
        obs = obs.flatten()
        assert self.observation_space.contains(obs)
        return obs

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        self.set_target_state()
        self.curr_state = torch.zeros_like(self.target_state)
        self.n_steps = 0
        return self.observe(), {}

    def compute_distance(self):
        return torch.abs(self.curr_state - self.target_state).sum()

    def step(self, action):
        # assert self.action_space.contains(action)
        action = self.denormalize_action(action)
        initial_distance = self.compute_distance()
        self.curr_state += action
        self.curr_state = torch.clip(
            self.curr_state,
            self.gen_pmin,
            self.gen_pmax,
        )
        new_distance = self.compute_distance()
        reward = (initial_distance - new_distance) / 100
        self.n_steps += 1
        done = self.n_steps >= self.max_episode_length
        return self.observe(), reward, done, False, {}

    def render(self):
        # Calculate x-axis limits once
        x_min = self.env.observation_space.gen_pmin.min()  # type: ignore
        x_max = self.env.observation_space.gen_pmax.max()  # type: ignore

        # Create subplots
        fig, axs = plt.subplots(
            self.n_gen, 1, figsize=(10, self.n_gen * 2), tight_layout=True
        )

        for i, ax in enumerate(axs):
            ax.set_xlim(x_min, x_max)
            ax.scatter(self.target_state[i], 0.5, c="red")
            ax.scatter(self.curr_state[i], 0.5, c="blue")
            ax.yaxis.set_visible(False)

        # Set common legend outside the loop
        axs[0].legend(["Target", "Agent"], loc="upper right")
        # Render the plot to a canvas and then convert to an RGB array
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        img_arr = np.array(img)

        return img_arr
