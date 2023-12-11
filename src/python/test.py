# -*- coding: utf-8 -*-
""" Evaluate the agent on the validation set. """
import argparse
from typing import Tuple, Union
import os
from pathlib import Path

from environments.Grid2OpResdispatchCurtail import Grid2OpEnvRedispatchCurtail
from environments.Grid2OpResdispatchCurtailFlattened import (
    Grid2OpEnvRedispatchCurtailFlattened,
)
from PPO import PPO
import pandas as pd
from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from grid2op.Environment import Environment
from agents.OptimCVXPY import make_agent
from grid2op.Agent import BaseAgent, DoNothingAgent


def get_model(
    model_name: str,
    env: Grid2OpEnvRedispatchCurtail,
    checkpoint_path: Union[Path, None] = None,
):
    if model_name == "graphnet":
        model = PPO(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_actor=0.0003,
            lr_critic=0.001,
            gamma=0.99,
            K_epochs=4,
            eps_clip=0.2,
            has_continuous_action_space=True,
        )
        if not isinstance(checkpoint_path, Path):
            raise ValueError("Checkpoint path must be specified for GraphNetAgent")
        model.load(checkpoint_path)
    elif model_name == "optim":
        model = make_agent(env.get_grid2op_env())
    elif model_name == "donothing":
        model = DoNothingAgent(env.get_grid2op_env().action_space)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


# class NoActionAgent:
#     def __init__(self):
#         pass

#     def select_action_eval(self, obs):
#         return None


# class FullUpAgent:
#     def __init__(self):
#         pass

#     def select_action_eval(self, obs):
#         return np.ones(6)


# class FullDownAgent:
#     def __init__(self):
#         pass

#     def select_action_eval(self, obs):
#         return -np.ones(6)


def play_episode(
    env: Grid2OpEnvRedispatchCurtail,
    model,
    episode_id: int,
    progress,
    task_step,
) -> Tuple[float, int]:
    obs, _ = env.reset(seed=episode_id, set_id=episode_id)
    done, time_step, reward = False, 0, 0
    while not done:
        progress.advance(task_step, 1)
        action = model.select_action_eval(obs)
        obs, step_reward, done, _, _ = env.step(action)
        reward += step_reward
        time_step += 1
    progress.advance(task_step, -time_step)
    return reward, time_step


def play_episode_default(
    env: Environment, model: BaseAgent, episode_id: int, progress, task_step
) -> Tuple[float, int]:
    env.set_id(episode_id)
    obs = env.reset()
    done, time_step, reward = False, 0, env.reward_range[0]  # type: ignore
    total_reward = 0
    while not done:
        progress.advance(task_step, 1)
        action = model.act(obs, reward)  # type: ignore
        obs, step_reward, done, _ = env.step(action)
        total_reward += step_reward
        time_step += 1
    progress.advance(task_step, -time_step)
    return total_reward, time_step


def test_env(
    env: Grid2OpEnvRedispatchCurtailFlattened, model, n_episodes=1
) -> pd.DataFrame:
    rewards = []
    episode_length = []
    with Progress(
        BarColumn(), TimeElapsedColumn(), TimeRemainingColumn(), MofNCompleteColumn()
    ) as progress:
        task_episode = progress.add_task("[red]Episode...", total=n_episodes)
        task_step = progress.add_task(
            "[green]Step...",
            total=env.get_grid2op_env().chronics_handler.max_timestep(),
        )
        for episode_idx in range(n_episodes):
            rewards.append(0)
            episode_length.append(0)
            if issubclass(type(model), BaseAgent):
                reward, length = play_episode_default(
                    env.get_grid2op_env(), model, episode_idx, progress, task_step
                )
            else:
                reward, length = play_episode(
                    env, model, episode_idx, progress, task_step
                )
            rewards.append(reward)
            episode_length.append(length)
            progress.advance(task_episode, 1)
            progress.advance(task_step, -env.get_time_step())

            print(f"Episode {episode_idx + 1}/{n_episodes}")
            print(f"Episode reward: {rewards[-1]}")
            print(
                f"Episode length: {episode_length[-1]} / {env.get_grid2op_env().chronics_handler.max_timestep()}"
            )
    print(f"Average reward: {sum(rewards) / n_episodes}")
    print(f"Average episode length: {sum(episode_length) / n_episodes}")
    df = pd.DataFrame({"rewards": rewards, "episode_length": episode_length})
    return df


# def convert_figure_to_numpy_HWC(figure):
#     """Convert a Matplotlib figure to a numpy array with format HWC."""
#     w, h = figure.get_size_inches() * figure.dpi
#     w = int(w)
#     h = int(h)
#     buf = io.BytesIO()
#     figure.canvas.print_raw(buf)
#     buf.seek(0)
#     img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
#     buf.close()
#     img_arr = np.reshape(img_arr, (h, w, 4))
#     return img_arr


# def create_plot(
#     obs, env: Grid2OpEnvRedispatchCurtail, gen_info: str = "actual_dispatch"
# ):
#     """Create a plot of the environment."""
#     plot_helper = PlotMatplot(env.get_grid2op_env().observation_space)
#     # fig = plot_helper.plot_obs(obs, gen_info="actual_dispatch")
#     fig = plot_helper.plot_obs(obs, gen_info=gen_info)
#     img_arr = convert_figure_to_numpy_HWC(fig)
#     plt.close(fig)  # Close the figure to free memory
#     return img_arr


# def plot_env(
#     model,
#     env: Grid2OpEnvRedispatchCurtail,
#     dir: Path,
#     chronic_id: int = 1,
# ):
#     """Plot the environment for a given chronic."""
#     dir.mkdir(parents=True, exist_ok=True)
#     obs, _ = env.reset(seed=chronic_id, set_id=chronic_id)
#     done, time_step = False, 0

#     categories = ["actual_dispatch", "target_dispatch", "p"]
#     images = [[] for _ in categories]

#     with Progress(
#         BarColumn(), TimeElapsedColumn(), TimeRemainingColumn(), MofNCompleteColumn()
#     ) as progress:
#         task_step = progress.add_task(
#             "[green]Step...",
#             total=env.get_grid2op_env().chronics_handler.max_timestep(),
#         )

#         while not done:
#             action = model.select_action_eval(obs)
#             obs, _, done, _, _ = env.step(action)
#             progress.advance(task_step, 1)

#             # Get the current frame as a numpy array
#             for i, category in enumerate(categories):
#                 img_arr = create_plot(
#                     obs=env.get_grid2op_obs(), env=env, gen_info=category
#                 )
#                 images[i].append(img_arr)

#             time_step += 1

#             if time_step > 100:
#                 break

#         # Create a GIF
#         for i, category in enumerate(categories):
#             with imageio.get_writer(dir / f"agent_{category}.gif", mode="I") as writer:
#                 for image in images[i]:
#                     writer.append_data(image)  # type: ignore


# def plot_power_generators(  # pylint: disable=too-many-locals
#     model, env: Grid2OpEnvRedispatchCurtail, directory: Path, chronic_id: int = 1
# ) -> List[float]:
#     """Plot the power generators for a given chronic."""
#     directory.mkdir(parents=True, exist_ok=True)
#     obs, _ = env.reset(seed=chronic_id, set_id=chronic_id)
#     done, time_step = False, 0

#     # Initialize lists to store values for each generator
#     power_levels = [[] for _ in range(6)]
#     redispatch_values = [[] for _ in range(6)]
#     rewards = []
#     with Progress(
#         BarColumn(), TimeElapsedColumn(), TimeRemainingColumn(), MofNCompleteColumn()
#     ) as progress:
#         task_step = progress.add_task(
#             "[green]Step...",
#             total=env.get_grid2op_env().chronics_handler.max_timestep(),
#         )

#         while True:
#             action = model.select_action_eval(obs)
#             obs, reward, done, _, _ = env.step(action)
#             rewards.append(reward)
#             progress.advance(task_step, 1)

#             # Update lists with values from the current step
#             power_level = env.get_grid2op_obs().prod_p

#             if done:
#                 break

#             if action is not None:
#                 unormalized_redispatch = env.denormalize_action(action)
#             else:
#                 unormalized_redispatch = np.zeros_like(power_level)
#             for i in range(6):
#                 power_levels[i].append(power_level[i])
#                 redispatch_values[i].append(unormalized_redispatch[i])

#             time_step += 1

#     # Plotting
#     fig, axes = plt.subplots(6, 1, figsize=(10, 15), tight_layout=True)
#     for i in range(6):
#         axes[i].plot(power_levels[i], label="Power Level")
#         axes[i].plot(redispatch_values[i], label="Redispatching Value")
#         axes[i].set_title(f"Generator {i+1}")
#         axes[i].legend()
#         axes[i].grid(True)

#     plt.savefig(directory / "power_generators_plot.pdf")
#     plt.close(fig)

#     # Plotting
#     fig, axes = plt.subplots(6, 1, figsize=(10, 15), tight_layout=True)
#     for i in range(6):
#         axes[i].plot(redispatch_values[i], label="Redispatching Value")
#         axes[i].set_title(f"Generator {i+1}")
#         axes[i].legend()
#         axes[i].grid(True)

#     plt.savefig(directory / "redispatching_plot.pdf")
#     plt.close(fig)

#     return rewards


def make_env() -> Grid2OpEnvRedispatchCurtail:
    env = Grid2OpEnvRedispatchCurtail("l2rpn_case14_sandbox_val")
    env.get_grid2op_env().chronics_handler.seed(1)
    env.get_grid2op_env().seed(1)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default="logs/PPO/Grid2OpEnvRedispatchCurtailFlattened//2023-11-27-14-14-14/time_step_7627532/",
    )
    parser.add_argument("--n_episodes", type=int, default=10)
    args = parser.parse_args()
    checkpoint_path = (
        args.checkpoint_dir
        / [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pth")][0]
    )

    # env = Grid2OpEnvRedispatchCurtailFlattened("l2rpn_case14_sandbox_val")
    # env.grid2op_env.chronics_handler.seed(1)
    # env.grid2op_env.seed(1)
    # model = get_model(env, checkpoint_path)

    # for i in range(10):
    #     directory = args.checkpoint_dir / f"chronic{i}"
    #     print(f"Plotting power generators for chronic {i}")
    #     rewards = plot_power_generators(model, env, directory, chronic_id=i)
    #     rewards_no_action = plot_power_generators(
    #         NoActionAgent(), env, directory / "baseline", chronic_id=i
    #     )
    #     rewards_full_up = plot_power_generators(
    #         FullUpAgent(), env, directory / "full_up", chronic_id=i
    #     )
    #     rewards_full_down = plot_power_generators(
    #         FullDownAgent(), env, directory / "full_down", chronic_id=i
    #     )

    #     print(f"Plotting rewards for chronic {i}")
    #     plt.figure()
    #     plt.plot(rewards, label="Agent")
    #     plt.plot(rewards_no_action, label="Baseline")
    #     plt.plot(rewards_full_up, label="Full Up")
    #     plt.plot(rewards_full_down, label="Full Down")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(directory / "rewards_plot.pdf")
    #     plt.close()

    #     print(f"Plotting environment for chronic {i}")
    #     plot_env(model, env, directory, chronic_id=i)
    # plot_env(NoActionAgent(), env, directory / "baseline", chronic_id=i)
    # plot_env(FullUpAgent(), env, directory / "full_up", chronic_id=i)
    # plot_env(FullDownAgent(), env, directory / "full_down", chronic_id=i)

    # print("Testing agent")
    # model_df = test_env(env, model, args.n_episodes)
    # baseline_df = test_env(env, NoActionAgent(), args.n_episodes)
    # print("Done testing agent")

    # baseline_df = baseline_df.rename(
    #     columns={
    #         "rewards": "no_action_rewards",
    #         "episode_length": "no_action_episode_length",
    #     }
    # )
    env = make_env()
    result_df_donothing = test_env(env, get_model("donothing", env), n_episodes=10)
    env = make_env()
    result_df_optim = test_env(env, get_model("optim", env), n_episodes=10)
    combined_df = pd.concat([result_df_optim, result_df_donothing], axis=1)
    print(combined_df)
    combined_df.to_csv(args.checkpoint_dir / "results.csv")
