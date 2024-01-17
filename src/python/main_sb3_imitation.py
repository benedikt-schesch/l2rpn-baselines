# -*- coding: utf-8 -*-
"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import types
from environments.Grid2OpRedispatchStorage import Grid2OpRedispatchStorage
from agents.OptimCVXPY import OptimCVXPY
from tqdm import tqdm

env = Grid2OpRedispatchStorage()
rng = np.random.default_rng(0)


def sample_expert_transitions():
    expert = OptimCVXPY(
        env.get_grid2op_env().action_space,
        env.get_grid2op_env(),
        penalty_redispatching_unsafe=0.0,
        penalty_storage_unsafe=0.04,
        penalty_curtailment_unsafe=0.01,
        rho_safe=0.85,
        rho_danger=0.95,
        margin_th_limit=0.93,
        alpha_por_error=0.5,
        weight_redisp_target=0.3,
    )
    # expert = OptimCVXPY(env.get_grid2op_env().action_space, env.get_grid2op_env())
    keys = ["obs", "next_obs", "acts", "dones", "infos"]
    trajectories = {k: [] for k in keys}
    obs, info = env.reset()
    for i in tqdm(range(288)):
        grid2op_action = expert.act(info["grid2op_obs"])
        action = np.concatenate([grid2op_action.storage_p, grid2op_action.redispatch])
        next_obs, reward, done, _, info = env.step(action)
        trajectories["obs"].append(obs)
        trajectories["next_obs"].append(next_obs)
        trajectories["acts"].append(action)
        trajectories["dones"].append(done)
        trajectories["infos"].append({})
        obs = next_obs
        if done:
            print("Done", i)
            obs, info = env.reset()
    for k in keys:
        trajectories[k] = np.array(trajectories[k])
    return types.Transitions(**trajectories)


transitions = sample_expert_transitions()
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    ent_weight=0,
    rng=rng,
    device="cpu",
)

evaluation_env = Grid2OpRedispatchStorage()

print("Evaluating the untrained policy.")
rewards, episode_lengths = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    evaluation_env,
    n_eval_episodes=3,
    return_episode_rewards=True,
)
print(f"Episode lengths before training: {episode_lengths}")

print("Training a policy using Behavior Cloning")
bc_trainer.train(n_epochs=1000)

print("Evaluating the trained policy.")
rewards, episode_lengths = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    evaluation_env,
    n_eval_episodes=3,
    return_episode_rewards=True,
)
print(f"Episode lengths after training: {episode_lengths}")
