# -*- coding: utf-8 -*-
from grid2op.Agent import DoNothingAgent
import numpy as np
import milp_agent
from milp_agent.agent import MILPAgent
from grid2op.Runner import Runner
from OptimCVXPY import get_env

env = get_env()
runner_params = env.get_params_for_runner()
runner_params["verbose"] = True
runner = Runner(
    **runner_params, agentClass=None, agentInstance=DoNothingAgent(env.action_space)
)
print("start")
res = runner.run(
    nb_episode=1,
    pbar=True,
)
print(res, "\n")

# env_name = "l2rpn_case14_sandbox"
env = get_env()
obs = env.get_obs()
margins = 0.95 * np.ones(obs.n_line)
obs = env.get_obs()
Agent_type = milp_agent.GLOBAL_SWITCH
Solver_type = milp_agent.MIP_CBC
agent = MILPAgent(env, Agent_type, margins, solver_name=Solver_type)
# and now you can use it as any grid2op compatible agent, for example:
# action = agent.act(obs, reward=0.0, done=False)
# new_state, reward, done, info = env.step(action)
runner_params = env.get_params_for_runner()
runner_params["verbose"] = True
runner = Runner(**runner_params, agentClass=None, agentInstance=agent)
res = runner.run(
    nb_episode=1,
    pbar=True,
)
