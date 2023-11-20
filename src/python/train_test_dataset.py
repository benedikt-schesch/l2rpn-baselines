import grid2op
from grid2op.Reward import LinesCapacityReward
from grid2op.Chronics import MultifolderWithCache
from lightsim2grid import LightSimBackend
from grid2op.Chronics import GridStateFromFileWithForecastsWithoutMaintenance
from grid2op.Action import DontAct
from grid2op.Opponent import BaseOpponent, NeverAttackBudget

env_name = "l2rpn_case14_sandbox"
env = grid2op.make(
    env_name,
    reward_class=LinesCapacityReward,
    backend=LightSimBackend(),
    chronics_class=MultifolderWithCache,
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
)

nm_env_train, nm_env_val, nm_env_test = env.train_val_split_random(
    add_for_test="test", pct_val=20.0, pct_test=10.0
)

for name in ["train", "val", "test"]:
    grid2op.make(dataset=env_name + "_" + name).generate_classes()

# and now you can use the training set only to train your agent:
print(f"The name of the training environment is \\{nm_env_train}\\")
print(f"The name of the validation environment is \\{nm_env_val}\\")
print(f"The name of the test environment is \\{nm_env_test}\\")
