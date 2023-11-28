from typing import Any, Union
from torch_geometric.data import HeteroData
import grid2op
from grid2op.Reward import LinesCapacityReward
from lightsim2grid import LightSimBackend
from ray.rllib.utils.spaces.repeated import Repeated
from gymnasium import Env
import numpy as np
from gymnasium import spaces
from collections import defaultdict
import torch
import networkx as nx
from collections import OrderedDict
from grid2op.Environment import Environment
from torch_geometric.transforms import ToUndirected, AddSelfLoops
from grid2op.Observation import BaseObservation
from grid2op.Chronics import GridStateFromFileWithForecastsWithoutMaintenance
from grid2op.Action import DontAct
from grid2op.Opponent import BaseOpponent, NeverAttackBudget
from grid2op.Chronics import MultifolderWithCache


class Grid2OpEnvRedispatchCurtail(Env):
    def __init__(self, env_name: str = "l2rpn_case14_sandbox") -> None:
        super().__init__()
        self.env_name = env_name
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
        )
        self.n_gen = self.grid2op_env.n_gen

        # Observation space normalization factors
        self.gen_pmax = torch.tensor(self.grid2op_env.observation_space.gen_pmax)
        self.gen_pmin = torch.tensor(self.grid2op_env.observation_space.gen_pmin)
        assert torch.all(self.gen_pmax >= self.gen_pmin) and torch.all(
            self.gen_pmin >= 0
        )  # type: ignore

        # Observation space observation
        self.observation_space: ObservationSpace = ObservationSpace(self.grid2op_env)
        self.elements_graph = self.grid2op_env.reset().get_elements_graph()
        self.elements_graph_pyg = self.observation_space.grid2op_to_pyg(
            self.elements_graph
        )

        # Action space
        self.action_space = spaces.Dict()
        self.action_space["redispatch"] = spaces.Box(
            low=-1, high=1, shape=(self.n_gen,), dtype=np.float32
        )

        # Action space normalization factor
        self.action_norm_factor = np.maximum(
            self.grid2op_env.observation_space.gen_max_ramp_up,  # type: ignore
            -self.grid2op_env.observation_space.gen_max_ramp_down,  # type: ignore
        )

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        action = action * self.action_norm_factor
        # action["redispatch"] = action["redispatch"] * self.action_norm_factor
        return action

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        set_id: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
            self.grid2op_env.seed(seed)
        if set_id is not None:
            self.grid2op_env.set_id(set_id)
        obs = self.grid2op_env.reset()
        self.grid2op_obs = obs
        elements_graph_pyg = self.observation_space.grid2op_to_pyg(
            obs.get_elements_graph()
        )
        return elements_graph_pyg, {}

    def step(self, action: Union[None, torch.Tensor]):
        grid2op_action = self.grid2op_env.action_space()
        if action is not None:
            action = self.denormalize_action(action)
            grid2op_action.redispatch = action

        obs, reward, done, info = self.grid2op_env.step(grid2op_action)
        self.grid2op_obs = obs
        elements_graph_pyg = self.observation_space.grid2op_to_pyg(
            obs.get_elements_graph()
        )
        return elements_graph_pyg, reward, done, False, info

    def get_grid2op_obs(self) -> BaseObservation:
        return self.grid2op_obs

    def render(self, mode="rgb_array"):
        return self.grid2op_env.render(mode)

    def get_grid2op_env(self) -> Environment:
        return self.grid2op_env


class ObservationSpace(spaces.Dict):
    def __init__(self, env: Environment):
        self.add_self_loops = AddSelfLoops()
        self.to_undirected = ToUndirected()
        graph = self.grid2op_to_pyg(
            env.reset().get_elements_graph(), reverse_edges=True, self_edges=True
        )

        dic = OrderedDict()
        dic["node_features"] = spaces.Dict()
        for node_type, _ in graph.node_items():
            dic["node_features"][node_type] = node_observation_space[node_type](
                len(graph[node_type].x)
            )
        self.n_gen = env.n_gen

        # Add edges
        dic["edge_list"] = spaces.Dict()
        dic["edge_features"] = spaces.Dict()
        for edge_type, _ in graph.edge_items():
            num_node_type_source = len(graph[edge_type[0]].x)
            num_node_type_target = len(graph[edge_type[2]].x)
            dic["edge_list"][edge_type] = Repeated(  # type: ignore
                spaces.MultiDiscrete([num_node_type_source, num_node_type_target]),
                max_len=num_node_type_source * num_node_type_target,
            )
            if edge_type[1].startswith("rev_"):
                dic["edge_features"][edge_type] = edge_observation_space[
                    (edge_type[2], edge_type[1][4:], edge_type[0])
                ](  # type: ignore
                    len(graph[edge_type[0]].x)
                )
            else:
                dic["edge_features"][edge_type] = edge_observation_space[edge_type](  # type: ignore
                    len(graph[edge_type[0]].x)
                )

        spaces.Dict.__init__(self, dic)

    def grid2op_to_pyg(
        self,
        elements_graph: nx.DiGraph,
        reverse_edges: bool = True,
        self_edges: bool = True,
    ) -> HeteroData:
        # Initialize HeteroData
        graph = HeteroData()
        id_map = defaultdict(lambda: defaultdict(int))
        nodes = defaultdict(list)
        edge_types = defaultdict(list)
        edge_features = defaultdict(list)

        # Node processing
        for new_id, (old_id, features) in enumerate(elements_graph.nodes(data=True)):
            node_type = features["type"]
            id_map[node_type][old_id] = len(id_map[node_type])
            nodes[node_type].append(
                torch.tensor(
                    [features.get(field, 0) for field in node_data_fields[node_type]]
                )
            )

        # Populate HeteroData nodes
        for key, vals in nodes.items():
            graph[key].x = torch.stack(vals)

        # Initialize dictionaries to hold edge features
        edge_features = defaultdict(list)

        # Edge processing
        for src, dst, attr in elements_graph.edges(data=True):
            src_type, dst_type = (
                elements_graph.nodes[src]["type"],
                elements_graph.nodes[dst]["type"],
            )
            edge_type = attr["type"]

            if edge_type not in edge_data_fields:
                raise Exception(f"Edge type {edge_type} not supported")

            edge_types[(src_type, edge_type, dst_type)].append(
                (id_map[src_type][src], id_map[dst_type][dst])
            )
            if edge_type != "bus_to_substation":
                edge_features[edge_type].append(
                    torch.ones(len(edge_data_fields[edge_type]))
                )
            else:
                edge_features[edge_type].append(
                    torch.tensor(
                        [attr.get(field, 0) for field in edge_data_fields[edge_type]]
                    )
                )

        # Populate HeteroData edges and edge features
        for key, vals in edge_types.items():
            graph[key].edge_index = (
                torch.tensor(vals, dtype=torch.long).t().contiguous()
            )
            if len(edge_data_fields[key[1]]) > 0:
                graph[key].edge_attr = torch.stack(edge_features[key[1]])
        if reverse_edges:
            graph = self.to_undirected(graph)

        if self_edges:
            graph = self.add_self_loops(graph)

        return graph


node_observation_space = OrderedDict(
    {
        "substation": lambda n_lements: spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_lements, 1), dtype=np.float32
        ),
        "bus": lambda n_lements: spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_lements, 3), dtype=np.float32
        ),
        "load": lambda n_lements: spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_lements, 1), dtype=np.float32
        ),
        "gen": lambda n_lements: spaces.Box(
            low=-1, high=1, shape=(n_lements, 9), dtype=np.float32
        ),
        "line": lambda n_lements: spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_lements, 2), dtype=np.float32
        ),
        "shunt": lambda n_lements: spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_lements, 1), dtype=np.float32
        ),
    }
)

node_data_fields = OrderedDict(
    {
        "substation": [
            "id"
        ],  # All fields: {'id': 1, 'type': 'substation', 'name': 'sub_1', 'cooldown': 0}
        "bus": [
            "v",
            "theta",
            "id",
        ],  # All fields: {'id': 2, 'global_id': 2, 'local_id': 1, 'type': 'bus', 'connected': True, 'v': 142.1, 'theta': -4.190119}
        "load": [
            "id"
        ],  # {'id': 8, 'type': 'load', 'name': 'load_11_8', 'connected': True}
        "gen": [
            "id",
            "target_dispatch",
            "actual_dispath",
            "gen_p_before_curtal",
            "curtalment_mw",
            "curtailement",
            "curtailment_limit",
            "gen_margin_up",
            "gen_margin_down",
            # "difference_dispatch",
            # "target_dispatch",
            # "actual_dispatch",
        ],  # {'id': 3, 'type': 'gen', 'name': 'gen_5_3', 'connected': True, 'target_dispatch': 0.0, 'actual_dispatch': 0.0, 'gen_p_before_curtail': 0.0, 'curtailment_mw': 0.0, 'curtailment': 0.0, 'curtailment_limit': 1.0, 'gen_margin_up': 0.0, 'gen_margin_down': 0.0}
        "line": [
            "id",
            "rho",
        ],  # {'id': 1, 'type': 'line', 'name': '0_4_1', 'rho': 0.36042336, 'connected': True, 'timestep_overflow': 0, 'time_before_cooldown_line': 0, 'time_next_maintenance': -1, 'duration_next_maintenance': 0}
        "shunt": [
            "id"
        ],  # {'id': 0, 'type': 'shunt', 'name': 'shunt_8_0', 'connected': True}
    }
)

edge_observation_space = OrderedDict(
    {
        ("bus", "bus_to_substation", "substation"): lambda n_lements: spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_lements, 1), dtype=np.float32
        ),
        ("load", "load_to_bus", "bus"): lambda n_lements: spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_lements, 5), dtype=np.float32
        ),
        ("gen", "gen_to_bus", "bus"): lambda n_lements: spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_lements, 5), dtype=np.float32
        ),
        ("line", "line_to_bus", "bus"): lambda n_lements: spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_lements, 6), dtype=np.float32
        ),
        ("shunt", "shunt_to_bus", "bus"): lambda n_lements: spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_lements, 4), dtype=np.float32
        ),
    }
)

edge_data_fields = OrderedDict(
    {
        "bus_to_substation": {
            "constant",
        },
        "load_to_bus": {
            "id",
            "p",
            "q",
            "v",
            "theta",
        },  # {'id': 0, 'type': 'load_to_bus', 'p': 21.9, 'q': 15.4, 'v': 142.1, 'theta': -1.4930121}
        "gen_to_bus": {
            "id",
            "p",
            "q",
            "v",
            "theta",
        },  # {'id': 0, 'type': 'gen_to_bus', 'p': -81.4, 'q': -19.496038, 'v': 142.1, 'theta': -1.4930121}
        "line_to_bus": {
            "id",
            "p",
            "q",
            "v",
            "theta",
            "a",
        },  # {'id': 0, 'type': 'line_to_bus', 'p': 42.346096, 'q': -16.060501, 'v': 142.1, 'a': 184.01027, 'side': 'or', 'theta': 0.0}
        "shunt_to_bus": {
            "id",
            "p",
            "v",
            "q",
        },  # {'id': 0, 'type': 'shunt_to_bus', 'p': -6.938894e-16, 'q': -21.208096, 'v': 21.13022}
    }
)
