from typing import Union, Tuple
import torch
from gymnasium import Env
from gymnasium import spaces
from torch_geometric.data import HeteroData
from .Grid2OpResdispatchCurtail import Grid2OpEnvRedispatchCurtail
from grid2op.Environment import Environment
from collections import OrderedDict


class Grid2OpEnvRedispatchCurtailFlattened(Env):
    def __init__(self, env_name: str = "l2rpn_case14_sandbox") -> None:
        super().__init__()
        # Initialize the Grid2OpEnvRedispatchCurtailFlattened environment
        self.base_env = Grid2OpEnvRedispatchCurtail(env_name)
        obs, info = self.base_env.reset()
        flat_features = self.flatten_features(obs)
        self.observation_space = spaces.Box(
            low=-float("inf"), high=float("inf"), shape=flat_features.shape
        )
        self.feature_dim = flat_features.shape[1]
        self.action_space = self.base_env.action_space

    def flatten_features(self, hetero_data: HeteroData) -> torch.Tensor:
        # Extract and flatten selected node features
        node_features = []
        for node_type in hetero_data.node_types:
            # Only take the fields specified in node_data_fields
            selected_fields = node_data_fields[node_type]
            if len(selected_fields) == 0:
                continue
            features = torch.stack(
                [
                    hetero_data[node_type].x[:, i]
                    for i, field in enumerate(selected_fields)
                ],
                dim=1,
            )
            node_features.append(features.flatten())

        flattened_node_features = torch.cat(node_features)
        # Define expected number of edges and default features for each edge type
        expected_edges = {
            edge_type: (expected_number_of_edges, default_feature_vector)
            for edge_type, expected_number_of_edges, default_feature_vector in [
                # Example: ("edge_type_name", 10, torch.zeros(feature_length)),
                # Add entries for each edge type with their expected number of edges and default features
            ]
        }

        # Extract and flatten all edge features, ensuring consistent number of edges
        edge_features = []
        for edge_type, (expected_count, default_features) in expected_edges.items():
            if edge_type in hetero_data.edge_types:
                current_edge_features = hetero_data[edge_type].edge_attr
                edge_count_difference = expected_count - current_edge_features.size(0)
                if edge_count_difference > 0:
                    # Append default features to match the expected count
                    padding = default_features.unsqueeze(0).repeat(
                        edge_count_difference, 1
                    )
                    current_edge_features = torch.cat(
                        [current_edge_features, padding], dim=0
                    )
                edge_features.append(current_edge_features.flatten())
            else:
                # Use default features for the entire edge type if it's completely missing
                edge_features.append(default_features.repeat(expected_count).flatten())

        flattened_edge_features = (
            torch.cat(edge_features) if edge_features else torch.tensor([])
        )

        # Combine node and edge features
        return torch.cat([flattened_node_features, flattened_edge_features]).unsqueeze(
            0
        )

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.base_env.denormalize_action(action)

    def reset(self, **kwargs) -> Tuple[torch.Tensor, dict]:
        hetero_data, info = self.base_env.reset(**kwargs)
        obs = self.flatten_features(hetero_data)
        assert obs.shape[1] == self.feature_dim
        return obs, info

    def step(
        self, action: Union[None, torch.Tensor]
    ) -> Tuple[torch.Tensor, float, bool, bool, dict]:
        hetero_data, reward, done, _, info = self.base_env.step(action)
        obs = self.flatten_features(hetero_data)
        if not done:
            assert obs.shape[1] == self.feature_dim
        self.prev = hetero_data
        return obs, reward, done, False, info

    def render(self, mode="rgb_array"):
        return self.base_env.render(mode)

    def get_grid2op_env(self) -> Environment:
        return self.base_env.get_grid2op_env()

    def get_grid2op_obs(self):
        return self.base_env.get_grid2op_obs()


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
    }
)
