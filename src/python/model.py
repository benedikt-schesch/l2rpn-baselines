import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import copy

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_space, action_space, hidden_dim):
        super(DeterministicPolicy, self).__init__()
        self.noise = torch.Tensor(action_space["redispatch"].shape[0]).to("cuda")
        self.graph_net = GraphNet(obs_space, action_space, hidden_dim, 1)

    def forward(self, state):
        mean, _ = self.graph_net(state)
        return mean

    def sample(self, state):
        mean = self.forward(state)
        # noise = self.noise.normal_(0., std=0.1)
        # noise = noise.clamp(-0.25, 0.25)
        action = mean
        return action, torch.tensor(0.), mean
    
from typing import Tuple
from torch import nn
import torch
from ray.rllib.models.torch.misc import normc_initializer
from torch.distributions import MultivariateNormal
from torch_geometric.data import HeteroData
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import FastRGCNConv, SAGEConv, GCNConv, EdgeConv
import torch.nn.functional as F


class GraphNet(nn.Module):
    def __init__(self, obs_space, action_space, embed_dim, out_dim):
        super().__init__()
        self.n_dim = action_space["redispatch"].shape[0]  # type: ignore
        self.embed_dim = embed_dim
        self.obs_space = obs_space

        self.node_embeder = nn.ModuleDict()
        for node_type in obs_space["node_features"]:
            self.node_embeder[node_type] = nn.Linear(
                obs_space["node_features"][node_type].shape[1], embed_dim
            )
        self.conv1 = EdgeConv(
            nn=nn.Linear(2 * self.embed_dim, self.embed_dim),
            aggr="mean",
        )
        # self.conv2 = SAGEConv(
        #     in_channels=self.embed_dim,
        #     out_channels=self.embed_dim,
        #     aggr="mean",
        # )
        self.act = nn.ReLU()
        self.final_layer = nn.Linear(2 * self.embed_dim, out_dim)
        self.val_layer = nn.Linear(2 * 6 * self.embed_dim, 1)
        normc_initializer(0.001)(self.final_layer.weight)
        normc_initializer(0.001)(self.val_layer.weight)

    def forward(self, input: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        for node_type in self.obs_space["node_features"]:
            input[node_type].x = self.node_embeder[node_type](
                input[node_type].x.float()
            )
        input_homogeneous = input.to_homogeneous()
        skip_connection = input_homogeneous.x[input_homogeneous.node_type == 3]
        input_homogeneous.x = self.act(
            self.conv1(
                input_homogeneous.x,
                input_homogeneous.edge_index,
            ),
        )

        # input_homogeneous.x = self.act(
        #     self.conv2(
        #         input_homogeneous.x,
        #         input_homogeneous.edge_index,
        #     ),
        # )
        result = input_homogeneous.x[input_homogeneous.node_type == 3]
        result = torch.cat([result, skip_connection], dim=1)
        value = result.reshape(input.num_graphs, -1)
        value = self.val_layer(value)
        result = self.final_layer(result)
        result = result.reshape(input.num_graphs, -1)
        return result, value


class QGraphNetwork(nn.Module):
    def __init__(self, obs_space, action_space):
        nn.Module.__init__(self)
        self.n_dim = action_space["redispatch"].shape[0]  # type: ignore
        self.embed_dim = 16

        self.original_space = copy.deepcopy(obs_space)
        self.action_space = copy.deepcopy(action_space)
        low_obs = np.concatenate([self.original_space["node_features"]["gen"].low,
                                self.action_space["redispatch"].low.reshape((-1,1))], axis=1)
        high_obs = np.concatenate([self.original_space["node_features"]["gen"].high,
                                self.action_space["redispatch"].high.reshape((-1,1))], axis=1)
        self.original_space["node_features"]["gen"] = gymnasium.spaces.Box(
            low=low_obs,
            high=high_obs,
            shape=low_obs.shape,
            dtype=np.float32,
        )
        self.action_space = action_space
        self.qf1 = GraphNet(self.original_space, action_space, self.embed_dim, 1)
        self.qf2 = GraphNet(self.original_space, action_space, self.embed_dim, 1)

    def forward(self, input: HeteroData, action: torch.Tensor):
        # Attach the action to the input
        input["gen"].x = torch.cat([input["gen"].x, action.reshape((-1,1))], dim=1)
        qf1 = self.qf1(input.clone())[1]
        qf2= self.qf2(input.clone())[1]
        return qf1, qf2


def special_init(module):
    is_last_linear_layer = True
    for m in reversed(list(module.modules())):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)
            if is_last_linear_layer:
                normc_initializer(0.001)(m.weight)
                is_last_linear_layer = False
            else:
                normc_initializer(1.0)(m.weight)

