from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Action import Action

if TYPE_CHECKING:
    from MuZeroConfig import MuZeroConfig


num_filters = 2
num_blocks = 8

# Credit: https://github.com/werner-duvaud/muzero-general/blob/master/models.py

@dataclass
class NetworkOutput:
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]

class NetworkTypes(Enum):
    fully_connected = "fully_connected"
    residual = "residual"

class MuZeroNetwork:
    def __new__(cls, config):
        if config.network_type == NetworkTypes.fully_connected:
            return MuZeroFullyConnectedNetwork(config)
        elif config.network_type == NetworkTypes.residual:
            pass
        else:
            raise NotImplementedError(
                "Wrong Network Type"
            )

def normalize_encoded_state(encoded_state):
    min_next_encoded_state = encoded_state.min(1, keepdim=True)[0]
    max_next_encoded_state = encoded_state.max(1, keepdim=True)[0]
    scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
    scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
    next_encoded_state_normalized = (
        encoded_state - min_next_encoded_state
    ) / scale_next_encoded_state
    return next_encoded_state_normalized


def mlp(input_size, 
        layer_sizes, 
        output_size, 
        output_activation=nn.Identity,
        activation=nn.ELU) -> nn.Sequential:
    """mlp.
    Builds a Multi-Layer Perceptron
    based on the given parameters

    Parameters
    ----------
    input_size :
        input_size
    layer_sizes :
        layer_sizes
    output_size :
        output_size
    output_activation :
        output_activation
    activation :
        activation
    """
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i, _ in enumerate(sizes[:-1]):
        act = activation if i < len(sizes) - 2 else output_activation
        layers.extend([torch.nn.Linear(sizes[i], sizes[i + 1]), act()])
    return torch.nn.Sequential(*layers)


def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x


def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits



class MuZeroFullyConnectedNetwork(nn.Module):
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.action_space_size = config.action_space_size
        self.full_support_size = 2 * config.support_size + 1
        self.support_size = config.support_size

        obs = config.observation_shape
        representation_input_size = obs[0] * obs[1] * obs[2] * (config.move_history + 1)\
            + config.move_history * obs[1] * obs[2]

        self.representation_network = torch.nn.DataParallel(mlp(
            representation_input_size,
            config.fc_representation_layers,
            config.encoding_size
        ))

        self.dynamics_network = torch.nn.DataParallel(mlp(
            config.encoding_size + self.action_space_size,
            config.fc_dynamics_layers,
            config.encoding_size
        ))

        self.dynamics_reward_network = torch.nn.DataParallel(mlp(
            config.encoding_size,
            config.fc_reward_layers, 
            self.full_support_size
        ))

        self.prediction_policy_network = torch.nn.DataParallel(mlp(
            config.encoding_size, config.fc_policy_layers, self.action_space_size
        ))

        self.prediction_value_network = torch.nn.DataParallel(mlp(
            config.encoding_size, config.fc_value_layers, self.full_support_size
        ))

    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def representation(self, observation: np.ndarray):
        observation = torch.Tensor(observation).float().unsqueeze(0)
        encoded_state = self.representation_network(
            observation.view(observation.shape[0], -1)
        )
        return normalize_encoded_state(encoded_state)

    def dynamics(self, encoded_state, action):
        action = torch.Tensor([[action]])
        action_one_hot = (
            torch.zeros((
                action.shape[0], self.action_space_size
            )).float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0) 
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_network(x)
        reward = self.dynamics_reward_network(next_encoded_state)
        
        return normalize_encoded_state(encoded_state), reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        reward = torch.log((
            torch.zeros(1, self.full_support_size)
            .scatter(1, torch.tensor([[
                self.full_support_size // 2]]).long(), 1.0)
            .repeat(len(observation), 1) 
        ))

        return NetworkOutput(value=support_to_scalar(value, self.support_size), 
                             reward=support_to_scalar(reward, self.support_size), 
                             policy_logits=policy_logits, 
                             hidden_state=encoded_state)

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return NetworkOutput(value=support_to_scalar(value, self.support_size),
                             reward=support_to_scalar(reward, self.support_size),
                             policy_logits=policy_logits,
                             hidden_state=next_encoded_state)


