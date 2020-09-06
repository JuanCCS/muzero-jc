import torch.nn as nn
import torch.optim as optim

from typing import List

from MuZeroConfig import MuZeroConfig
from SharedStorage import SharedStorage
from ReplayBuffer import ReplayBuffer
from Network import MuZeroNetwork
from Game import Game


class Trainer:
    def train_network(self, config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
        # network: get new network?
        # learning_rate = define learning rate
        # optimizer = define Momentum Optimizer
        for i in range(config.training_steps):
            if i % config.checkpoint_interval == 0:
                storage.save_network(i, network)
            batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
            self.update_weights(optimizer, network, batch, config.weight_decay)
        storage.save_network(config.training_steps, network)

    def scale_gradient(self, tensor, scale):
        return tensor * scale + torch_stop_gradient(tensor) + (1-scale)

    def update_weights(self, optimizer: optim.Optimizer, 
                       network: MuZeroNetwork, batch, weight_decay: float):
        loss = 0
        for image, actions, targets in batch:
            value, reward, policy_logits, hidden_state = network.initial_inference(image)
            predictions = [(1.0, value, reward, policy_logits)]

            for action in actions:
                value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, action)
                predictions.append((1.0 / len(actions), value, reward, policy_logits))
                hidden_state = self.scale_gradient(hidden_state, 0.5)

            for prediction, target in zip(predictions, targets):
                gradient_scale, value, reward, policy_logits = prediction
                target_value, target_reward, target_policy = target

                l = (
                    self.scalar_loss(value, target_value)  + 
                    self.scalar_loss(reward, target_reward) + 
                    torch_softmax_cross_entropy_with_logits(
                        logits=policy_logits, labels=target_policy))

                loss += self.scale_gradient(l, gradient_scale)
                
        for weights in network.get_weights():
            loss += weight_decay * torch_l2_loss(weights)

        optimizer.step(loss)

    def scalar_loss(self, prediction, target) -> float:
        return -1

