import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import List

from MuZeroConfig import MuZeroConfig
from SharedStorage import SharedStorage
from ReplayBuffer import ReplayBuffer
from Network import MuZeroNetwork
from Game import Game


class Trainer:
    def train_network(self, config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
        network = MuZeroNetwork(config)
        optimizer = optim.SGD(network.parameters(), config.lr_init, config.momentum, config.weight_decay)
        for i in range(config.training_steps):
            if i % config.checkpoint_interval == 0:
                storage.save_network(i, network)
            batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
            self.update_weights(optimizer, network, batch, config.weight_decay)
        storage.save_network(config.training_steps, network)

    def scale_gradient(self, tensor, scale):
        return tensor * scale + tensor.detach() + (1-scale)

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
                if target_policy:
                    neg_policy = -torch.Tensor(target_policy)
                    log_softmax = nn.LogSoftmax(dim=1)
                    log_sm = log_softmax(policy_logits)

                    l = (
                        self.scalar_loss(value, target_value)  + 
                        self.scalar_loss(reward, target_reward) + 
                        torch.sum(neg_policy * log_sm)
                    )

                    loss += self.scale_gradient(l, gradient_scale)
                
        for _, weights in network.state_dict().items():
            print(loss)
            print(weight_decay)
            mse_loss = torch.sum(torch.pow(weights,2))/2
            print(weight_decay * mse_loss)
            loss += weight_decay * mse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)


    def scalar_loss(self, prediction, target) -> float:
        return -1


if __name__ == '__main__':
    from games.TicTacToe import TicTacToeConfig
    from SelfPlay import SelfPlay
    self_play = SelfPlay()
    config = TicTacToeConfig()
    replay_buffer = ReplayBuffer(config)
    storage = SharedStorage(config)
    trainer = Trainer()
    trainer.train_network(config=config, storage=storage, replay_buffer=replay_buffer)

