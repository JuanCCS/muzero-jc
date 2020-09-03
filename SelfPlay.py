from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ReplayBuffer import ReplayBuffer
from Mcts import Mcts
from Node import Node
from Game import Game
from ActionHistory import ActionHistory
from Action import Action
from SharedStorage import SharedStorage
 
if TYPE_CHECKING:
    from MuZeroConfig import MuZeroConfig
    from Network import MuZeroNetwork


class SelfPlay:

    def __init__(self):
        self.mcts = Mcts()

    def run_selfplay(self, config: MuZeroConfig ,storage: SharedStorage, 
                     replay_buffer: ReplayBuffer):
        """run_selfplay.

        Parameters
        ----------
        config : MuZeroConfig
            config
        storage : SharedStorage
            storage
        replay_buffer : ReplayBuffer
            replay_buffer
        """
        while True:
            network = storage.latest_network()
            game = self.play_game(config, network)
            replay_buffer.save_game(game)

    def play_game(self, config: MuZeroConfig, network: MuZeroNetwork) -> Game:
        """play_game.

        Parameters
        ----------
        config : MuZeroConfig
            config
        network : MuZeroNetwork
            network

        Returns
        -------
        Game

        """
        game = config.new_game()

        while not game.terminal() and len(game.history) < config.max_moves:
            root = Node()
            current_observation = game.make_image(-1)
            self.mcts.expand_node(root, game.to_play(), game.legal_actions(),
                                  network.initial_inference(current_observation))
            self.add_exploration_noise(config, root)
            self.mcts.run(config, root, game.action_history(), network)
            action = self.select_action(config, len(game.history), root, network)
            game.apply_action(Action(action))
            game.store_search_stats(root)

        return game

    def select_action(self, config: MuZeroConfig, num_moves: int, node: Node, network: MuZeroNetwork):
        """select_action.

        Parameters
        ----------
        config : MuZeroConfig
            config
        num_moves : int
            num_moves
        node : Node
            node
        network : MuZeroNetwork
            network
        """
        visit_counts = [
            (child.visit_count, action) for action, child in node.children.items()
        ]
        t = config.visit_softmax_temperature_fn(
            num_moves
        ) # other games may need a more complex function
        _, action = self.softmax_sample(visit_counts, t)
        return action

    def softmax_sample(self, distribution, temperature: float):
        if temperature == 0:
            temperature = 1
        distribution = np.array([d[0] for d in distribution])**(1/temperature)
        p_sum = distribution.sum()
        sample_temp = distribution/p_sum
        return 0, np.argmax(np.random.multinomial(1, sample_temp, 1))
    
    def add_exploration_noise(self, config: MuZeroConfig, node: Node):
        actions = list(node.children.keys()) 
        noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
        frac = config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1-frac) + n * frac


if __name__ == '__main__':
    from games.TicTacToe import TicTacToeConfig
    self_play = SelfPlay()
    config = TicTacToeConfig()
    replay_buffer = ReplayBuffer(config)
    shared_storage = SharedStorage(config)
    self_play.run_selfplay(
        config,
        shared_storage,
        replay_buffer
    )

