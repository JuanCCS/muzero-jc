class MuZeroConfig:
    pass

class SharedStorage:
    def latest_network(self):
        pass

class ReplayBuffer:
    pass

class Network:
    pass

class Node:
    pass

class MinMaxStats:
    def __init__(known_bounds):
        pass

class ActionHistory:
    pass


class SelfPlay:

    def __init__():
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
            game = self.play_game()
            replay_buffer.save_game(game)

    def play_game(self, config: MuZeroConfig, network: Network) -> Game:
        """play_game.

        Parameters
        ----------
        config : MuZeroConfig
            config
        network : Network
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
            game.apply(action)
            game.store_search_stats(root)

        return game

    def select_action(self, config: MuZeroConfig, num_moves: int, node: Node, network: Network):
        """select_action.

        Parameters
        ----------
        config : MuZeroConfig
            config
        num_moves : int
            num_moves
        node : Node
            node
        network : Network
            network
        """
        visit_counts = [
            (child.visit_count, action) for action, child in node.children.items()
        ]
        t = config.visit_softmax_temperatur_fn(
            num_moves=num_moves, training_steps=network.training_steps()
        )
        _, action = self.softmax_sample(visit_counts, t)
        return action
   
