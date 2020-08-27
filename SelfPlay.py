class MuZeroConfig:
    pass

class SharedStorage:
    def latest_network(self):
        pass

class ReplayBuffer:
    pass

class Game:
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


class Mcts:
    def run(self, config: MuZeroConfig, root: Node, 
            action_history: ActionHistory, network: Network):
        """run.

        Parameters
        ----------
        config : MuZeroConfig
            config
        root : Node
            root
        action_history : ActionHistory
            action_history
        network : Network
            network
        """
        min_max_stats = MinMaxStats(config.known_bounds)

        for _ in range(config.num_simulations):
            history = action_history.clone()
            node = root
            search_path = [node]

            while node.expanded():
                action, node = self.select_child(config, node, min_max_stats)
                history.add_action(action)
                search_path.append(node)

            parent = search_path[-2]
            network_output = network.recurrent_inference(parent.hidden_state,
                                                         history.last_action())
            self.expand_node(node, history.to_play(), history.action_space(), network_output)
            self.backpropagate(search_path, network_output.value, history.to_play(),
                          config.discount, min_max_stats)

    def expand_node(self, node: Node, to_play: Player, legal_actions: List[Action], network_output: NetworkOutput):
        node.to_play = to_play
        node.hidden_state = netowrk_output.hidden_state
        node.reward = network_output.reward
        policy = {a: math.exp(nework_output.policy_logits[a]) for a in legal_actions}
        policy_sum = sum(policy.values())
        for action, p in policy.items():
            node.children[action] = Node(p/policy_sum)


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
   
