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

    def expand_node(self, node: Node, to_play: Player, 
                    legal_actions: List[Action], network_output: NetworkOutput):
        """expand_node.

        Parameters
        ----------
        node : Node
            node
        to_play : Player
            to_play
        legal_actions : List[Action]
            legal_actions
        network_output : NetworkOutput
            network_output
        """
        node.to_play = to_play
        node.hidden_state = netowrk_output.hidden_state
        node.reward = network_output.reward
        policy = {a: math.exp(nework_output.policy_logits[a]) for a in legal_actions}
        policy_sum = sum(policy.values())
        for action, p in policy.items():
            node.children[action] = Node(p/policy_sum)

    def backpropagate(self, search_path: List[Node], value: float,
                      to_play: Player, discount: float, min_max_stats: MinMaxStats):
        """backpropagate.

        Parameters
        ----------
        search_path : List[Node]
            search_path
        value : float
            value
        to_play : Player
            to_play
        discount : float
            discount
        min_max_stats : MinMaxStats
            min_max_stats
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            min_max_stats.update(node.value())
            value = node.reward + discount * value
