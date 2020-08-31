from abc import ABC, abstractmethod
from typing import List

from Action import Action
from Environment import Environment
from Node import Node
from ActionHistory import ActionHistory

class Game(ABC):
    def __init__(self, action_space_size: int, discount: float):
        self.env = None 
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount
    
    @abstractmethod
    def terminal(self) -> bool:
        pass

    @abstractmethod
    def legal_actions(self) -> List[Action]:
        pass

    @abstractmethod
    def apply_action(self, action: Action):
        reward = self.env.step() # must extract reward from environment or network?
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_stats(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())
    
    @abstractmethod
    def make_image(self, state_index: int):
        pass

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int,
                    to_play: int):
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount ** td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount ** i

            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else:
                last_reward = 0

            if current_index < len(self.root_values):
                targets.append((value, last_reward, self.child_visits[current_index]))
            else:
                targets.append((0, last_reward, []))
        return targets
    
    @abstractmethod
    def to_play(self) -> int:
        return 0

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)
