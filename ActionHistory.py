from typing import List
import copy

from Action import Action

class ActionHistory:
    def __init__(self, history: List[Action], action_space_size:int):
        self.history = copy.copy(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> int:
        return (len(self.history) % 2) + 1
