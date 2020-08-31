import numpy as np
from typing import List
from itertools import cycle
from more_itertools import all_equal
from math import floor

from Game import Game
from Environment import Environment
from Action import Action

class TicTacToeGame(Game):
    def __init__(self, action_space_size: int, discount: float):
        super(TicTacToeGame, self).__init__(self, action_space_size, discount)
        self.environment = TicTacToe()

    def make_image(self, state_index:int) -> np.ndarray:
        obs_board = np.zeros((3,9))
        # player_one
        obs_board[0] = [1 if x == 1 else 0 for x in self.env.board] 
        obs_board[1] = [1 if x == 2 else 0 for x in self.env.board] 
        obs_board[2] = np.ones(9) if self.env.player == 1 else np.zeros(9)
        return self.env.board

    def legal_actions(self) -> List[Action]:
        empty_squares = list(filter(None, self.env.board))
        return [Action(x) for x in empty_squares]

    def terminal(self) -> bool:
        winners=((0,1,2),(3,4,5),(6,7,8),(0,3,6),
                 (1,4,7),(2,5,8),(0,4,8),(2,4,6))
        for w in winners:
            if self.env.board[w[0]] and all_equal(self.env.board[w]):
                return True
        return False

    def apply(self, action: Action):
        self.env.step(action)
        self.env.player = next(self.env.players) 

    def render(self) -> str:
        """render.
        This function is in charge of drawing the board. 
        The output you can directly print to the console.

        Parameters
        ----------

        Returns
        -------
        str
            A string representation of the board you can directly print

        """
        str_board = ""
        for i in range(5):
            if i % 2 == 0:
                str_board += "| {}  " * 3
            else:
                str_board += " --- " * 3
            str_board += "\n"

            char_map = (' ', 'x', 'o')
            chars = map(lambda x: char_map[x], board)

        return str_board.format(*chars)
    
class TicTacToe(Environment):
    def __init__(self):
        self.start_game()

    def start_game(self):
        self.board = np.zeros(9)
        self.players = cycle([1, 2])
        self.player = next(self.players) 

    def step(self, action: Action):
        self.board[action.index] = self.player

