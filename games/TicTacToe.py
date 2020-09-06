import numpy as np
import os
from typing import List
from itertools import cycle
from more_itertools import all_equal
from math import floor

from Game import Game
from Environment import Environment
from Action import Action
from MuZeroConfig import MuZeroConfig

class TicTacToeConfig(MuZeroConfig):
    def __init__(self,
                 action_space_size: int = 9,
                 max_moves: int = 9,
                 discount: float = 1.0,
                 dirichlet_alpha: float = 0.3,
                 num_simulations: int = 800,
                 batch_size: int = 2048,
                 td_steps: int = 5,
                 num_actors:int = 3000,
                 lr_init: float = 0.1,
                 lr_decay_steps: float = 4e5):

        super(TicTacToeConfig, self).__init__(action_space_size,
                                              max_moves,
                                              discount,
                                              dirichlet_alpha,
                                              num_simulations,
                                              batch_size,
                                              td_steps,
                                              num_actors,
                                              lr_init,
                                              lr_decay_steps,
                                              lambda x: 1.0)
        self.observation_shape = (3, 3, 3)
        self.path = os.path.join(self.path, 'tic_tac_toe')

    def new_game(self):
        return TicTacToeGame(self.action_space_size, self.discount)


class TicTacToeGame(Game):
    def __init__(self, action_space_size: int, discount: float):
        super(TicTacToeGame, self).__init__(action_space_size, discount)
        self.env = TicTacToe()

    def make_image(self, state_index:int) -> np.ndarray:
        """make_image.
        Produces the observation that is fed to the Neural Nets

        Parameters
        ----------
        state_index : int
            state_index

        Returns
        -------
        np.ndarray
            The output is a 3x9 Matrix where the rows stand for:
                1. Player 1's pieces
                2. Player 2's pieces
                3. Current turn (1 for player 1 and 0 for player 2)

        """
        board_shape = (3,3)
        obs_board = np.zeros((3, 3, 3))
        # player_one
        obs_board[0] = np.reshape([1 if x == 1 else 0 for x in self.env.board], 
                                  board_shape) 
        #player two
        obs_board[1] = np.reshape([1 if x == 2 else 0 for x in self.env.board],
                                  board_shape)
        #turn
        obs_board[2] = np.ones(board_shape) if self.env.player == 1 else np.zeros(board_shape)
        return obs_board 

    def legal_actions(self) -> List[Action]:
        empty_squares = np.where(self.env.board == 0)[0]
        return [Action(x) for x in empty_squares]

    def terminal(self) -> bool:
        winners=((0,1,2),(3,4,5),(6,7,8),(0,3,6),
                 (1,4,7),(2,5,8),(0,4,8),(2,4,6))
        for w in winners:
            if self.env.board[w[0]] and all_equal(self.env.board[list(w)]):
                return True
        return False

    def apply_action(self, action: Action):
        self.env.step(action)
        self.history.append(action)

    def to_play(self) -> int:
        return self.env.player

    def render(self) -> str:
        """render.
        Classical reperesentation of the board
        with Xs and Os

        Parameters
        ----------

        Returns
        -------
        str
            Console printable string
        """
        str_board = ""
        for i in range(5):
            if i % 2 == 0:
                str_board += "| {}  " * 3
            else:
                str_board += " --- " * 3
            str_board += "\n"

            char_map = (' ', 'x', 'o')
            chars = map(lambda x: char_map[x], 
                        self.env.board.astype(int).tolist())

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
        self.player = next(self.players) 

