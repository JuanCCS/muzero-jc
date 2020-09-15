from __future__ import annotations
from typing import TYPE_CHECKING

import pickle
import numpy as np
import uuid
import datetime as dt
import os
from pathlib import Path

from Game import Game

if TYPE_CHECKING:
    from MuZeroConfig import MuZeroConfig

class ReplayBuffer:
    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []
        self.counter = 0
        self.folder_name = os.path.join(config.path, 
                                        dt.datetime.now().strftime('%Y%m%d'))

        # Hacky way to use selfplay games from the nn trainer for now
        Path(self.folder_name).mkdir(parents=True, exist_ok=True)
        for game_file in os.listdir(self.folder_name):
            if game_file.endswith('.pkl'):
                file_path = os.path.join(self.folder_name, game_file)
                game = pickle.load(open(file_path, 'rb'))
                self.buffer.append(game)
        self.buffer = self.buffer[-10:] if len(self.buffer) > 10 else self.buffer


    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)
        self.counter += 1
        pickle.dump(game, 
                    open(os.path.join(self.folder_name, 
                                      f'{self.counter}.pkl'), 'wb'))

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.make_image(i.index), g.history[i.index:i.index + num_unroll_steps],
                 g.make_target(i.index, num_unroll_steps, td_steps, g.to_play()))
                for (g,i) in game_pos]

    def sample_game(self) -> Game:
        return np.random.choice(self.buffer)

    def sample_position(self, game) -> int:
        return np.random.choice(game.history)

