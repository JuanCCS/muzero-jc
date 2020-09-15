from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import os

from Network import MuZeroNetwork

if TYPE_CHECKING:
    from MuZeroConfig import MuZeroConfig


class SharedStorage:
    def __init__(self, config: MuZeroConfig):
        self.config = config
        self._networks = {}

    def latest_network(self) -> MuZeroNetwork:
        if self._networks:
            return self._networks[max(self.networks.keys())]
        else:
            return MuZeroNetwork(self.config)

    def save_network(self, step: int, network: MuZeroNetwork):
        self._networks[step] = network
        torch.save(network, os.path.join(self.config.path, 
                                         str(step).zfill(2) + '.pkl'))
