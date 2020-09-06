# Copyright (c) 2019 Eric Steinberger


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .LayerNorm import LayerNorm
from .CardEmbedding import CardEmbedding


class MainPokerModuleFLAT(nn.Module):
    """
    Feeds parts of the observation through different fc layers before the RNN

    Structure (each branch merge is a concat):

    Table & Player state --> FC -> RE -> FCS -> RE ----------------------------.
    Board Cards ---> FC -> RE --> cat -> FC -> RE -> FCS -> RE -> FC -> RE --> cat --> FC -> RE -> FCS-> RE -> Normalize
    Private Cards -> FC -> RE -'


    where FCS refers to FC+Skip and RE refers to ReLU
    """

    def __init__(self,
                 game,
                 device,
                 mpm_args,
                 ):
        super().__init__()
        self.args = mpm_args
        self.game = game

        self.N_SEATS = self.game.num_players()
        self.device = device

        self._embedding_size = self.game.information_state_tensor_size()
        self._board_and_hole_size = (self.game.num_hole_cards() + self.game.total_board_cards()) * 3
        self.card_emb = CardEmbedding(game=game, dim=mpm_args.other_units, device=device)

        self.dropout = nn.Dropout(p=mpm_args.dropout)
        if self.args.normalize:
            self.norm = LayerNorm(mpm_args.other_units)

        self.layer_1 = nn.Linear(in_features=self.card_emb.out_size + self._embedding_size - self._board_and_hole_size,
                                 out_features=mpm_args.other_units)
        self.layer_2 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)
        self.layer_3 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)

        self.to(device)

    @property
    def output_units(self):
        return self.args.other_units

    def forward(self, info_state):
        """
        1. do list -> padded
        2. feed through pre-processing fc layers
        3. PackedSequence (sort, pack)
        4. rnn
        5. unpack (unpack re-sort)
        6. cut output to only last entry in sequence
        """

        if isinstance(info_state, list):
            info_state = torch.from_numpy(np.array(info_state)).to(self.device, torch.float32)

        hist_o = info_state[:, :-self._board_and_hole_size]
        card_o = self.card_emb(info_state=info_state)

        if self.args.dropout > 0:
            A = lambda x: self.dropout(F.relu(x))
        else:
            A = lambda x: F.relu(x)

        y = torch.cat([hist_o, card_o], dim=-1)
        y = A(self.layer_1(y))
        y = A(self.layer_2(y) + y)
        y = A(self.layer_3(y) + y)

        if self.args.normalize:
            y = self.norm(y)

        return y

class MPMArgsFLAT:

    def __init__(self,
                 other_units=64,
                 normalize=True,
                 dropout=0.0,
                 ):
        self.other_units = other_units
        self.normalize = normalize
        self.dropout = dropout

    def get_mpm_cls(self):
        return MainPokerModuleFLAT
