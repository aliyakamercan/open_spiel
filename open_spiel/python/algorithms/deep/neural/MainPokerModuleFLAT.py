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
        self.N_ROUNDS = self.game.num_rounds()
        self.device = device

        self._embedding_size = self.game.information_state_tensor_size()
        # Current Player | Current Round | History | Cards
        self._info_size = self.N_SEATS + self.N_ROUNDS
        self._history_size = 4 * self.N_SEATS * self.game.num_rounds()
        self._board_and_hole_size = (self.game.num_hole_cards() + self.game.total_board_cards())

        self._history_start = self._embedding_size - self._board_and_hole_size - self._history_size

        self.card_emb = CardEmbedding(game=game, dim=mpm_args.other_units, device=device)

        self.dropout = nn.Dropout(p=mpm_args.dropout)
        if self.args.normalize:
            self.norm = LayerNorm(mpm_args.other_units)

        self.cards_fc_1 = nn.Linear(in_features=self.card_emb.out_size, out_features=mpm_args.other_units)
        self.cards_fc_2 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)
        self.cards_fc_3 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)

        self.history_1 = nn.Linear(in_features=self._history_size * 2, out_features=mpm_args.other_units)
        self.history_2 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)

        self.comb_1 = nn.Linear(in_features=self._info_size + mpm_args.other_units * 2,
                                out_features=mpm_args.other_units)
        self.comb_2 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)
        self.comb_3 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)

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

        info_o = info_state[:, :self._info_size]
        hist_o = info_state[:, self._history_start:self._history_start+self._history_size]
        card_o = self.card_emb(info_state=info_state)

        if self.args.dropout > 0:
            A = lambda x: self.dropout(F.relu(x))
        else:
            A = lambda x: F.relu(x)

        card_o = A(self.cards_fc_1(card_o))
        card_o = A(self.cards_fc_2(card_o))
        card_o = A(self.cards_fc_3(card_o))

        ## bet
        bet_size = hist_o.clamp(0, 1e6)
        bet_occurred = hist_o.ge(0)
        bet_feats = torch.cat([bet_size, bet_occurred.float()], dim=-1)

        hist_o = A(self.history_1(bet_feats))
        hist_o = A(self.history_2(hist_o) + hist_o)

        y = A(self.comb_1(torch.cat([info_o, hist_o, card_o], dim=-1)))
        y = A(self.comb_2(y) + y)
        y = A(self.comb_3(y) + y)

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
