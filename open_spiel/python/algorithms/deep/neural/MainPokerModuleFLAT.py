# Copyright (c) 2019 Eric Steinberger


import numpy as np
import torch
import torch.nn as nn


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
        self._relu = nn.ReLU(inplace=False)

        self.final_fc_1 = nn.Linear(in_features=self._embedding_size, out_features=mpm_args.other_units)
        self.final_fc_2 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)

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

        y = torch.cat((info_state,), dim=-1)

        final = self._relu(self.final_fc_1(y))
        final = self._relu(self.final_fc_2(final) + final)

        # Normalize last layer
        if self.args.normalize:
            final = final - final.mean(dim=-1).unsqueeze(-1)
            final = final / final.std(dim=-1).unsqueeze(-1)

        return final


class MPMArgsFLAT:

    def __init__(self,
                 other_units=64,
                 normalize=True,
                 ):
        self.other_units = other_units
        self.normalize = normalize

    def get_mpm_cls(self):
        return MainPokerModuleFLAT
