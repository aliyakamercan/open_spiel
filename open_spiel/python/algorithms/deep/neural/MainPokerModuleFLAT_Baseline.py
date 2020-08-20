# Copyright (c) 2019 Eric Steinberger


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class MainPokerModuleFLAT_Baseline(nn.Module):

    def __init__(self,
                 game,
                 device,
                 mpm_args,
                 ):
        super().__init__()

        self._args = mpm_args
        self._game = game

        self._device = device
        self.dropout = nn.Dropout(p=mpm_args.dropout)

        self._embedding_size = self._game.information_state_tensor_size()

        self.layer_1 = nn.Linear(in_features=self._embedding_size, out_features=mpm_args.dim)
        self.layer_2 = nn.Linear(in_features=mpm_args.dim, out_features=mpm_args.dim)
        self.layer_3 = nn.Linear(in_features=mpm_args.dim, out_features=mpm_args.dim)

        if self._args.normalize:
            self.norm = LayerNorm(mpm_args.dim)

        self.to(device)
        # print("n parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad))

    @property
    def output_units(self):
        return self._args.dim

    @property
    def device(self):
        return self._device

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
            info_state = torch.from_numpy(np.array(info_state)).to(self._device, torch.float32)

        # """""""""""""""""""""""
        # Network
        # """""""""""""""""""""""
        if self._args.dropout > 0:
            A = lambda x: self.dropout(F.relu(x))
        else:
            A = lambda x: F.relu(x)

        y = A(self.layer_1(info_state))
        y = A(self.layer_2(y) + y)
        y = A(self.layer_3(y) + y)

        # """""""""""""""""""""""
        # Normalize last layer
        # """""""""""""""""""""""
        if self._args.normalize:
            y = self.norm(y)

        return y


class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-7):
        super().__init__()
        self.a = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class MPMArgsFLAT_Baseline:

    def __init__(self,
                 dim=128,
                 dropout=0.0,
                 normalize=True,
                 ):
        self.dim = dim
        self.dropout = dropout
        self.normalize = normalize

    def get_mpm_cls(self):
        return MainPokerModuleFLAT_Baseline
