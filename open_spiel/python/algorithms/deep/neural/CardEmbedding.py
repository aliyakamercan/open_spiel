import numpy as np
import torch
from torch import nn

# TODO: change these to be read from game
NUM_ROUNDS = 1
NUM_RANKS = 3
NUM_SUITS = 1
NUM_PLAYERS = 2
NUM_CARDS = 1


class CardEmbedding(nn.Module):

    def __init__(self, game, dim, device):
        super().__init__()
        self._game = game
        self._device = device

        # number of rounds TODO: fix
        n_card_types = NUM_ROUNDS
        self.card_embs = nn.ModuleList([
            _CardGroupEmb(game=game, dim=dim)
            for _ in range(n_card_types)
        ])
        self._n_card_types = n_card_types
        self._dim = dim

        self.to(device)

    @property
    def out_size(self):
        return self._n_card_types * self._dim

    @property
    def device(self):
        return self._device

    def forward(self, info_state):

        priv_cards = info_state[:, NUM_PLAYERS:NUM_CARDS+NUM_PLAYERS].round().to(torch.long)

        # TODO: board cards

        card_batches = [(priv_cards // 4, priv_cards % 4, priv_cards)]

        off = 0
        # for round_ in self._game.rules.ALL_ROUNDS_LIST:
        #     n = self._game.lut_holder.DICT_LUT_CARDS_DEALT_IN_TRANSITION_TO[round_]
        #     if n > 0:
        #         card_batches.append(
        #             # rank, suit, card
        #             (board[:, off:off + 3 * n:3],
        #              board[:, off + 1:off + 1 + 3 * n:3],
        #              board[:, off + 2:off + 2 + 3 * n:3],)
        #         )
        #         off += n

        card_o = []
        for emb, (ranks, suits, cards) in zip(self.card_embs, card_batches):
            card_o.append(emb(ranks=ranks, suits=suits, cards=cards))

        return torch.cat(card_o, dim=1)


class _CardGroupEmb(nn.Module):

    def __init__(self, game, dim):
        super().__init__()
        self._game = game
        self._dim = dim
        self.rank = nn.Embedding(NUM_RANKS, dim)
        self.suit = nn.Embedding(NUM_SUITS, dim)
        self.card = nn.Embedding(52, dim)

    def forward(self, ranks, suits, cards):

        bs, n_cards = cards.shape

        r = ranks.view(-1)
        valid_r = r.ge(0).unsqueeze(1).to(torch.float32)
        r = r.clamp(min=0)
        embs = self.rank(r) * valid_r


        s = suits.view(-1)
        c = cards.view(-1)
        valid_s = s.ge(0).unsqueeze(1).to(torch.float32)
        valid_c = c.ge(0).unsqueeze(1).to(torch.float32)
        s = s.clamp(min=0)
        c = c.clamp(min=0)

        embs += self.card(c) * valid_c + self.suit(s) * valid_s

        return embs.view(bs, n_cards, -1).sum(1)
