import numpy as np
import torch
from torch import nn


NUM_CARDS = 52


class CardEmbedding(nn.Module):

    def __init__(self, game, dim, device):
        super().__init__()
        self._game = game
        self._device = device

        self.board_start = -(self._game.num_hole_cards() + self._game.total_board_cards())
        self.hole_start = -self._game.num_hole_cards()

        self._n_card_types = game.num_rounds()
        self.card_embs = nn.ModuleList([
            _CardGroupEmb(game=game, dim=dim)
            for _ in range(self._n_card_types)
        ])

        self._dim = dim

        self.to(device)

    @property
    def out_size(self):
        return self._n_card_types * self._dim

    @property
    def device(self):
        return self._device

    def forward(self, info_state):

        priv_cards = info_state[:, self.hole_start:].round().to(torch.long)
        board_cards = info_state[:, self.board_start:self.hole_start].round().to(torch.long)

        # TODO: board cards

        card_batches = [(
            priv_cards[:, 0:self._game.num_hole_cards()] // 4,
            priv_cards[:, 0:self._game.num_hole_cards()] % 4,
            priv_cards[:, 0:self._game.num_hole_cards()]
        )]

        off = 0
        for round_ in range(1, self._n_card_types):
            n = self._game.board_cards_for_round(round_)
            if n > 0:
                round_cards = board_cards[:, off:off+n]
                card_batches.append(
                     # rank, suit, card
                     (round_cards[:, off:off + n] // 4,
                      round_cards[:, off:off + n] % 4,
                      round_cards[:, off:off + n])
                )
                off += n

        card_o = []
        for emb, (ranks, suits, cards) in zip(self.card_embs, card_batches):
            card_o.append(emb(ranks=ranks, suits=suits, cards=cards))

        return torch.cat(card_o, dim=1)


class _CardGroupEmb(nn.Module):

    def __init__(self, game, dim):
        super().__init__()
        self._game = game
        self._dim = dim
        self.rank = nn.Embedding(13, dim)
        self.suit = nn.Embedding(4, dim)
        self.card = nn.Embedding(NUM_CARDS, dim)

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
