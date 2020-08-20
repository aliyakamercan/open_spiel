import torch

from open_spiel.python.algorithms.deep.buffers._ReservoirBufferBase import ReservoirBufferBase as _ResBufBase


class AdvReservoirBuffer(_ResBufBase):

    def __init__(self, nn_type, max_size, game, iter_weighting_exponent):
        super().__init__(max_size=max_size, game=game, nn_type=nn_type,
                         iter_weighting_exponent=iter_weighting_exponent)

        self._adv_buffer = torch.zeros((max_size, game.num_distinct_actions()), dtype=torch.float32, device=self.device)

    def add(self, info_state, legal_action_mask, adv, iteration):
        if self.size < self._max_size:
            self._add(idx=self.size,
                      info_state=info_state,
                      legal_action_mask=legal_action_mask,
                      adv=adv,
                      iteration=iteration)
            self.size += 1

        elif self._should_add():
            self._add(idx=self._random_idx(),
                      info_state=info_state,
                      legal_action_mask=legal_action_mask,
                      adv=adv,
                      iteration=iteration)

        self.n_entries_seen += 1

    def sample(self, batch_size, device):
        indices = torch.randint(0, self.size, (batch_size,), dtype=torch.long, device=self.device)

        if self._nn_type == "recurrent":
            obses = self._info_state_buffer[indices.cpu().numpy()]
        elif self._nn_type == "feedforward":
            obses = self._info_state_buffer[indices].to(device)
        else:
            raise NotImplementedError

        return \
            obses, \
            self._legal_action_mask_buffer[indices].to(device), \
            self._adv_buffer[indices].to(device), \
            self._iteration_buffer[indices].to(device) / self._last_cfr_iteration_seen

    def _add(self, idx, info_state, legal_action_mask, adv, iteration):
        if self._nn_type == "feedforward":
            info_state = torch.from_numpy(info_state)

        self._info_state_buffer[idx] = info_state
        self._legal_action_mask_buffer[idx] = legal_action_mask
        self._adv_buffer[idx] = adv

        self._iteration_buffer[idx] = float(iteration) ** self._iter_weighting_exponent

        self._last_cfr_iteration_seen = iteration

    def state_dict(self):
        return {
            "base": super().state_dict(),
            "adv": self._adv_buffer,
        }

    def load_state_dict(self, state):
        super().load_state_dict(state["base"])
        self._adv_buffer = state["adv"]
