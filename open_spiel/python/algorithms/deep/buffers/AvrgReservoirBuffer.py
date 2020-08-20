import torch

from open_spiel.python.algorithms.deep.buffers._ReservoirBufferBase import ReservoirBufferBase as _ResBufBase
from open_spiel.python.algorithms.deep import utils


class AvrgReservoirBuffer(_ResBufBase):
    """
    Reservoir buffer to store state+action samples for the average strategy network
    """

    def __init__(self, nn_type, max_size, game, iter_weighting_exponent):
        super().__init__(max_size=max_size, game=game, nn_type=nn_type,
                         iter_weighting_exponent=iter_weighting_exponent)

        self._a_probs_buffer = torch.zeros((max_size, game.num_distinct_actions()), dtype=torch.float32,
                                           device=self.device)

    def add(self, info_state, legal_actions_list, a_probs, iteration):
        if self.size < self._max_size:
            self._add(idx=self.size,
                      info_state=info_state,
                      legal_action_mask=self._get_mask(legal_actions_list),
                      action_probs=a_probs,
                      iteration=iteration)
            self.size += 1

        elif self._should_add():
            self._add(idx=self._random_idx(),
                      info_state=info_state,
                      legal_action_mask=self._get_mask(legal_actions_list),
                      action_probs=a_probs,
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
            self._a_probs_buffer[indices].to(device), \
            self._iteration_buffer[indices].to(device) / self._last_cfr_iteration_seen

    def _add(self, idx, info_state, legal_action_mask, action_probs, iteration):
        if self._nn_type == "feedforward":
            info_state = torch.from_numpy(info_state)

        self._info_state_buffer[idx] = info_state
        self._legal_action_mask_buffer[idx] = legal_action_mask
        self._a_probs_buffer[idx] = action_probs

        # In "https://arxiv.org/pdf/1811.00164.pdf", Brown et al. weight by floor((t+1)/2), but we assume that
        # this is due to incrementation happening for every alternating update. We count one iteration as an
        # update for both plyrs.
        self._iteration_buffer[idx] = float(iteration) ** self._iter_weighting_exponent
        self._last_cfr_iteration_seen = iteration

    def _get_mask(self, legal_actions_list):
        return utils.get_legal_action_mask_torch(n_actions=self._game.num_distinct_actions(),
                                                 legal_actions_list=legal_actions_list,
                                                 device=self.device, dtype=torch.float32)

    def state_dict(self):
        return {
            "base": super().state_dict(),
            "a_probs": self._a_probs_buffer,
        }

    def load_state_dict(self, state):
        super().load_state_dict(state["base"])
        self._a_probs_buffer = state["a_probs"]
