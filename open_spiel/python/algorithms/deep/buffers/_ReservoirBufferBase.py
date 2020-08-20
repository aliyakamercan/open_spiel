import numpy as np
import torch


class ReservoirBufferBase:

    def __init__(self, max_size, game, nn_type, iter_weighting_exponent):
        self._game = game
        self.device = torch.device("cpu")

        self._max_size = max_size
        self._nn_type = nn_type
        self.size = 0
        self.n_entries_seen = 0

        if nn_type == "recurrent":
            self._info_state_buffer = np.empty(shape=(max_size,), dtype=object)
        elif nn_type == "feedforward":
            self._info_state_buffer = torch.zeros((max_size, self._game.information_state_tensor_size()),
                                                  dtype=torch.float32,
                                                  device=self.device)
        else:
            raise ValueError(nn_type)

        self._legal_action_mask_buffer = torch.zeros((max_size, game.num_distinct_actions(),),
                                                     dtype=torch.float32, device=self.device)
        self._iteration_buffer = torch.zeros((max_size,), dtype=torch.float32, device=self.device)
        self._iter_weighting_exponent = iter_weighting_exponent

        self._last_cfr_iteration_seen = None

    def add(self, **kwargs):
        """
        Dont forget to n_entries_seen+=1 !!
        """
        raise NotImplementedError

    def sample(self, batch_size, device):
        raise NotImplementedError

    def _should_add(self):
        return np.random.random() < (float(self._max_size) / float(self.n_entries_seen))

    def _np_to_torch(self, arr):
        return torch.from_numpy(np.copy(arr)).to(self.device)

    def _random_idx(self):
        return np.random.randint(low=0, high=self._max_size)

    def state_dict(self):
        return {
            "max_size": self._max_size,
            "nn_type": self._nn_type,
            "size": self.size,
            "n_entries_seen": self.n_entries_seen,
            "iter_weighting_exponent": self._iter_weighting_exponent,

            "info_state_buffer": self._info_state_buffer,
            "legal_action_mask_buffer": self._legal_action_mask_buffer,
            "iteration_buffer": self._iteration_buffer,
        }

    def load_state_dict(self, state):
        assert self._max_size == state["max_size"]
        assert self._nn_type == state["nn_type"]

        self.size = state["size"]
        self.n_entries_seen = state["n_entries_seen"]
        self._iter_weighting_exponent = state["iter_weighting_exponent"]

        if self._nn_type == "recurrent":
            self._info_state_buffer = state["info_state_buffer"]
            self._legal_action_mask_buffer = state["legal_action_mask_buffer"]
            self._iteration_buffer = state["iteration_buffer"]

        elif self._nn_type == "feedforward":
            self._info_state_buffer = state["info_state_buffer"].to(self.device)
            self._legal_action_mask_buffer = state["legal_action_mask_buffer"].to(self.device)
            self._iteration_buffer = state["iteration_buffer"].to(self.device)

        else:
            raise ValueError(self._nn_type)
