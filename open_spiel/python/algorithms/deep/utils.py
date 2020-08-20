import time


import torch
import numpy as np


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


def get_legal_action_mask_torch(n_actions, legal_actions_list, device, dtype=torch.uint8):
    """
    Args:
        legal_actions_list (list):  List of legal actions as integers, where 0 is always FOLD, 1 is CHECK/CALL.
                                    2 is BET/RAISE for continuous PokerEnvs, and for DiscretePokerEnv subclasses,
                                    numbers greater than 1 are all the raise sizes.

        device (torch.device):      device the mask shall be put on

        dtype:                      dtype the mask shall have

    Returns:
        torch.Tensor:               a many-hot representation of the list of legal actions.
    """
    idxs = torch.LongTensor(legal_actions_list, device=device)
    mask = torch.zeros((n_actions), device=device, dtype=dtype)
    mask[idxs] = 1
    return mask


def batch_get_legal_action_mask_torch(n_actions, legal_actions_lists, device, dtype=torch.uint8):
    """

    Args:
        legal_actions_lists (list): List of lists. Each of the 2nd level lists contains legal actions as integers,
                                    where 0 is always FOLD, 1 is CHECK/CALL. 2 is BET/RAISE for continuous
                                    PokerEnvs, and for DiscretePokerEnv subclasses, numbers greater than 1 are all
                                    the raise sizes.

        device (torch.device):      device the mask shall be put on

        dtype:                      dtype the mask shall have

    Returns:
        torch.Tensor:               a many-hot representation of the list of legal actions.

    """
    assert isinstance(legal_actions_lists[0], list), "need list of lists of legal actions (as ints)!"

    mask = torch.zeros((len(legal_actions_lists), n_actions,), device=device, dtype=dtype)
    for i, legal_action_list in enumerate(legal_actions_lists):
        mask[i, torch.LongTensor(legal_action_list, device=device)] = 1
    return mask


def str_to_optim_cls(optim_string):
    if optim_string.lower() == "sgd":
        return torch.optim.SGD

    elif optim_string.lower() == "adam":
        def fn(parameters, lr):
            return torch.optim.Adam(parameters, lr=lr)

        return fn

    elif optim_string.lower() == "rms":
        def fn(parameters, lr):
            return torch.optim.RMSprop(parameters, lr=lr)

        return fn

    elif optim_string.lower() == "sgdmom":
        def fn(parameters, lr):
            return torch.optim.SGD(parameters, lr=lr, momentum=0.9, nesterov=True)

        return fn

    else:
        raise ValueError(optim_string)


def get_legal_action_mask_np(n_actions, legal_actions_list, dtype=np.uint8):
    """

    Args:
        legal_actions_list (list):  List of legal actions as integers, where 0 is always FOLD, 1 is CHECK/CALL.
                                    2 is BET/RAISE for continuous PokerEnvs, and for DiscretePokerEnv subclasses,
                                    numbers greater than 1 are all the raise sizes.

        dtype:                      dtype the mask shall have

    Returns:
        np.ndarray:                 a many-hot representation of the list of legal actions.

    """
    mask = np.zeros(shape=n_actions, dtype=dtype)
    mask[legal_actions_list] = 1
    return mask
