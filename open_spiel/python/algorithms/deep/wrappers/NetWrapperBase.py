# Copyright (c) 2019 Eric Steinberger


import torch


def str_to_loss_cls(loss_str):
    if loss_str.lower() == "mse":
        return torch.nn.MSELoss()

    elif loss_str.lower() == "weighted_mse":
        return lambda y, trgt, w: torch.mean(w * ((y - trgt) ** 2))

    elif loss_str.lower() == "ce":
        return torch.nn.CrossEntropyLoss()

    elif loss_str.lower() == "smoothl1":
        return torch.nn.SmoothL1Loss()

    else:
        raise ValueError(loss_str)


class NetWrapperBase:

    def __init__(self, net, args, device):
        self._args = args
        self.device = device

        self._criterion = str_to_loss_cls(self._args.loss_str)
        self.loss_last_batch = None

        self._net = net
        self._optim, self._lr_scheduler = self._get_new_optim()

    def train_one_loop(self, buffer):
        if buffer.size < self._args.batch_size:
            return

        loss = 0.0
        for micro_batch_id in range(self._args.n_mini_batches_per_update):
            loss += self._mini_batch_loop(buffer=buffer)

        self.loss_last_batch = loss / self._args.n_mini_batches_per_update
        return self.loss_last_batch

    def _apply_grads(self, list_of_grads, optimizer, grad_norm_clip=None):
        # optimizer.zero_grad()

        if grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(parameters=self._net.parameters(), max_norm=grad_norm_clip)

        optimizer.step()

    def apply_grads(self, grads):
        self._apply_grads(grads, self._optim, self._args.grad_norm_clipping)

    def step_scheduler(self, loss):
        if loss:
            self._lr_scheduler.step(loss)

    def _mini_batch_loop(self, buffer):
        raise NotImplementedError

    def load_net_state_dict(self, state_dict):
        self._net.load_state_dict(state_dict)

    def net_state_dict(self):
        return self._net.state_dict()

    def train(self):
        self._net.train()

    def eval(self):
        self._net.eval()

    def net(self):
        return self._net

    def state_dict(self):
        """ Override, if necessary"""
        return self.net_state_dict()

    def load_state_dict(self, state):
        """ Override, if necessary"""
        self.load_net_state_dict(state)

    def _get_new_optim(self):
        raise NotImplementedError


class NetWrapperArgsBase:

    def __init__(self,
                 batch_size,
                 n_mini_batches_per_update,
                 optim_str,
                 loss_str,
                 lr,
                 grad_norm_clipping,
                 device_training
                 ):
        assert isinstance(device_training, str), "Please pass a string (either 'cpu' or 'cuda')!"
        self.batch_size = batch_size
        self.n_mini_batches_per_update = n_mini_batches_per_update
        self.optim_str = optim_str
        self.loss_str = loss_str
        self.lr = lr
        self.grad_norm_clipping = grad_norm_clipping
        self.device_training = torch.device(device_training)
