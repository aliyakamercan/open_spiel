import torch

from torch.optim import lr_scheduler

from open_spiel.python.algorithms.deep.wrappers import NetWrapperBase, NetWrapperArgsBase
from open_spiel.python.algorithms.deep.neural.DuelingQNet import DuelingQNet
from open_spiel.python.algorithms.deep import utils


class AdvWrapper(NetWrapperBase):

    def __init__(self, game, adv_training_args, device, online):
        super().__init__(
            net=DuelingQNet(game=game, q_args=adv_training_args.adv_net_args, device=device),
            args=adv_training_args,
            device=device
        )
        self._game = game
        self._online = online

    def get_advantages(self, info_state, legal_action_mask):
        self._net.eval()
        with torch.no_grad():
            return self._net(info_state=info_state, legal_action_masks=legal_action_mask)

    def _mini_batch_loop(self, buffer):
        batch_pub_obs, \
            batch_legal_action_masks, \
            batch_adv, \
            batch_loss_weight, \
            = buffer.sample(device=self.device, batch_size=self._args.batch_size)
        self._net.train()

        self._optim.zero_grad()
        # [batch_size, n_actions]
        adv_pred = self._net(info_state=batch_pub_obs,
                             legal_action_masks=batch_legal_action_masks)

        # Compute and print loss
        loss = self._criterion(adv_pred,
                               batch_adv,
                               batch_loss_weight.unsqueeze(-1).expand_as(batch_adv))

        # Zero gradients, perform a backward pass, and update the weights.

        loss.backward()

        if self._args.grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_norm_(parameters=self._net.parameters(), max_norm=self._args.grad_norm_clipping)

        self._optim.step()

        return loss.item()

    def reset(self):
        if self._args.init_adv_model == "last":
            self._net.zero_grad()
            if not self._online:
                self._optim, self._lr_scheduler = self._get_new_optim()
        elif self._args.init_adv_model == "random":
            self._net = DuelingQNet(game=self._game, q_args=self._args.adv_net_args, device=self.device)
            self._optim, self._lr_scheduler = self._get_new_optim()
        else:
            raise ValueError(self._args.init_adv_model)

    def _get_new_optim(self):
        opt = utils.str_to_optim_cls(self._args.optim_str)(self._net.parameters(), lr=self._args.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=opt,
                                                   threshold=0.001,
                                                   factor=0.5,
                                                   patience=self._args.lr_patience,
                                                   min_lr=0.00002)
        return opt, scheduler


class AdvTrainingArgs(NetWrapperArgsBase):

    def __init__(self,
                 adv_net_args,
                 n_batches_adv_training=1000,
                 batch_size=4096,
                 n_mini_batches_per_update=1,
                 optim_str="adam",
                 loss_str="weighted_mse",
                 lr=0.001,
                 grad_norm_clipping=10.0,
                 device_training="cpu",
                 max_buffer_size=2e6,
                 lr_patience=100,
                 init_adv_model="last",
                 ):
        super().__init__(batch_size=batch_size, n_mini_batches_per_update=n_mini_batches_per_update,
                         optim_str=optim_str, loss_str=loss_str, lr=lr, grad_norm_clipping=grad_norm_clipping,
                         device_training=device_training)
        self.adv_net_args = adv_net_args
        self.n_batches_adv_training = n_batches_adv_training
        self.lr_patience = lr_patience
        self.max_buffer_size = int(max_buffer_size)
        self.init_adv_model = init_adv_model
