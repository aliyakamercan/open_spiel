import torch
import torch.nn.functional as nnf
from torch.optim import lr_scheduler


from open_spiel.python.algorithms.deep import utils
from open_spiel.python.algorithms.deep.neural.AvrgStrategyNet import AvrgStrategyNet
from open_spiel.python.algorithms.deep.wrappers import NetWrapperBase, NetWrapperArgsBase


class AvrgWrapper(NetWrapperBase):

    def __init__(self, game, avrg_training_args, device, online):
        super().__init__(
            net=AvrgStrategyNet(avrg_net_args=avrg_training_args.avrg_net_args, game=game, device=device),
            args=avrg_training_args,
            device=device
        )
        self._game = game
        self._online = online

    def get_a_probs(self, info_state, legal_actions_lists):
        """
        Args:
            info_state (list):             list of np arrays of shape [np.arr([history_len, n_features]), ...)
            legal_actions_lists (list:  list of lists. each 2nd level lists contains ints representing legal actions
        """
        with torch.no_grad():
            masks = utils.batch_get_legal_action_mask_torch(n_actions=self._game.num_distinct_actions(),
                                                            legal_actions_lists=legal_actions_lists,
                                                            device=self.device)
            masks = masks.view(1, -1)
            return self.get_a_probs2(info_state=info_state, legal_action_masks=masks)

    def get_a_probs2(self, info_state, legal_action_masks):
        with torch.no_grad():
            pred = self._net(info_state=info_state,
                             legal_action_masks=legal_action_masks)

            return nnf.softmax(pred, dim=-1).cpu().numpy()

    def _mini_batch_loop(self, buffer):
        batch_pub_obs, \
            batch_legal_action_masks, \
            batch_a_probs, \
            batch_loss_weight, \
            = buffer.sample(device=self.device, batch_size=self._args.batch_size)
        self._net.train()

        # [batch_size, n_actions]
        strat_pred = self._net(info_state=batch_pub_obs,
                               legal_action_masks=batch_legal_action_masks)
        strat_pred = nnf.softmax(strat_pred, dim=-1)

        # Compute and print loss
        loss = self._criterion(strat_pred,
                               batch_a_probs,
                               batch_loss_weight.unsqueeze(-1).expand_as(batch_a_probs))

        # Zero gradients, perform a backward pass, and update the weights.
        self._optim.zero_grad()
        loss.backward()

        if self._args.grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_norm_(parameters=self._net.parameters(), max_norm=self._args.grad_norm_clipping)

        self._optim.step()

        return loss.item()

    def reset(self):
        if self._args.init_avrg_model == "last":
            self._net.zero_grad()
            if not self._online:
                self._optim, self._lr_scheduler = self._get_new_optim()
        elif self._args.init_avrg_model == "random":
            self._net = AvrgStrategyNet(game=self._game, avrg_net_args=self._args.avrg_net_args, device=self.device)
            self._optim, self._lr_scheduler = self._get_new_optim()
        else:
            raise ValueError(self._args.init_adv_model)

    def _get_new_optim(self):
        opt = utils.str_to_optim_cls(self._args.optim_str)(self._net.parameters(), lr=self._args.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=opt,
                                                   threshold=0.0001,
                                                   factor=0.5,
                                                   patience=self._args.lr_patience,
                                                   min_lr=0.00002)
        return opt, scheduler


class AvrgTrainingArgs(NetWrapperArgsBase):

    def __init__(self,
                 avrg_net_args,
                 n_batches_avrg_training=1000,
                 batch_size=4096,
                 n_mini_batches_per_update=1,
                 optim_str="adam",
                 loss_str="weighted_mse",
                 lr=0.001,
                 grad_norm_clipping=10.0,
                 device_training="cpu",
                 max_buffer_size=2e6,
                 lr_patience=100,
                 init_avrg_model="random",
                 ):
        super().__init__(batch_size=batch_size,
                         n_mini_batches_per_update=n_mini_batches_per_update,
                         optim_str=optim_str,
                         loss_str=loss_str,
                         lr=lr,
                         grad_norm_clipping=grad_norm_clipping,
                         device_training=device_training)

        self.avrg_net_args = avrg_net_args
        self.n_batches_avrg_training = n_batches_avrg_training
        self.max_buffer_size = int(max_buffer_size)
        self.lr_patience = lr_patience
        self.init_avrg_model = init_avrg_model
