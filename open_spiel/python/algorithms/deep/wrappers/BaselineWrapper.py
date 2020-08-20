# Copyright (c) Eric Steinberger 2020

import torch

from open_spiel.python.algorithms.deep import utils
from open_spiel.python.algorithms.deep.neural.DuelingQNet import DuelingQNet
from open_spiel.python.algorithms.deep.wrappers import NetWrapperBase, NetWrapperArgsBase


class BaselineWrapper(NetWrapperBase):

    def __init__(self, game, baseline_args):
        super().__init__(
            net=DuelingQNet(game=game, q_args=baseline_args.q_net_args, device=baseline_args.device_training),
            args=baseline_args,
            device=baseline_args.device_training,
        )

        self._game = game
        self._batch_arranged = torch.arange(self._args.batch_size, dtype=torch.long, device=self.device)
        self._minus_e20 = torch.full((self._args.batch_size, self._game.num_distinct_actions(),),
                                     fill_value=-10e20,
                                     device=self.device,
                                     dtype=torch.float32,
                                     requires_grad=False)

    def get_b(self, info_states, legal_actions_lists, to_np=False):
        with torch.no_grad():
            masks = utils.batch_get_legal_action_mask_torch(n_actions=self._game.num_distinct_actions(),
                                                            legal_actions_lists=legal_actions_lists,
                                                            device=self.device, dtype=torch.float32)
            self.eval()
            q = self._net(info_state=info_states, legal_action_masks=masks)
            q *= masks

            if to_np:
                q = q.cpu().numpy()

            return q

    def _mini_batch_loop(self, buffer):
        batch_pub_obs_t, \
            batch_legal_action_mask_t, \
            batch_a_t, \
            batch_r_t, \
            batch_pub_obs_tp1, \
            batch_legal_action_mask_tp1, \
            batch_done, \
            batch_strat_tp1 = \
            buffer.sample(device=self.device, batch_size=self._args.batch_size)

        self._net.train()

        # [batch_size, n_actions]
        q1_t = self._net(info_state=batch_pub_obs_t,
                         legal_action_masks=batch_legal_action_mask_t.to(torch.float32))
        q1_tp1 = self._net(info_state=batch_pub_obs_tp1,
                           legal_action_masks=batch_legal_action_mask_tp1.to(torch.float32)).detach()

        # ______________________________________________ TD Learning _______________________________________________
        # [batch_size]
        q1_t_of_a_selected = q1_t[self._batch_arranged, batch_a_t]

        # only consider allowed actions for tp1
        q1_tp1 = torch.where(batch_legal_action_mask_tp1,
                             q1_tp1,
                             self._minus_e20)

        # [batch_size]
        q_tp1_of_atp1 = (q1_tp1 * batch_strat_tp1).sum(-1)
        q_tp1_of_atp1 *= (1.0 - batch_done)
        target = batch_r_t + q_tp1_of_atp1

        loss = self._criterion(q1_t_of_a_selected,
                               target)

        # Zero gradients, perform a backward pass, and update the weights.
        self._optim.zero_grad()
        loss.backward()

        if self._args.grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_norm_(parameters=self._net.parameters(), max_norm=self._args.grad_norm_clipping)

        self._optim.step()

        return loss.item()

    def _get_new_optim(self):
        opt = utils.str_to_optim_cls(self._args.optim_str)(self._net.parameters(),
                                                           lr=self._args.lr)
        return opt, None


class BaselineArgs(NetWrapperArgsBase):

    def __init__(self,
                 q_net_args,
                 max_buffer_size=2e5,
                 n_batches_per_iter_baseline=500,
                 batch_size=512,
                 optim_str="adam",
                 loss_str="mse",
                 lr=0.001,
                 grad_norm_clipping=1.0,
                 device_training="cpu",
                 ):
        super().__init__(batch_size=batch_size,
                         optim_str=optim_str,
                         loss_str=loss_str,
                         lr=lr,
                         n_mini_batches_per_update=1,
                         grad_norm_clipping=grad_norm_clipping,
                         device_training=device_training)
        self.q_net_args = q_net_args
        self.max_buffer_size = int(max_buffer_size)
        self.n_batches_per_iter_baseline = n_batches_per_iter_baseline
