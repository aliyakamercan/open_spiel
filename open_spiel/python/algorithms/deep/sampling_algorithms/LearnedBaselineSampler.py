# Copyright (c) Eric Steinberger 2020

import numpy as np
import torch

from open_spiel.python.algorithms.deep.sampling_algorithms._SamplerBase import SamplerBase
from open_spiel.python.algorithms.deep import utils


class LearnedBaselineSampler(SamplerBase):
    """
    How to get to next state:
        -   Each time ""traverser"" acts, a number of sub-trees are followed. For each sample, the remaining deck is
            reshuffled to ensure a random future.

        -   When any other player acts, 1 action is chosen w.r.t. their strategy.

        -   When the environment acts, 1 action is chosen according to its natural dynamics. Note that the PokerRL
            environment does this inherently, which is why there is no code for that in this class.


    When what is stored to where:
        -   At every time a player other than ""traverser"" acts, we store their action probability vector to their
            reservoir buffer.

        -   Approximate immediate regrets are stored to ""traverser""'s advantage buffer at every node at which they
            act.
    """

    def __init__(self,
                 game,
                 adv_buffers,
                 baseline_net,
                 baseline_buf,
                 eps=0.5,
                 avrg_buffers=None,
                 ):
        super().__init__(game=game, adv_buffers=adv_buffers, avrg_buffers=avrg_buffers)
        self._baseline_net = baseline_net
        self._baseline_buf = baseline_buf

        # self._reg_buf = None

        self._eps = eps
        self._actions_arranged = np.arange(self._game.num_distinct_actions())

        self.total_node_count_traversed = 0

    def generate(self, n_traversals, traverser, iteration_strats, cfr_iter, ):
        # self._reg_buf = [[] for _ in range(self._game.rules.N_CARDS_IN_DECK)]

        super().generate(n_traversals, traverser, iteration_strats, cfr_iter)
        # if traverser == 0:
        #     print("STD:  ", np.sum(np.array([np.array(x).std(axis=0) for x in self._reg_buf]), axis=0))
        #     print("Mean: ", np.sum(np.array([np.array(x).mean(axis=0) for x in self._reg_buf]), axis=0))

    def _traverser_act(self, current_state, traverser, trav_depth, iteration_strats, sample_reach,
                       cfr_iter):
        """
        Last state values are the average, not the sum of all samples of that state since we add
        v~(I) = * p(a) * |A(I)|. Since we sample multiple actions on each traverser node, we have to average over
        their returns like: v~(I) * Sum_a=0_N (v~(I|a) * p(a) * ||A(I)|| / N).
        """
        self.total_node_count_traversed += 1
        legal_actions_list = current_state.legal_actions()
        legal_action_mask = utils.get_legal_action_mask_torch(n_actions=self._game.num_distinct_actions(),
                                                              legal_actions_list=legal_actions_list,
                                                              device=self._adv_buffers[traverser].device,
                                                              dtype=torch.float32)
        info_state_t = current_state.information_state_tensor(traverser)
        info_state = np.array(info_state_t)

        # """""""""""""""""""""""""
        # Strategy
        # """""""""""""""""""""""""
        strat_i = iteration_strats[traverser].get_a_probs(
            info_state=[info_state],
            legal_actions_lists=[legal_actions_list],
            to_np=False
        )[0]

        # """""""""""""""""""""""""
        # Sample action
        # """""""""""""""""""""""""
        n_legal_actions = len(legal_actions_list)
        sample_strat = (1 - self._eps) * strat_i + self._eps * (legal_action_mask.cpu() / n_legal_actions)
        a = torch.multinomial(sample_strat.cpu(), num_samples=1).item()

        # Step
        next_state = current_state.child(a)

        while next_state.is_chance_node():
            action = np.random.choice([i[0] for i in next_state.chance_outcomes()])
            next_state = next_state.child(action)

        done = next_state.is_terminal()
        rew_for_all = next_state.returns()
        info_state_tp1 = next_state.information_state_tensor(traverser)
        legal_action_mask_tp1 = utils.get_legal_action_mask_torch(n_actions=self._game.num_distinct_actions(),
                                                                  legal_actions_list=next_state.legal_actions(),
                                                                  device=self._adv_buffers[traverser].device,
                                                                  dtype=torch.float32)

        # """""""""""""""""""""""""
        # Recursion
        # """""""""""""""""""""""""
        if done:
            strat_tp1 = torch.zeros_like(strat_i)
        else:
            u_bootstrap, strat_tp1 = self._recursive_traversal(
                current_state=next_state,
                traverser=traverser,
                trav_depth=trav_depth + 1,
                iteration_strats=iteration_strats,
                cfr_iter=cfr_iter,
                sample_reach=sample_reach * sample_strat[a] * n_legal_actions
            )

        # """""""""""""""""""""""""
        # Utility
        # """""""""""""""""""""""""
        utility = self._get_utility(
            traverser=traverser,
            u_bootstrap=rew_for_all[traverser] if done else u_bootstrap,
            info_state=info_state_t,
            legal_actions_list=legal_actions_list,
            legal_action_mask=legal_action_mask,
            a=a,
            sample_strat=sample_strat,
        )

        # Regret
        aprx_imm_reg = torch.full(size=(self._game.num_distinct_actions(),),
                                  fill_value=-(utility * strat_i).sum(),
                                  dtype=torch.float32,
                                  device=self._adv_buffers[traverser].device)
        aprx_imm_reg += utility
        aprx_imm_reg *= legal_action_mask

        # add current datapoint to ADVBuf
        self._adv_buffers[traverser].add(info_state=info_state,
                                         legal_action_mask=legal_action_mask,
                                         adv=aprx_imm_reg,
                                         iteration=(cfr_iter + 1) / sample_reach,
                                         )

        # add datapoint to baseline net
        self._baseline_buf.add(
            info_state=info_state,
            legal_action_mask=legal_action_mask,
            r=rew_for_all[0],
            a=a,
            done=done,
            info_state_tp1=np.array(info_state_tp1),
            strat_tp1=strat_tp1,
            legal_action_mask_tp1=legal_action_mask_tp1,
        )

        # if trav_depth == 0 and traverser == 0:
        #     self._reg_buf[traverser_range_idx].append(aprx_imm_reg.clone().cpu().numpy())

        return (utility * strat_i).sum(), strat_i

    def _any_non_traverser_act(self, current_state, traverser, trav_depth, iteration_strats,
                               sample_reach, cfr_iter):
        self.total_node_count_traversed += 1

        p_id_acting = current_state.current_player()

        current_info_state = current_state.information_state_tensor()
        legal_actions_list = current_state.legal_actions()
        legal_action_mask = utils.get_legal_action_mask_torch(n_actions=self._game.num_distinct_actions(),
                                                              legal_actions_list=legal_actions_list,
                                                              device=self._adv_buffers[traverser].device,
                                                              dtype=torch.float32)
        # """""""""""""""""""""""""
        # The players strategy
        # """""""""""""""""""""""""
        strat_opp = iteration_strats[p_id_acting].get_a_probs(
            info_state=[current_info_state],
            legal_actions_lists=[legal_actions_list],
            to_np=False
        )[0]

        # """""""""""""""""""""""""
        # Execute action from strat
        # """""""""""""""""""""""""
        a = torch.multinomial(strat_opp.cpu(), num_samples=1).item()
        next_state = current_state.child(a)

        while next_state.is_chance_node():
            action = np.random.choice([i[0] for i in next_state.chance_outcomes()])
            next_state = next_state.child(action)

        done = next_state.is_terminal()
        rew_for_all = next_state.returns()
        info_state_tp1 = next_state.information_state_tensor(traverser)
        legal_action_mask_tp1 = utils.get_legal_action_mask_torch(n_actions=self._game.num_distinct_actions(),
                                                                  legal_actions_list=next_state.legal_actions(),
                                                                  device=self._adv_buffers[traverser].device,
                                                                  dtype=torch.float32)

        # """""""""""""""""""""""""
        # Adds to opponent's
        # average buffer if
        # applicable
        # """""""""""""""""""""""""
        if self._avrg_buffers is not None:
            self._avrg_buffers[p_id_acting].add(
                info_state=np.array(current_info_state),
                legal_actions_list=legal_actions_list,
                a_probs=strat_opp.to(self._avrg_buffers[p_id_acting].device).squeeze(),
                iteration=(cfr_iter + 1) / sample_reach
            )

        # """""""""""""""""""""""""
        # Recursion
        # """""""""""""""""""""""""
        if done:
            strat_tp1 = torch.zeros_like(strat_opp)
            self.total_node_count_traversed += 1
        else:
            u_bootstrap, strat_tp1 = self._recursive_traversal(
                current_state=next_state,
                traverser=traverser,
                trav_depth=trav_depth + 1,
                iteration_strats=iteration_strats,
                cfr_iter=cfr_iter,
                sample_reach=sample_reach
            )

        # """""""""""""""""""""""""
        # Utility
        # """""""""""""""""""""""""
        utility = self._get_utility(
            traverser=traverser,
            u_bootstrap=rew_for_all[traverser] if done else u_bootstrap,
            info_state=current_info_state,
            legal_actions_list=legal_actions_list,
            legal_action_mask=legal_action_mask,
            a=a,
            sample_strat=strat_opp,
        )

        # add datapoint to baseline net
        self._baseline_buf.add(
            info_state=np.array(current_info_state),
            legal_action_mask=legal_action_mask,
            r=rew_for_all[0],  # 0 bc we mirror for 1... zero-sum
            a=a,
            done=done,
            info_state_tp1=np.array(info_state_tp1),
            strat_tp1=strat_tp1,
            legal_action_mask_tp1=legal_action_mask_tp1,
        )

        return (utility * strat_opp).sum(), strat_opp

    def _get_utility(self, traverser, info_state,
                     legal_actions_list, legal_action_mask, u_bootstrap, a, sample_strat):

        ######################
        # Remove variance from
        # action
        ######################
        baselines = self._baseline_net.get_b(
            info_states=[info_state],
            legal_actions_lists=[legal_actions_list],
            to_np=False,
        )[0] * (1 if traverser == 0 else -1)

        # print(baselines[a], u_bootstrap, a)
        utility = baselines * legal_action_mask
        utility[a] += (u_bootstrap - utility[a]) / sample_strat[a]

        return utility

