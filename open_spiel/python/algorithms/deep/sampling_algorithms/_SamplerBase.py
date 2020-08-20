import torch
import numpy as np


class SamplerBase:

    def __init__(self, game, adv_buffers, avrg_buffers=None, ):
        self._game = game
        self._adv_buffers = adv_buffers
        self._avrg_buffers = avrg_buffers
        self._root_node = self._game.new_initial_state()
        self.total_node_count_traversed = 0

    def _traverser_act(self, current_state, traverser, trav_depth, iteration_strats, sample_reach, cfr_iter):
        raise NotImplementedError

    def generate(self, n_traversals, traverser, iteration_strats, cfr_iter, ):
        for _ in range(n_traversals):
            self._traverse_once(traverser=traverser, iteration_strats=iteration_strats, cfr_iter=cfr_iter)

    def _traverse_once(self, traverser, iteration_strats, cfr_iter, ):
        """
        Args:
            traverser (int):                    seat id of the traverser
            iteration_strats (IterationStrategy):
            cfr_iter (int):                  current iteration of Deep CFR
        """
        self._recursive_traversal(current_state=self._root_node,
                                  traverser=traverser,
                                  trav_depth=0,
                                  iteration_strats=iteration_strats,
                                  sample_reach=1.0,
                                  cfr_iter=cfr_iter,
                                  )

    def _recursive_traversal(self, current_state, traverser, trav_depth, iteration_strats,
                             cfr_iter, sample_reach):
        """
        assumes passed state_dict is NOT done!
        """

        if current_state.current_player() == traverser:
            return self._traverser_act(current_state=current_state,
                                       traverser=traverser,
                                       trav_depth=trav_depth,
                                       iteration_strats=iteration_strats,
                                       sample_reach=sample_reach,
                                       cfr_iter=cfr_iter)

        if current_state.is_chance_node():
            action = np.random.choice([i[0] for i in current_state.chance_outcomes()])
            next_state = current_state.child(action)
            return self._recursive_traversal(current_state=next_state,
                                             traverser=traverser,
                                             trav_depth=trav_depth,
                                             iteration_strats=iteration_strats,
                                             sample_reach=sample_reach,
                                             cfr_iter=cfr_iter)

        return self._any_non_traverser_act(current_state=current_state,
                                           traverser=traverser,
                                           trav_depth=trav_depth,
                                           iteration_strats=iteration_strats,
                                           sample_reach=sample_reach,
                                           cfr_iter=cfr_iter)

    def _any_non_traverser_act(self, current_state, traverser, trav_depth, iteration_strats,
                               sample_reach, cfr_iter):

        p_id_acting = current_state.current_player()

        info_state = np.array(current_state.information_state_tensor(p_id_acting))
        legal_actions_list = current_state.legal_actions()

        # """""""""""""""""""""""""
        # The players strategy
        # """""""""""""""""""""""""
        a_probs = iteration_strats[p_id_acting].get_a_probs(
            info_state=[info_state],
            legal_actions_lists=[legal_actions_list],
            to_np=False
        )[0]

        # """""""""""""""""""""""""
        # Adds to opponent's
        # average buffer if
        # applicable
        # """""""""""""""""""""""""
        if self._avrg_buffers is not None:
            self._avrg_buffers[p_id_acting].add(
                info_state=info_state,
                legal_actions_list=legal_actions_list,
                a_probs=a_probs.to(self._avrg_buffers[p_id_acting].device).squeeze(),
                iteration=cfr_iter + 1)

        # """""""""""""""""""""""""
        # Execute action from strat
        # """""""""""""""""""""""""
        a = torch.multinomial(a_probs.cpu(), num_samples=1).item()
        next_state = current_state.child(a)
        _rew_traverser = next_state.returns()[traverser]
        _done = next_state.is_terminal()

        # """""""""""""""""""""""""
        # Recurse or Return if done
        # """""""""""""""""""""""""
        if _done:
            self.total_node_count_traversed += 1
            return _rew_traverser

        return _rew_traverser + self._recursive_traversal(
            current_state=next_state,
            traverser=traverser,
            trav_depth=trav_depth,
            iteration_strats=iteration_strats,
            sample_reach=sample_reach,
            cfr_iter=cfr_iter
        )
