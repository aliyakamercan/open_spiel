import copy
import time
import logging
import pickle


import numpy as np
from torch.utils.tensorboard import SummaryWriter


from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms.deep import utils
from open_spiel.python.algorithms.deep.buffers import AdvReservoirBuffer, \
    AvrgReservoirBuffer, StrategyBuffer, CrazyBaselineQCircularBuffer
from open_spiel.python.algorithms.deep.sampling_algorithms import MultiOutcomeSampler, LearnedBaselineSampler
from open_spiel.python.algorithms.deep.wrappers import AdvWrapper, AvrgWrapper, IterationWrapper, BaselineWrapper


logging.basicConfig(level=logging.INFO)


class DCFR(policy.Policy):
    """Implements a solver for the Deep CFR Algorithm.

      See https://arxiv.org/abs/1811.00164.

      Define all networks and sampling buffers/memories.  Derive losses & learning
      steps. Initialize the game state and algorithmic variables.

      Note: batch sizes default to `None` implying that training over the full
            dataset in memory is done by default.  To sample from the memories you
            may set these values to something less than the full capacity of the
            memory.
    """

    def __init__(self,
                 game,
                 t_prof,
                 num_iterations
                 ):

        all_players = list(range(game.num_players()))
        super(DCFR, self).__init__(game, all_players)

        self._game = game
        self._num_iterations = num_iterations
        self._t_prof = t_prof

        self._num_players = self._game.num_players()

        self._AVRG = False
        self._SINGLE = True
        self._BASELINE = t_prof.sampler.lower() == "learned_baseline"
        self._adv_args = t_prof.module_args["adv_training"]
        self._avrg_args = t_prof.module_args["avrg_training"]
        self._baseline_args = t_prof.module_args["mccfr_baseline"]
        self._exp_writer = self.create_summary_writer("Exploitibility")
        self._iter_nr = 0

        logging.info("Setting up advantage buffers.")
        self._adv_buffers = [
            AdvReservoirBuffer(game=self._game,
                               max_size=self._adv_args.max_buffer_size,
                               nn_type=self._t_prof.nn_type,
                               iter_weighting_exponent=self._t_prof.iter_weighting_exponent)
            for _ in range(self._num_players)
        ]

        logging.info("Setting up advantage wrappers.")
        self._adv_wrappers = [
            AdvWrapper(game=self._game,
                       adv_training_args=self._adv_args,
                       device=self._adv_args.device_training,
                       online=self._t_prof.online)
            for _ in range(self._num_players)
        ]

        # """"""""""""""""""
        # DEEP CFR
        # """"""""""""""""""

        if self._AVRG:
            logging.info("Setting up average buffers.")
            self._avrg_buffers = [
                AvrgReservoirBuffer(game=self._game,
                                    max_size=self._avrg_args.max_buffer_size,
                                    nn_type=t_prof.nn_type,
                                    iter_weighting_exponent=self._t_prof.iter_weighting_exponent)
                for _ in range(self._num_players)
            ]

            logging.info("Setting up average wrappers.")
            self._avrg_wrappers = [
                AvrgWrapper(game=self._game,
                            avrg_training_args=self._avrg_args,
                            device=self._avrg_args.device_training,
                            online=self._t_prof.online)
                for _ in range(self._num_players)
            ]

        # """"""""""""""""""
        # SINGLE-DEEP CFR
        # """"""""""""""""""

        if self._SINGLE:
            logging.info("Setting up strategy buffers.")
            self._strategy_buffers = [
                StrategyBuffer(t_prof=t_prof,
                               game=self._game,
                               max_size=None,
                               device=self._t_prof.device_inference)
                for _ in range(self._num_players)
            ]

        # """"""""""""""""""
        # DREAM
        # """"""""""""""""""

        if self._BASELINE:
            assert t_prof.module_args["mccfr_baseline"] is not None, "Please give 'baseline_args' for VR Sampler."
            logging.info("Setting up baseline wrapper.")
            self._baseline_wrapper = BaselineWrapper(game=self._game,
                                                     baseline_args=self._baseline_args)

            logging.info("Setting up baseline buffer.")
            self._baseline_buf = CrazyBaselineQCircularBuffer(owner=None, game=self._game,
                                                              max_size=self._baseline_args.max_buffer_size,
                                                              nn_type=t_prof.nn_type)

        # """"""""""""""""""
        # SAMPLER
        # """"""""""""""""""
        if self._BASELINE:  # must be baseline sampler
            logging.info("Setting up baseline sampler.")
            self._data_sampler = LearnedBaselineSampler(
                game=self._game,
                adv_buffers=self._adv_buffers,
                eps=self._t_prof.os_eps,
                baseline_net=self._baseline_wrapper,
                avrg_buffers=self._avrg_buffers if self._AVRG else None,
                baseline_buf=self._baseline_buf,
            )
        elif self._t_prof.sampler.lower() == "mo":
            logging.info("Setting up multi-outcome sampler.")
            self._data_sampler = MultiOutcomeSampler(
                game=self._game,
                adv_buffers=self._adv_buffers,
                avrg_buffers=self._avrg_buffers if self._AVRG else None,
                n_actions_traverser_samples=self._t_prof.n_actions_traverser_samples)
        elif self._t_prof.sampler.lower() == "es":
            assert not self._AVRG, "External sampling not supported with average"
            # TODO: implement ES
            raise ValueError("Currently we don't support", self._t_prof.sampler.lower(), "sampling.")
        else:
            raise ValueError("Currently we don't support", self._t_prof.sampler.lower(), "sampling.")

    def solve(self):
        while self._iter_nr < self._num_iterations:
            logging.info("Iteration: {}".format(self._iter_nr))
            data_gen_time, training_time = self.run_one_iter_alternating_update(cfr_iter=self._iter_nr)
            logging.info("\t Data gen took {} seconds.".format(data_gen_time))
            logging.info("\t Training took {} seconds.".format(training_time))

            self.evaluate(self._iter_nr)
            self.maybe_checkpoint()
            self._iter_nr += 1

        if self._AVRG:
            self._train_average_nets(self._num_iterations-1)

    def run_one_iter_alternating_update(self, cfr_iter):
        data_gen_time, training_time = 0, 0

        for p_traverser in range(self._num_players):
            t0 = time.time()

            iteration_strats = [
                IterationWrapper(t_prof=self._t_prof,
                                 game=self._game,
                                 device=self._t_prof.device_inference,
                                 adv_net=self._adv_wrappers[p].net() if cfr_iter > 0 else None)
                for p in range(self._num_players)
            ]

            self._data_sampler.generate(n_traversals=self._t_prof.n_traversals_per_iter,
                                        traverser=p_traverser,
                                        iteration_strats=iteration_strats,
                                        cfr_iter=cfr_iter,
                                        )
            t1 = time.time()
            data_gen_time += t1 - t0

            self._train_adv(p_id=p_traverser, cfr_iter=cfr_iter)

            training_time += time.time() - t1
            # PUSH ADV NETWORK for single
            if self._SINGLE:
                self.add_new_iteration_strategy_model(p_traverser, self._adv_wrappers[p_traverser].net(), cfr_iter)

        if self._BASELINE:
            self._train_baseline(n_updates=self._baseline_args.n_batches_per_iter_baseline, cfr_iter=cfr_iter)

        return data_gen_time, training_time

    # """"""""""""""""
    # ADV TRAINING
    # """"""""""""""""
    def _train_adv(self, p_id, cfr_iter):
        writer = self.create_summary_writer("adv_p{}_iter{}".format(p_id, cfr_iter))
        self.train_network(self._adv_wrappers[p_id],
                           self._adv_buffers[p_id],
                           writer,
                           self._adv_args.n_batches_adv_training)

    # """"""""""""""""
    # AVRG TRAINING
    # """"""""""""""""
    def _train_average_nets(self, cfr_iter):
        avg_train_start = time.time()
        for p_id in range(self._num_players):
            self._train_avrg(p_id, cfr_iter)
        logging.info("Avrg net training took {} seconds.".format(time.time() - avg_train_start))

    def _train_avrg(self, p_id, cfr_iter):
        writer = self.create_summary_writer("avrg_p{}_iter{}".format(p_id, cfr_iter))
        self.train_network(self._avrg_wrappers[p_id],
                           self._avrg_buffers[p_id],
                           writer,
                           self._avrg_args.n_batches_avrg_training)

    # """"""""""""""""
    # STORE ITERATION STRATEGIES
    # """"""""""""""""
    def add_new_iteration_strategy_model(self, p_id, adv_net, cfr_iter):
        iter_strat = IterationWrapper(t_prof=self._t_prof,
                                      game=self._game,
                                      device=self._t_prof.device_inference,
                                      adv_net=copy.deepcopy(adv_net),
                                      cfr_iter=cfr_iter)

        self._strategy_buffers[p_id].add(iteration_strat=iter_strat)

    # """"""""""""""
    # BASELINE
    # """"""""""""""
    def _train_baseline(self, n_updates, cfr_iter):
        writer = self.create_summary_writer("baseline_iter{}".format(cfr_iter))
        self.train_network(self._baseline_wrapper,
                           self._baseline_buf,
                           writer,
                           n_updates,
                           reset=False)

    # """"""""""""""
    # POLICY
    # """"""""""""""
    def action_probabilities(self, state, player_id=None):
        if self._AVRG:
            return self.action_probabilities_avrg(state, player_id)
        elif self._SINGLE:
            return self.action_probabilities_sd(state, player_id)

    def action_probabilities_avrg(self, state, _player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        info_state_vector = state.information_state_tensor()

        probs = self._avrg_wrappers[cur_player].get_a_probs(info_state_vector, [legal_actions])

        return {action: probs[0][action] for action in legal_actions}

    def action_probabilities_sd(self, state, _player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        info_state_vector = state.information_state_tensor()

        if self._strategy_buffers[cur_player].size == 0:
            unif_rand_legal = np.full(
                shape=self._game.num_distinct_actions(),
                fill_value=1.0 / len(legal_actions)
            ) * utils.get_legal_action_mask_np(n_actions=self._game.num_distinct_actions(),
                                               legal_actions_list=[legal_actions],
                                               dtype=np.float32)
            return unif_rand_legal
        else:
            # """""""""""""""""""""
            # Weighted by Iteration
            # """"""""""""""""""""""
            # Dim: [model_idx, action_p]
            a_probs_each_model = np.array([
                weight * strat.get_a_probs(info_state=[info_state_vector],
                                           legal_actions_lists=[legal_actions]
                                           )[0]
                for strat, weight in self._strategy_buffers[cur_player].get_strats_and_weights()
            ])

            # """"""""""""""""""""""
            # Weighted by Reach
            # """"""""""""""""""""""
            reach = self._get_reach_for_each_model(
                p_id_acting=cur_player,
                state=state
            )
            a_probs_each_model *= np.expand_dims(reach, axis=1)

            # """"""""""""""""""""""
            # Normalize
            # """"""""""""""""""""""
            # Dim: [action_p]
            a_probs = np.sum(a_probs_each_model, axis=0)

            # Dim: []
            a_probs_sum = np.sum(a_probs)

            probs = a_probs / a_probs_sum

            return {action: probs[action] for action in legal_actions}

    def _get_reach_for_each_model(self, p_id_acting, state):
        models = self._strategy_buffers[p_id_acting].strategies

        hist = {
            'info_states': [],
            'legal_action_list_batch': [],
            'a_batch': [],
            'len': 0
        }

        curr_state = self._game.new_initial_state()
        for action in state.history():
            if curr_state.current_player() == p_id_acting:
                hist["info_states"].append(np.array(curr_state.information_state_tensor(p_id_acting)))
                hist["legal_action_list_batch"].append(curr_state.legal_actions())
                hist["a_batch"].append(action)
                hist["len"] += 1
            curr_state.apply_action(action)

        if hist['len'] == 0:
            # Dim: [model_idx]
            return np.ones(shape=(len(models)), dtype=np.float32)

        # """"""""""""""""""""""
        # Batch calls history
        # and computes product
        # of result
        # """"""""""""""""""""""
        # Dim: [model_idx, history_time_step]
        prob_a_each_model_each_timestep = np.array(
            [
                model.get_a_probs(
                    info_state=hist['info_states'],
                    legal_actions_lists=hist['legal_action_list_batch'],
                )[np.arange(hist['len']), hist['a_batch']]

                for model in models
            ]
        )
        # Dim: [model_idx]
        return np.prod(a=prob_a_each_model_each_timestep, axis=1)

    # """"""""""""""
    # EVALUATE
    # """"""""""""""

    def evaluate(self, cfr_iter):
        if (cfr_iter + 1) % self._t_prof.eval_every_n_iters == 0:
            if self._AVRG:
                self._train_average_nets(cfr_iter)
                average_policy = policy.tabular_policy_from_callable(
                    self._game, self.action_probabilities_avrg)
                conv = exploitability.exploitability(self._game, average_policy)
                self._exp_writer.add_scalar("Exp/Nodes Visited/avrg", conv,
                                            self._data_sampler.total_node_count_traversed)
                self._exp_writer.add_scalar("Exp/Iteration/avrg", conv, cfr_iter)
            if self._SINGLE:
                average_policy = policy.tabular_policy_from_callable(
                    self._game, self.action_probabilities_sd)
                conv = exploitability.exploitability(self._game, average_policy)
                self._exp_writer.add_scalar("Exp/Nodes Visited/single", conv,
                                            self._data_sampler.total_node_count_traversed)
                self._exp_writer.add_scalar("Exp/Iteration/single", conv, cfr_iter)

    # """""""""""""""""
    # SAVE / LOAD
    # """""""""""""""""
    def maybe_checkpoint(self):
        if self._iter_nr + 1 % self._t_prof.checkpoint_freq == 0:
            logging.info("Check pointing")
            self.save()

    def save(self):
        state = {
            'advantage_buffers': [
                self._adv_buffers[p_id].state_dict()
                for p_id in range(self._num_players)
            ],
            'iter': self._iter_nr
        }

        if self._AVRG:
            state['avg_wrappers'] = [
                self._avrg_wrappers[p_id].state_dict()
                for p_id in range(self._num_players)
            ]
            state['avg_buffers'] = [
                self._avrg_buffers[p_id].state_dict()
                for p_id in range(self._num_players)
            ]

        if self._SINGLE:
            state['strategy_buffers'] = [
                self._strategy_buffers[p_id].state_dict()
                for p_id in range(self._num_players)
            ]

        if self._BASELINE:
            state['baseline_buffer'] = self._baseline_buf.state_dict()
            state['baseline_wrapper'] = self._baseline_wrapper.state_dict()

        with open("%s/%s" % (self._t_prof.checkpoint_dir, self._t_prof.name), "wb") as pkl_file:
            pickle.dump(obj=state, file=pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, "rb") as pkl_file:
            state = pickle.load(pkl_file)

        for p_id in range(self._num_players):
            self._adv_buffers[p_id].load_state_dict(state['advantage_buffers'][p_id])
            if self._AVRG:
                self._avrg_wrappers[p_id].load_state_dict(state['avg_wrappers'][p_id])
                self._avrg_buffers[p_id].load_state_dict(state['avg_buffers'][p_id])
            if self._SINGLE:
                self._strategy_buffers[p_id].load_state_dict(state['strategy_buffers'][p_id])

        if self._BASELINE:
            self._baseline_buf.load_state_dict(state['baseline_buffer'])
            self._baseline_wrapper.load_state_dict(state['baseline_wrapper'])

        self._iter_nr = state['iter']

    # """""""""""""""
    # utils
    # """"""""""""""
    def create_summary_writer(self, name):
        return SummaryWriter(
            log_dir="runs/" + self._t_prof.name + "/" + name,
            max_queue=4,
            flush_secs=2
        )

    @staticmethod
    def train_network(wrapper, buffer, writer, epochs, reset=True):
        if reset:
            wrapper.reset()
        for epoch_nr in range(epochs):
            averaged_loss = wrapper.train_one_loop(buffer=buffer)
            if averaged_loss:
                wrapper.step_scheduler(averaged_loss)
                writer.add_scalar("Loss", averaged_loss, epoch_nr)
