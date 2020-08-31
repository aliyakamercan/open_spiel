# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python Deep CFR example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
from absl import app
from absl import flags
from open_spiel.python.algorithms.deep import DCFR
from open_spiel.python.algorithms.deep.TrainingProfile import TrainingProfile
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 150, "Number of iterations")


def main(unused_argv):
    game = pyspiel.load_game(
        "universal_poker",
        {"betting": pyspiel.GameParameter("limit"),
         "numPlayers": pyspiel.GameParameter(2),
         "numRounds": pyspiel.GameParameter(1),
         "blind": pyspiel.GameParameter("1 1"),
         "raiseSize": pyspiel.GameParameter("1 "),
         "firstPlayer": pyspiel.GameParameter("1 "),
         "maxRaises": pyspiel.GameParameter("1 "),
         "numSuits": pyspiel.GameParameter(1),
         "numRanks": pyspiel.GameParameter(3),
         "numHoleCards": pyspiel.GameParameter(1),
         "numBoardCards": pyspiel.GameParameter("0 "),
         "bettingAbstraction": pyspiel.GameParameter("limit")})
    # game = pyspiel.load_game(FLAGS.game_name)
    deep_cfr_solver = DCFR(
        game=game,
        t_prof=TrainingProfile(
            name="test-" + datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S)"),
            nn_type="feedforward",

            n_batches_adv_training=70,
            dim_adv=16,
            lr_adv=0.001,
            lr_patience_adv=2,
            iter_weighting_exponent=.5,
            # grad_norm_clipping_adv=None,
            # init_adv_model="last",

            n_traversals_per_iter=900,
            sampler="mo",
            n_batches_per_iter_baseline=150,

            n_actions_traverser_samples=None,


            os_eps=0.0,
            n_batches_avrg_training=4000,

            eval_every_n_iters=2,
            log_verbose=False
        ),
        num_iterations=FLAGS.num_iterations
    )

    deep_cfr_solver.solve()


def main2(args):
    game_2 = pyspiel.UniversalPokerGame(
        {"betting": pyspiel.GameParameter("limit"),
         "numPlayers": pyspiel.GameParameter(2),
         "numRounds": pyspiel.GameParameter(1),
         "blind": pyspiel.GameParameter("1 1"),
         "raiseSize": pyspiel.GameParameter("1 "),
         "firstPlayer": pyspiel.GameParameter("1 "),
         "maxRaises": pyspiel.GameParameter("1 "),
         "numSuits": pyspiel.GameParameter(1),
         "numRanks": pyspiel.GameParameter(3),
         "numHoleCards": pyspiel.GameParameter(1),
         "numBoardCards": pyspiel.GameParameter("0 "),
         "bettingAbstraction": pyspiel.GameParameter("limit")})
    # state = game_2.new_initial_state()
    # print(game_2.num_distinct_actions())
    # # print(state.legal_actions())
    # state.apply_action(0)
    # state.apply_action(1)
    #
    # print(state.information_state_string())
    # # pcool(state.information_state_tensor())
    # # print(state.legal_actions())
    #
    # state.apply_action(2)
    # print(state.information_state_string())
    # print(state.information_state_tensor())
    # pretty_universal_poker_tensor(game_2, state.information_state_tensor())


def pretty_universal_poker_tensor(game, tensor, rounds=1, board_cards=0):
    num_players = game.num_players()
    current_actor_start = 4 + num_players
    round_start = current_actor_start + num_players
    p_states_start = round_start + 1
    history_start = p_states_start + num_players * 4
    board_start = history_start + rounds * 4 * num_players
    hole_start = board_start + 3 * board_cards
    s = """
Min Raise: {min_raise}
Main Pot: {main_pot}
Biggest Bet To Call: {bbtc}
Last Action Size (BB): {last_action}
Last Actor (1-hot): {last_actor}
Current Actor (1-hot): {current_actor}
Round: {round}
Player States: 
{player_states}
History:
{history}
Board: {board}
Hole Cards: {hole_cards}
    """.format(
        min_raise=tensor[0],
        main_pot=tensor[1],
        bbtc=tensor[2],
        last_action=tensor[3],
        last_actor=tensor[4:p_states_start],
        current_actor=tensor[current_actor_start:round_start],
        round=tensor[round_start],
        player_states=tensor[p_states_start:history_start],
        history=tensor[history_start:board_start],
        board=tensor[board_start:hole_start],
        hole_cards=tensor[hole_start:],
    )
    print(s)


def main3(args):
    game = pyspiel.load_game("universal_poker")
    bots = [
        pyspiel.make_uniform_random_bot(0, 1234),
        uniform_random.UniformRandomBot(1, np.random.RandomState(4321)),
    ]
    results = np.array([
        pyspiel.evaluate_bots(game.new_initial_state(), bots, iteration)
        for iteration in range(10000)
    ])
    universal_poker_average_results = np.mean(results, axis=0)
    print(universal_poker_average_results)


if __name__ == "__main__":
  app.run(main3)
