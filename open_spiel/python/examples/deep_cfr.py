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

from absl import app
from absl import flags
from absl import logging

from open_spiel.python import policy
from open_spiel.python.algorithms.deep.DeepCFRSolver import DeepCFRSolver
from open_spiel.python.algorithms.deep.SingleDeepCFRSolver import SingleDeepCFRSolver
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms.deep.TrainingProfile import TrainingProfile
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 60, "Number of iterations")
flags.DEFINE_integer("num_traversals", 100, "Number of traversals/games")
flags.DEFINE_string("game_name", "kuhn_poker", "Name of the game")


def evaluate(game):
    def inner(solver):
        average_policy = policy.tabular_policy_from_callable(
            game, solver.action_probabilities)

        conv = exploitability.nash_conv(game, average_policy)
        logging.info("Deep CFR in '%s' - NashConv: %s", FLAGS.game_name, conv)

        average_policy_values = expected_game_score.policy_value(
            game.new_initial_state(), [average_policy] * 2)
        print("Computed player 0 value: {}".format(average_policy_values[0]))
        print("Expected player 0 value: {}".format(-1 / 18))
        print("Computed player 1 value: {}".format(average_policy_values[1]))
        print("Expected player 1 value: {}".format(1 / 18))
    return inner


def main(unused_argv):
    logging.info("Loading %s", FLAGS.game_name)
    game = pyspiel.load_game(FLAGS.game_name)
    # deep_cfr_solver = DeepCFRSolver(
    deep_cfr_solver = SingleDeepCFRSolver(
        game=game,
        t_prof=TrainingProfile(
            name="test",
            n_traversals_per_iter=FLAGS.num_traversals,

            mini_batch_size_adv=128,
            n_batches_adv_training=160,
            n_merge_and_table_layer_units_adv=8,
            n_units_final_adv=16,

            mini_batch_size_avrg=1024,
            n_batches_avrg_training=400,
            n_merge_and_table_layer_units_avrg=8,
            n_units_final_avrg=16,

            eval_every_n_iters=20,
            log_verbose=False
        ),
        num_iterations=FLAGS.num_iterations,
        evaluators=[evaluate(game)]
    )

    deep_cfr_solver.solve()




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
