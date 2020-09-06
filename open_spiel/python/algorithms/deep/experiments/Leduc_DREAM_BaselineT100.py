import datetime

from open_spiel.python.algorithms.deep import DCFR
from open_spiel.python.algorithms.deep.TrainingProfile import TrainingProfile
from .HYPERS import OS_EPS


def run(game, hypers, num_iterations):
    t_prof = TrainingProfile(
        name="DREAM_BaselineT100_" + game.short_name + "_" + datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        nn_type="feedforward",

        n_batches_adv_training=hypers['BATCHES'],

        n_traversals_per_iter=hypers['TRAVERSALS_OS'],
        sampler="learned_baseline",
        n_batches_per_iter_baseline=100,

        os_eps=OS_EPS,
    ),

    solver = DCFR(
        game=game,
        t_prof=t_prof,
        num_iterations=num_iterations
    )
    solver.solve()
