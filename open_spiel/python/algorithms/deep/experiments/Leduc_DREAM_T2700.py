import datetime

from open_spiel.python.algorithms.deep import DCFR
from open_spiel.python.algorithms.deep.TrainingProfile import TrainingProfile
from .HYPERS import OS_EPS


def run(game, hypers, num_iterations):
    t_prof = TrainingProfile(
        name="SD-CFR_" + game.short_name() + "_LB_2700trav_095_SEED" + datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),

        n_traversals_per_iter=2700,

        n_batches_adv_training=3000,
        sampler="learned_baseline",
        os_eps=0.5,
    )
    solver = DCFR(
        game=game,
        t_prof=t_prof,
        num_iterations=num_iterations
    )
    solver.solve()
