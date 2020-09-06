import datetime

from open_spiel.python.algorithms.deep import DCFR
from open_spiel.python.algorithms.deep.TrainingProfile import TrainingProfile
from open_spiel.python.algorithms.deep.experiments.hypers import kuhn_hyper_params
from open_spiel.python.algorithms.deep.experiments.games import kuhn_poker_2p, leduc_poker_2p


OS_EPS = 0.5


def prof(name, op_eps, baseline_iters, num_traversal, hypers):
    return TrainingProfile(
        name="_".join(["DREAM", name, "eps", str(op_eps), "BT", str(baseline_iters), "T", str(num_traversal),
                       datetime.datetime.now().strftime("%d-%m-%YT%H:%M:%S")]),
        nn_type="feedforward",

        n_batches_adv_training=hypers.advantage_batches,
        sampler="learned_baseline",
        # sampler="mo",

        lr_patience_adv=2,
        dim_baseline=hypers.baseline_dim,
        dim_adv=hypers.advantage_dim,

        os_eps=op_eps,
        n_batches_per_iter_baseline=baseline_iters,
        n_traversals_per_iter=num_traversal,
    )


# kuhn_eps05_BT100_T900 = prof("kuhn", 0.25, 200, 900, kuhn_hyper_params)
leduc_eps05_BT1000_T900 = prof("leduc", 1, 1000, 200, kuhn_hyper_params)


def run(game, t_prof, num_iterations):
    solver = DCFR(
        game=game,
        t_prof=t_prof,
        num_iterations=num_iterations
    )
    solver.solve()


if __name__ == "__main__":
    run(leduc_poker_2p, leduc_eps05_BT1000_T900, 150)
