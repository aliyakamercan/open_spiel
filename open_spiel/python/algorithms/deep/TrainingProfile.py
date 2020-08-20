import torch

from open_spiel.python.algorithms.deep.wrappers.AdvWrapper import AdvTrainingArgs
from open_spiel.python.algorithms.deep.wrappers.AvrgWrapper import AvrgTrainingArgs
from open_spiel.python.algorithms.deep.wrappers.BaselineWrapper import BaselineArgs
from open_spiel.python.algorithms.deep.neural import AvrgNetArgs, DuelingQArgs
from open_spiel.python.algorithms.deep.neural.MainPokerModuleFLAT import MPMArgsFLAT
from open_spiel.python.algorithms.deep.neural.MainPokerModuleFLAT_Baseline import MPMArgsFLAT_Baseline


class TrainingProfile:

    def __init__(self,
                 name,
                 # ------ General
                 nn_type="feedforward",  # "recurrent" or "feedforward"
                 log_verbose=True,
                 # ------ Computing
                 device_inference="cpu",
                 device_training="cpu",

                 # --- Evaluation
                 eval_every_n_iters=999999999,

                 # ------ General Deep CFR params
                 n_traversals_per_iter=30000,
                 iter_weighting_exponent=1.0,
                 n_actions_traverser_samples=3,

                 sampler="mo",
                 turn_off_baseline=False,  # Only for VR-OS
                 os_eps=1,
                 periodic_restart=1,

                 online=False,

                 # --- Baseline Hyperparameters
                 max_buffer_size_baseline=2e5,
                 batch_size_baseline=512,
                 n_batches_per_iter_baseline=300,

                 dim_baseline=64,
                 normalize_last_layer_FLAT_baseline=True,

                 # --- Adv Hyperparameters
                 n_batches_adv_training=5000,
                 init_adv_model="random",
                 mini_batch_size_adv=2048,
                 dim_adv=64,
                 n_mini_batches_per_la_per_update_adv=1,  # TODO: remove
                 optimizer_adv="adam",
                 loss_adv="weighted_mse",
                 lr_adv=0.001,
                 grad_norm_clipping_adv=1.0,
                 lr_patience_adv=999999999,
                 normalize_last_layer_flat_adv=True,

                 max_buffer_size_adv=2e6,

                 # ------ SPECIFIC TO AVRG NET
                 n_batches_avrg_training=15000,
                 init_avrg_model="random",
                 dim_avrg=64,
                 mini_batch_size_avrg=2048,
                 n_mini_batches_per_la_per_update_avrg=1,
                 loss_avrg="weighted_mse",
                 optimizer_avrg="adam",
                 lr_avrg=0.001,
                 grad_norm_clipping_avrg=1.0,
                 lr_patience_avrg=999999999,
                 normalize_last_layer_flat_avrg=True,

                 max_buffer_size_avrg=2e6,

                 # ------ SPECIFIC TO SINGLE
                 export_each_net=False,
                 eval_agent_max_strat_buf_size=None,

                 ):
        print(" ************************** Initing args for: ", name, "  **************************")

        mpm_args_adv = MPMArgsFLAT(other_units=dim_adv,
                                   normalize=normalize_last_layer_flat_adv)
        mpm_args_avrg = MPMArgsFLAT(other_units=dim_avrg,
                                    normalize=normalize_last_layer_flat_avrg)
        mpm_args_baseline = MPMArgsFLAT_Baseline(dim=dim_baseline,
                                                 normalize=normalize_last_layer_FLAT_baseline)
        # t_prof
        self.name = name
        self.nn_type = nn_type
        self.module_args = {
                "adv_training": AdvTrainingArgs(
                    adv_net_args=DuelingQArgs(
                        mpm_args=mpm_args_adv,
                        n_units_final=dim_adv,
                    ),
                    n_batches_adv_training=n_batches_adv_training,
                    init_adv_model=init_adv_model,
                    batch_size=mini_batch_size_adv,
                    n_mini_batches_per_update=n_mini_batches_per_la_per_update_adv,
                    optim_str=optimizer_adv,
                    loss_str=loss_adv,
                    lr=lr_adv,
                    grad_norm_clipping=grad_norm_clipping_adv,
                    device_training=device_training,
                    max_buffer_size=max_buffer_size_adv,
                    lr_patience=lr_patience_adv,
                ),
                "avrg_training": AvrgTrainingArgs(
                    avrg_net_args=AvrgNetArgs(
                        mpm_args=mpm_args_avrg,
                        n_units_final=dim_avrg,
                    ),
                    n_batches_avrg_training=n_batches_avrg_training,
                    init_avrg_model=init_avrg_model,
                    batch_size=mini_batch_size_avrg,
                    n_mini_batches_per_update=n_mini_batches_per_la_per_update_avrg,
                    loss_str=loss_avrg,
                    optim_str=optimizer_avrg,
                    lr=lr_avrg,
                    grad_norm_clipping=grad_norm_clipping_avrg,
                    device_training=device_training,
                    max_buffer_size=max_buffer_size_avrg,
                    lr_patience=lr_patience_avrg,
                ),
                "mccfr_baseline": BaselineArgs(
                    q_net_args=DuelingQArgs(
                        mpm_args=mpm_args_baseline,
                        n_units_final=dim_baseline,
                    ),
                    max_buffer_size=max_buffer_size_baseline,
                    batch_size=batch_size_baseline,
                    n_batches_per_iter_baseline=n_batches_per_iter_baseline,
                ),
            }

        self.os_eps = os_eps
        self.log_verbose = log_verbose
        self.HAVE_GPU = torch.cuda.is_available()
        assert isinstance(device_inference, str), "Please pass a string (either 'cpu' or 'cuda')!"
        self.device_inference = torch.device(device_inference)

        self.online = online
        self.n_traversals_per_iter = n_traversals_per_iter
        self.iter_weighting_exponent = iter_weighting_exponent
        self.sampler = sampler
        self.n_actions_traverser_samples = n_actions_traverser_samples

        self.eval_every_n_iters = eval_every_n_iters
        # SINGLE
        self.export_each_net = export_each_net
        self.eval_agent_max_strat_buf_size = eval_agent_max_strat_buf_size
