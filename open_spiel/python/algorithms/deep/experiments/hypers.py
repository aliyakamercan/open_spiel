from dataclasses import dataclass


@dataclass
class HyperParams:
    baseline_dim: int = 64
    advantage_dim: int = 64
    advantage_batches: int = 2048


kuhn_hyper_params = HyperParams(32, 32, 128)
leduc_hyper_params = HyperParams(64, 64, 3000)