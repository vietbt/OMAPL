from dataclasses import dataclass


SMACV1_ENV_NAMES = ["2c_vs_64zg", "5m_vs_6m", "6h_vs_8z", "corridor"]
SMACV2_ENV_NAMES = [f"{map_name}_{map_mode}" for map_name in ["protoss", "terran", "zerg"] for map_mode in ["5_vs_5", "10_vs_10", "10_vs_11", "20_vs_20", "20_vs_23"]]
MAMUJOCO_ENV_NAMES = ["Hopper-v2", "Ant-v2", "HalfCheetah-v2"]


@dataclass
class Args:
    algo: str = "OMAPL"
    env_name: str = "5m_vs_6m"
    device: str = "cuda"
    seed: int = 0
    n_epochs: int = 100
    n_evals: int = 100
    n_workers: int = 4
    n_episodes: int = 32
    batch_size: int = 32
    action_scale: float = 1.0
    action_bias: float = 0.0
    use_llm: bool = False

    lr: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005
    grad_norm_clip: float = 1.0
    alpha: float = 10.0

    eval_step: int = 100