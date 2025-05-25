import torch.jit as jit
from algos.utils import ActorDiscrete, ActorContinuous, QNetDiscrete, QNetContinuous, VNet


class PolicyDiscrete(jit.ScriptModule):

    def __init__(self, st_dim, ob_dim, ac_dim, n_agents, h_dim=256):
        super().__init__()
        self.st_dim = st_dim
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.n_agents = n_agents
        self.h_dim = h_dim

        self.q = QNetDiscrete(ob_dim, ac_dim, n_agents, h_dim)
        self.v = VNet(ob_dim, n_agents, h_dim)
        self.actor = ActorDiscrete(ob_dim, ac_dim, n_agents, h_dim)


class PolicyContinuous(jit.ScriptModule):

    def __init__(self, st_dim, ob_dim, ac_dim, n_agents, h_dim=256):
        super().__init__()
        self.st_dim = st_dim
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.n_agents = n_agents
        self.h_dim = h_dim

        self.q = QNetContinuous(ob_dim, ac_dim, n_agents, h_dim)
        self.v = VNet(ob_dim, n_agents, h_dim)
        self.actor = ActorContinuous(ob_dim, ac_dim, n_agents, h_dim)