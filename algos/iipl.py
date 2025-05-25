import torch
import torch.nn as nn
import torch.jit as jit
from algos.utils import ActorDiscrete, ActorContinuous, QNetDiscrete, QNetContinuous, VNet, MixNet



class ModuleListDiscrete(nn.ModuleList):

    def forward(self, x):
        outputs = [module.forward(x[i].unsqueeze(0)) for i, module in enumerate(self)]
        outputs = torch.stack(outputs, dim=0).squeeze(1)
        return outputs


class ModuleListContinuous(nn.ModuleList):

    def forward(self, x):
        out_means, out_stds = [], []
        for i, module in enumerate(self):
            mean, std = module.forward(x[:, :, i].unsqueeze(2))
            out_means.append(mean)
            out_stds.append(std)
        out_means = torch.stack(out_means, dim=2).squeeze(3)
        out_stds = torch.stack(out_stds, dim=2).squeeze(3)
        return out_means, out_stds
    

class PolicyDiscrete(jit.ScriptModule):

    def __init__(self, st_dim, ob_dim, ac_dim, n_agents, h_dim=256):
        super().__init__()
        self.st_dim = st_dim
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.n_agents = n_agents
        self.h_dim = h_dim

        self.q_mix_model = nn.ModuleList([MixNet(st_dim, n_agents) for _ in range(n_agents)])
        self.q = nn.ModuleList([QNetDiscrete(ob_dim, ac_dim, n_agents, h_dim) for _ in range(n_agents)])
        self.v = nn.ModuleList([VNet(ob_dim, n_agents, h_dim) for _ in range(n_agents)])
        self.actor = ModuleListDiscrete([ActorDiscrete(ob_dim, ac_dim, n_agents, h_dim) for _ in range(n_agents)])


class PolicyContinuous(jit.ScriptModule):

    def __init__(self, st_dim, ob_dim, ac_dim, n_agents, h_dim=256):
        super().__init__()
        self.st_dim = st_dim
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.n_agents = n_agents
        self.h_dim = h_dim

        self.q_mix_model = nn.ModuleList([MixNet(st_dim, n_agents) for _ in range(n_agents)])
        self.q = nn.ModuleList([QNetContinuous(ob_dim, ac_dim, n_agents, h_dim) for _ in range(n_agents)])
        self.v = nn.ModuleList([VNet(ob_dim, n_agents, h_dim) for _ in range(n_agents)])
        self.actor = ModuleListContinuous([ActorContinuous(ob_dim, ac_dim, n_agents, h_dim) for _ in range(n_agents)])