import torch
from trainers.base import BaseTrainer
from tensordict.tensordict import TensorDict
from torch.distributions import Categorical, Normal
from algos.bc import PolicyDiscrete, PolicyContinuous


EPS = 1e-6


class TrainerDiscrete(BaseTrainer):

    def __init__(self, model: PolicyDiscrete, logdir, dataset, n_agents, args):
        super().__init__(model, logdir, dataset, n_agents, args)
        self.actor_param = list(model.actor.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor_param, lr=self.lr)

        self.model: PolicyDiscrete
        self.target_model: PolicyDiscrete

    def update(self, data):
        all_obs = torch.cat((data["obs_x"], data["obs_y"]))
        all_avails = torch.cat((data["avails_x"], data["avails_y"]))
        actions = torch.cat((data["actions_x"], data["actions_y"]))

        agent_ids = torch.eye(self.n_agents, device=self.device).expand(all_obs.shape[0], all_obs.shape[1], -1, -1)
        all_obs = torch.cat((all_obs, agent_ids), -1)
        
        obs = all_obs[:, :-1]
        avails = all_avails[:, :-1]

        self.model.train()
        self.global_step += 1
        
        log_avails = avails.log()
        _log_avails = log_avails[..., 0]
        _log_avails[_log_avails.isinf()] = 0

        logits = self.model.actor.forward(obs) + log_avails
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        actor_loss = -log_prob.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_param, self.grad_norm_clip)
        self.actor_optimizer.step()

        self.soft_update_target()

        with torch.no_grad():
            q_loss = torch.tensor(0.0)
            v_loss = torch.tensor(0.0)
            logits_high = torch.tensor(0.0)
            logits_low = torch.tensor(0.0)
        
        return TensorDict({
            "q_loss": q_loss.item(),
            "v_loss": v_loss.item(),
            "pi_loss": actor_loss.item(),
            "logits_high": logits_high.item(),
            "logits_low": logits_low.item(),
        })


class TrainerContinuous(BaseTrainer):

    def __init__(self, model: PolicyContinuous, logdir, dataset, n_agents, args):
        super().__init__(model, logdir, dataset, n_agents, args)
        self.actor_param = list(model.actor.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor_param, lr=self.lr)
        
        self.model: PolicyContinuous
        self.target_model: PolicyContinuous

    def update(self, data):
        all_obs = torch.cat((data["obs_x"], data["obs_y"]))
        actions = torch.cat((data["actions_x"], data["actions_y"]))

        obs = all_obs[:, :-1]
        actions = (actions - self.action_bias) / self.action_scale

        self.model.train()
        self.global_step += 1

        one_hot_agent_id = torch.eye(self.n_agents, device=self.device).expand(obs.shape[0], obs.shape[1], -1, -1)
        o_with_id = torch.cat((obs, one_hot_agent_id), -1)

        mean, std = self.model.actor.forward(o_with_id)
        dist = Normal(mean, std)
        actions = torch.clamp(actions, -1. + EPS, 1. - EPS)
        pretanh_actions = torch.atanh(actions)
        pretanh_log_prob = dist.log_prob(pretanh_actions)
        log_prob = pretanh_log_prob - torch.log(1 - actions.pow(2) + EPS)
        log_prob = torch.sum(log_prob, -1)
        actor_loss = -log_prob.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_param, self.grad_norm_clip)
        self.actor_optimizer.step()

        self.soft_update_target()

        with torch.no_grad():
            q_loss = torch.tensor(0.0)
            v_loss = torch.tensor(0.0)
            logits_high = torch.tensor(0.0)
            logits_low = torch.tensor(0.0)
        
        return TensorDict({
            "q_loss": q_loss.item(),
            "v_loss": v_loss.item(),
            "pi_loss": actor_loss.item(),
            "logits_high": logits_high.item(),
            "logits_low": logits_low.item(),
        })