import torch
import torch.nn as nn
from torch.nn import functional as F
from trainers.base import BaseTrainer
from tensordict.tensordict import TensorDict
from torch.distributions import Categorical, Normal
from algos.slmarl import PolicyDiscrete, PolicyContinuous


EPS = 1e-6


class TrainerDiscrete(BaseTrainer):

    def __init__(self, model: PolicyDiscrete, logdir, dataset, n_agents, args):
        super().__init__(model, logdir, dataset, n_agents, args)
        self.reward_param = list(model.reward.parameters())
        self.v_param = list(model.v.parameters())
        self.q_param = list(model.q.parameters()) + list(model.q_mix_model.parameters())
        self.actor_param = list(model.actor.parameters())

        self.reward_optimizer = torch.optim.Adam(self.reward_param, lr=self.lr)
        self.v_optimizer = torch.optim.Adam(self.v_param, lr=self.lr)
        self.q_optimizer = torch.optim.Adam(self.q_param, lr=self.lr)
        self.actor_optimizer = torch.optim.Adam(self.actor_param, lr=self.lr)

        self.model: PolicyDiscrete
        self.target_model: PolicyDiscrete

    def get_rewards(self, states, obs, next_states, next_obs, actions, dones):
        q_values = self.model.q.forward(obs).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        mw_q, mb_q = self.model.q_mix_model.forward(states)
        q_values_tot = (mw_q * q_values).sum(-1) + mb_q.squeeze(-1)
        with torch.no_grad():
            next_v_values = self.target_model.v.forward(next_obs)
            mw_next, mb_next = self.target_model.q_mix_model.forward(next_states)
            next_v_values = (mw_next * next_v_values).sum(-1) + mb_next.squeeze(-1)
            target_v_values = (1 - dones.float()) * next_v_values
        return q_values_tot - self.gamma * target_v_values

    def update_rewards(self, states, labels):
        rewards = self.model.reward.forward(states).mean(-1).mean(-1)
        all_labels = torch.cat((labels, 1 - labels))
        self.reward_optimizer.zero_grad()
        F.mse_loss(rewards, all_labels).backward()
        self.reward_optimizer.step()

    def update(self, data):
        all_states = torch.cat((data["states_x"], data["states_y"]))
        all_obs = torch.cat((data["obs_x"], data["obs_y"]))
        all_avails = torch.cat((data["avails_x"], data["avails_y"]))
        actions = torch.cat((data["actions_x"], data["actions_y"]))
        dones = torch.cat((data["dones_x"], data["dones_y"]))
        labels = data["labels"].float()

        agent_ids = torch.eye(self.n_agents, device=self.device).expand(all_obs.shape[0], all_obs.shape[1], -1, -1)
        all_obs = torch.cat((all_obs, agent_ids), -1)
        
        states = all_states[:, :-1]
        obs = all_obs[:, :-1]
        avails = all_avails[:, :-1]

        next_states = all_states[:, 1:]
        next_obs = all_obs[:, 1:]

        self.model.train()
        self.global_step += 1

        self.update_rewards(states, labels)
        
        pred_rewards = self.get_rewards(states, obs, next_states, next_obs, actions, dones)
        with torch.no_grad():
            rewards = self.model.reward.forward(states).sum(-1)
        q_loss = F.mse_loss(rewards, pred_rewards)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q_param, self.grad_norm_clip)
        self.q_optimizer.step()

        with torch.no_grad():
            q_target_values = self.target_model.q.forward(obs).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            w_target, b_target = self.target_model.q_mix_model.forward(states)
        v_values = self.model.v.forward(obs)
        z = 1 / self._alpha * w_target * (q_target_values - v_values)
        z = torch.clamp(z, min=-10.0, max=10.0)

        with torch.no_grad():
            exp_a = torch.exp(z).squeeze(-1)
            max_z = torch.max(z)
            max_z = torch.where(max_z < -1.0, torch.tensor(-1.0).to(self.device), max_z)

        v_loss = torch.exp(z - max_z) + torch.exp(-max_z) * w_target * v_values / self._alpha
        v_loss = v_loss.mean()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v_param, self.grad_norm_clip)
        self.v_optimizer.step()

        log_avails = avails.log()
        _log_avails = log_avails[..., 0]
        _log_avails[_log_avails.isinf()] = 0

        logits = self.model.actor.forward(obs) + log_avails
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        actor_loss = -(exp_a * log_prob).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_param, self.grad_norm_clip)
        self.actor_optimizer.step()

        self.soft_update_target()

        with torch.no_grad():
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
        self.reward_param = list(model.reward.parameters())
        self.v_param = list(model.v.parameters())
        self.q_param = list(model.q.parameters()) + list(model.q_mix_model.parameters())
        self.actor_param = list(model.actor.parameters())

        self.reward_optimizer = torch.optim.Adam(self.reward_param, lr=self.lr)
        self.v_optimizer = torch.optim.Adam(self.v_param, lr=self.lr)
        self.q_optimizer = torch.optim.Adam(self.q_param, lr=self.lr)
        self.actor_optimizer = torch.optim.Adam(self.actor_param, lr=self.lr)

        self.model: PolicyContinuous
        self.target_model: PolicyContinuous

    def get_rewards(self, obs, states, all_obs, all_states, dones):
        q_values = self.model.q.forward(obs)
        mw_q, mb_q = self.model.q_mix_model.forward(states)
        q_values_tot = (mw_q * q_values).sum(-1) + mb_q.squeeze(-1)
        with torch.no_grad():
            all_v_values = self.target_model.v.forward(all_obs)
            mw_all, mb_all = self.target_model.q_mix_model.forward(all_states)
            all_v_values = (mw_all * all_v_values).sum(-1) + mb_all.squeeze(-1)
            target_v_values = (1 - dones.float()) * all_v_values[:, 1:]
        return q_values_tot - self.gamma * target_v_values
    
    def update_rewards(self, states, labels):
        rewards = self.model.reward.forward(states).mean(-1).mean(-1)
        all_labels = torch.cat((labels, 1 - labels))
        self.reward_optimizer.zero_grad()
        F.mse_loss(rewards, all_labels).backward()
        self.reward_optimizer.step()

    def update(self, data):
        all_states = torch.cat((data["states_x"], data["states_y"]))
        all_obs = torch.cat((data["obs_x"], data["obs_y"]))
        actions = torch.cat((data["actions_x"], data["actions_y"]))
        dones = torch.cat((data["dones_x"], data["dones_y"]))
        labels = data["labels"].float()

        states = all_states[:, :-1]
        obs = all_obs[:, :-1]
        actions = (actions - self.action_bias) / self.action_scale

        self.model.train()
        self.global_step += 1

        self.update_rewards(states, labels)

        one_hot_agent_id = torch.eye(self.n_agents, device=self.device).expand(obs.shape[0], obs.shape[1], -1, -1)
        all_one_hot_agent_id = torch.eye(self.n_agents, device=self.device).expand(all_obs.shape[0], all_obs.shape[1], -1, -1)

        o_with_id = torch.cat((obs, one_hot_agent_id), -1)
        o_with_a_id = torch.cat((obs, actions, one_hot_agent_id), -1)
        all_obs_with_id = torch.cat((all_obs, all_one_hot_agent_id), -1)
        
        pred_rewards = self.get_rewards(o_with_a_id, states, all_obs_with_id, all_states, dones)
        with torch.no_grad():
            rewards = self.model.reward.forward(states).sum(-1)
        q_loss = F.mse_loss(rewards, pred_rewards)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q_param, self.grad_norm_clip)
        self.q_optimizer.step()

        with torch.no_grad():
            q_target_values = self.target_model.q.forward(o_with_a_id)
            w_target, b_target = self.target_model.q_mix_model.forward(states)

        v_values = self.model.v.forward(o_with_id)
        z = 1 / self._alpha * w_target * (q_target_values - v_values)
        z = torch.clamp(z, min=-10.0, max=10.0)

        with torch.no_grad():
            exp_a = torch.exp(z).squeeze(-1)
            max_z = torch.max(z)
            max_z = torch.where(max_z < -1.0, torch.tensor(-1.0).to(self.device), max_z)

        v_loss = torch.exp(z - max_z) + torch.exp(-max_z) * w_target * v_values / self._alpha
        v_loss = v_loss.mean()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v_param, self.grad_norm_clip)
        self.v_optimizer.step()

        mean, std = self.model.actor.forward(o_with_id)
        dist = Normal(mean, std)
        actions = torch.clamp(actions, -1. + EPS, 1. - EPS)
        pretanh_actions = torch.atanh(actions)
        pretanh_log_prob = dist.log_prob(pretanh_actions)
        log_prob = pretanh_log_prob - torch.log(1 - actions.pow(2) + EPS)
        log_prob = torch.sum(log_prob, -1)
        actor_loss = -(exp_a * log_prob).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_param, self.grad_norm_clip)
        self.actor_optimizer.step()

        self.soft_update_target()

        with torch.no_grad():
            logits_high = torch.tensor(0.0)
            logits_low = torch.tensor(0.0)
        
        return TensorDict({
            "q_loss": q_loss.item(),
            "v_loss": v_loss.item(),
            "pi_loss": actor_loss.item(),
            "logits_high": logits_high.item(),
            "logits_low": logits_low.item(),
        })