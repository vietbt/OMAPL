import torch
import torch.nn as nn
from torch.nn import functional as F
from trainers.base import BaseTrainer
from tensordict.tensordict import TensorDict
from torch.distributions import Categorical, Normal
from algos.iipl import PolicyDiscrete, PolicyContinuous


EPS = 1e-6


class TrainerDiscrete(BaseTrainer):

    def __init__(self, model: PolicyDiscrete, logdir, dataset, n_agents, args):
        super().__init__(model, logdir, dataset, n_agents, args)
        self.v_param = [list(model.v[agent_id].parameters()) for agent_id in range(self.n_agents)]
        self.q_param = [list(model.q[agent_id].parameters()) + list(model.q_mix_model[agent_id].parameters()) for agent_id in range(self.n_agents)]
        self.actor_param = [list(model.actor[agent_id].parameters()) for agent_id in range(self.n_agents)]

        self.v_optimizer = [torch.optim.Adam(self.v_param[agent_id], lr=self.lr) for agent_id in range(self.n_agents)]
        self.q_optimizer = [torch.optim.Adam(self.q_param[agent_id], lr=self.lr) for agent_id in range(self.n_agents)]
        self.actor_optimizer = [torch.optim.Adam(self.actor_param[agent_id], lr=self.lr) for agent_id in range(self.n_agents)]

        self.model: PolicyDiscrete
        self.target_model = PolicyDiscrete(self.model.st_dim, self.model.ob_dim, self.model.ac_dim, self.n_agents, self.model.h_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

    def get_rewards(self, states, obs, next_states, next_obs, actions, dones, agent_id):
        q_values = self.model.q[agent_id].forward(obs).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        mw_q, mb_q = self.model.q_mix_model[agent_id].forward(states)
        q_values_tot = (mw_q * q_values).sum(-1) + mb_q.squeeze(-1)
        with torch.no_grad():
            next_v_values = self.target_model.v[agent_id].forward(next_obs)
            mw_next, mb_next = self.target_model.q_mix_model[agent_id].forward(next_states)
            next_v_values = (mw_next * next_v_values).sum(-1) + mb_next.squeeze(-1)
            target_v_values = (1 - dones.float()) * next_v_values
        return q_values_tot - self.gamma * target_v_values

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

        outputs = []
        for agent_id in range(self.n_agents):
            output = self._update(states, obs, next_states, next_obs, actions, dones, labels, avails, agent_id)
            outputs.append(output)
        return torch.stack(outputs).mean()
        
    def _update(self, states, obs, next_states, next_obs, actions, dones, labels, avails, agent_id):
        obs = obs[:, :, agent_id].unsqueeze(2)
        next_obs = next_obs[:, :, agent_id].unsqueeze(2)
        actions = actions[:, :, agent_id].unsqueeze(2)
        avails = avails[:, :, agent_id].unsqueeze(2)
        
        pred_rewards = self.get_rewards(states, obs, next_states, next_obs, actions, dones, agent_id)
        pred_rewards_x, pred_rewards_y = pred_rewards.chunk(2, dim=0)
        logits_x = pred_rewards_x.sum(1)
        logits_y = pred_rewards_y.sum(1)
        logits = logits_x - logits_y
        q_loss = F.binary_cross_entropy_with_logits(logits, labels)
        reg_loss = 0.5 * pred_rewards.pow(2).mean()

        self.q_optimizer[agent_id].zero_grad()
        (q_loss + reg_loss).backward()
        nn.utils.clip_grad_norm_(self.q_param[agent_id], self.grad_norm_clip)
        self.q_optimizer[agent_id].step()

        with torch.no_grad():
            q_target_values = self.target_model.q[agent_id].forward(obs).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            w_target, b_target = self.target_model.q_mix_model[agent_id].forward(states)
        v_values = self.model.v[agent_id].forward(obs)
        z = 1 / self._alpha * w_target * (q_target_values - v_values)
        z = torch.clamp(z, min=-10.0, max=10.0)

        with torch.no_grad():
            exp_a = torch.exp(z).squeeze(-1)
            max_z = torch.max(z)
            max_z = torch.where(max_z < -1.0, torch.tensor(-1.0).to(self.device), max_z)

        v_loss = torch.exp(z - max_z) + torch.exp(-max_z) * w_target * v_values / self._alpha
        v_loss = v_loss.mean()

        self.v_optimizer[agent_id].zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v_param[agent_id], self.grad_norm_clip)
        self.v_optimizer[agent_id].step()

        log_avails = avails.log()
        _log_avails = log_avails[..., 0]
        _log_avails[_log_avails.isinf()] = 0

        logits = self.model.actor[agent_id].forward(obs) + log_avails
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        actor_loss = -(exp_a * log_prob).mean()

        self.actor_optimizer[agent_id].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_param[agent_id], self.grad_norm_clip)
        self.actor_optimizer[agent_id].step()

        self.soft_update_target()

        with torch.no_grad():
            logits_high = torch.where(labels==1, logits_x, logits_y).mean()
            logits_low = torch.where(labels==1, logits_y, logits_x).mean()
        
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
        self.v_param = [list(model.v[agent_id].parameters()) for agent_id in range(self.n_agents)]
        self.q_param = [list(model.q[agent_id].parameters()) + list(model.q_mix_model[agent_id].parameters()) for agent_id in range(self.n_agents)]
        self.actor_param = [list(model.actor[agent_id].parameters()) for agent_id in range(self.n_agents)]

        self.v_optimizer = [torch.optim.Adam(self.v_param[agent_id], lr=self.lr) for agent_id in range(self.n_agents)]
        self.q_optimizer = [torch.optim.Adam(self.q_param[agent_id], lr=self.lr) for agent_id in range(self.n_agents)]
        self.actor_optimizer = [torch.optim.Adam(self.actor_param[agent_id], lr=self.lr) for agent_id in range(self.n_agents)]

        self.model: PolicyContinuous
        self.target_model = PolicyContinuous(self.model.st_dim, self.model.ob_dim, self.model.ac_dim, self.n_agents, self.model.h_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

    def get_rewards(self, obs, states, all_obs, all_states, dones, agent_id):
        q_values = self.model.q[agent_id].forward(obs)
        mw_q, mb_q = self.model.q_mix_model[agent_id].forward(states)
        q_values_tot = (mw_q * q_values).sum(-1) + mb_q.squeeze(-1)
        with torch.no_grad():
            all_v_values = self.target_model.v[agent_id].forward(all_obs)
            mw_all, mb_all = self.target_model.q_mix_model[agent_id].forward(all_states)
            all_v_values = (mw_all * all_v_values).sum(-1) + mb_all.squeeze(-1)
            target_v_values = (1 - dones.float()) * all_v_values[:, 1:]
        return q_values_tot - self.gamma * target_v_values

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

        one_hot_agent_id = torch.eye(self.n_agents, device=self.device).expand(obs.shape[0], obs.shape[1], -1, -1)
        all_one_hot_agent_id = torch.eye(self.n_agents, device=self.device).expand(all_obs.shape[0], all_obs.shape[1], -1, -1)

        o_with_id = torch.cat((obs, one_hot_agent_id), -1)
        o_with_a_id = torch.cat((obs, actions, one_hot_agent_id), -1)
        all_obs_with_id = torch.cat((all_obs, all_one_hot_agent_id), -1)

        outputs = []
        for agent_id in range(self.n_agents):
            output = self._update(o_with_id, o_with_a_id, states, all_obs_with_id, all_states, actions, dones, labels, agent_id)
            outputs.append(output)
        return torch.stack(outputs).mean()

    def _update(self, o_with_id, o_with_a_id, states, all_obs_with_id, all_states, actions, dones, labels, agent_id):
        o_with_id = o_with_id[:, :, agent_id].unsqueeze(2)
        o_with_a_id = o_with_a_id[:, :, agent_id].unsqueeze(2)
        all_obs_with_id = all_obs_with_id[:, :, agent_id].unsqueeze(2)
        actions = actions[:, :, agent_id].unsqueeze(2)
        
        pred_rewards = self.get_rewards(o_with_a_id, states, all_obs_with_id, all_states, dones, agent_id)
        pred_rewards_x, pred_rewards_y = pred_rewards.chunk(2, dim=0)
        logits_x = pred_rewards_x.sum(1)
        logits_y = pred_rewards_y.sum(1)
        logits = logits_x - logits_y
        q_loss = F.binary_cross_entropy_with_logits(logits, labels)
        reg_loss = 0.5 * pred_rewards.pow(2).mean()

        self.q_optimizer[agent_id].zero_grad()
        (q_loss + reg_loss).backward()
        nn.utils.clip_grad_norm_(self.q_param[agent_id], self.grad_norm_clip)
        self.q_optimizer[agent_id].step()

        with torch.no_grad():
            q_target_values = self.target_model.q[agent_id].forward(o_with_a_id)
            w_target, b_target = self.target_model.q_mix_model[agent_id].forward(states)

        v_values = self.model.v[agent_id].forward(o_with_id)
        z = 1 / self._alpha * w_target * (q_target_values - v_values)
        z = torch.clamp(z, min=-10.0, max=10.0)

        with torch.no_grad():
            exp_a = torch.exp(z).squeeze(-1)
            max_z = torch.max(z)
            max_z = torch.where(max_z < -1.0, torch.tensor(-1.0).to(self.device), max_z)

        v_loss = torch.exp(z - max_z) + torch.exp(-max_z) * w_target * v_values / self._alpha
        v_loss = v_loss.mean()

        self.v_optimizer[agent_id].zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v_param[agent_id], self.grad_norm_clip)
        self.v_optimizer[agent_id].step()

        mean, std = self.model.actor[agent_id].forward(o_with_id)
        dist = Normal(mean, std)
        actions = torch.clamp(actions, -1. + EPS, 1. - EPS)
        pretanh_actions = torch.atanh(actions)
        pretanh_log_prob = dist.log_prob(pretanh_actions)
        log_prob = pretanh_log_prob - torch.log(1 - actions.pow(2) + EPS)
        log_prob = torch.sum(log_prob, -1)
        actor_loss = -(exp_a * log_prob).mean()

        self.actor_optimizer[agent_id].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_param[agent_id], self.grad_norm_clip)
        self.actor_optimizer[agent_id].step()

        self.soft_update_target()

        with torch.no_grad():
            logits_high = torch.where(labels==1, logits_x, logits_y).mean()
            logits_low = torch.where(labels==1, logits_y, logits_x).mean()
        
        return TensorDict({
            "q_loss": q_loss.item(),
            "v_loss": v_loss.item(),
            "pi_loss": actor_loss.item(),
            "logits_high": logits_high.item(),
            "logits_low": logits_low.item(),
        })