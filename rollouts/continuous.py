import torch
import numpy as np
from tqdm import tqdm


class RolloutWorkerContinuous:

    def __init__(self, model, n_agents, action_scale=1, action_bias=0, device="cuda"):
        self.model = model
        self.n_agents = n_agents
        self.device = device
        self.action_scale = action_scale
        self.action_bias = action_bias
    
    def sample(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            one_hot_agent_id = torch.eye(self.n_agents).expand(obs.shape[0], -1, -1).to(self.device)
            o_with_id = torch.cat((obs, one_hot_agent_id), dim=-1)
            o_with_id = o_with_id.unsqueeze(0)
            pretanh_actions, _ = self.model.forward(o_with_id)
            pretanh_actions = pretanh_actions.squeeze(0)
            actions = torch.tanh(pretanh_actions)
            actions = self.action_scale * actions + self.action_bias
            actions = actions.cpu().numpy()
        return actions

    def rollout(self, env, num_episodes=32, verbose=False):
        self.model.eval()
        T_rewards = []
        for _ in tqdm(range(num_episodes), "Evaluating ...", ncols=80, leave=False, disable=not verbose):
            total_reward = 0
            obs, _, _ = env.reset()
            while True:
                actions = self.sample(obs)
                obs, _, rewards, dones, _, _ = env.step(actions)
                total_reward += np.mean(rewards)
                if np.all(dones):
                    T_rewards.append(total_reward)
                    break
        self.model.train()
        return {"returns": T_rewards}