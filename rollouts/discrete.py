import torch
import numpy as np
from torch.distributions import Categorical
from tqdm import tqdm


class RolloutWorkerDiscrete:

    def __init__(self, model, n_agents, device="cuda"):
        self.model = model
        self.n_agents = n_agents
        self.device = device
    
    def sample(self, obs, avails, deterministic=False):
        with torch.no_grad():
            obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device).squeeze(0)
            avails = torch.tensor(np.array(avails), dtype=torch.float32, device=self.device).squeeze(0)
            inputs = torch.cat([obs, torch.eye(self.n_agents, device=self.device)], -1)
            logits = self.model.forward(inputs) + avails.log()
            actions = logits.argmax(-1) if deterministic else Categorical(logits=logits).sample()
            actions = actions.cpu().numpy()
        return actions

    def rollout(self, env, num_episodes=32, verbose=False):
        self.model.eval()
        T_rewards, T_wins = [], []
        for _ in tqdm(range(num_episodes), "Evaluating ...", ncols=80, leave=False, disable=not verbose):
            reward_sum = 0
            obs, _, avails = env.reset()
            while True:
                actions = self.sample(obs, avails, deterministic=True)
                obs, _, rewards, dones, infos, avails = env.step(actions)
                reward_sum += np.mean(rewards)
                if np.all(dones):
                    T_rewards.append(reward_sum)
                    T_wins.append(1 if infos[0]["won"] else 0)
                    break
        self.model.train()
        return {"returns": T_rewards, "winrates": T_wins}