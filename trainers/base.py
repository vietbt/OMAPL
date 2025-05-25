import os
import copy
import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from concurrent.futures import ThreadPoolExecutor as Pool
from config import MAMUJOCO_ENV_NAMES, SMACV1_ENV_NAMES, SMACV2_ENV_NAMES, Args


class BaseTrainer:

    def __init__(self, model: nn.Module, logdir, dataset, n_agents, args: Args):
        self.model = model
        self.args = args
        self.device = args.device
        self.env_name = args.env_name

        self.lr = args.lr
        self.gamma = args.gamma
        self.tau = args.tau
        self.grad_norm_clip = args.grad_norm_clip
        self.global_step = 0
        
        self._alpha = args.alpha
        self._seed = args.seed

        self.target_model = copy.deepcopy(model).eval()
        self.target_model.load_state_dict(model.state_dict())

        self.writer = SummaryWriter(logdir)
        if self.args.use_llm:
            self.task_name = f"{args.algo}/{args.env_name}_llm/seed{args.seed}"
        else:
            self.task_name = f"{args.algo}/{args.env_name}/seed{args.seed}"

        self.n_agents = n_agents
        self.offline_data = dataset

        self.action_scale = args.action_scale
        self.action_bias = args.action_bias

    def soft_update_target(self):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def load_env(self, actor):
        if self.env_name in MAMUJOCO_ENV_NAMES:
            from envs.mamujoco.env import MaMujocoWrapper as EnvWrapper
            from rollouts.continuous import RolloutWorkerContinuous as RolloutWorker
            rollout_worker = RolloutWorker(actor, self.n_agents, self.action_scale, self.action_bias, self.device)
        elif self.env_name in SMACV1_ENV_NAMES:
            from envs.smacv1.env import SMACWrapper as EnvWrapper
            from rollouts.discrete import RolloutWorkerDiscrete as RolloutWorker
            rollout_worker = RolloutWorker(actor, self.n_agents, self.device)
        elif self.env_name in SMACV2_ENV_NAMES:
            from envs.smacv2.env import SMACWrapper as EnvWrapper
            from rollouts.discrete import RolloutWorkerDiscrete as RolloutWorker
            rollout_worker = RolloutWorker(actor, self.n_agents, self.device)
        else:
            raise NotImplementedError
        env = EnvWrapper(self.env_name, self._seed)
        return env, rollout_worker

    def eval(self, actor, step, verbose=False, evaluate=False):
        path = f"saved_models/{self.task_name}/actor_step{step}.pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        actor.save(path)
        if not evaluate:
            return
        
        env, rollout_worker = self.load_env(actor)
        results = rollout_worker.rollout(env, self.args.n_episodes, verbose=verbose)
        log_str = f"Step: {step} - env: {self.task_name}"
        if "returns" in results:
            returns = results["returns"]
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            self.writer.add_scalar("game/avg_return", avg_return, step)
            self.writer.add_scalar("game/std_return", std_return, step)
            log_str += f" - return: {avg_return:.3f}"
        if "winrates" in results:
            winrates = results["winrates"]
            avg_winrate = np.mean(winrates)
            std_winrate = np.std(winrates)
            self.writer.add_scalar("game/avg_winrate", avg_winrate, step)
            self.writer.add_scalar("game/std_winrate", std_winrate, step)
            log_str += f" - winrate: {avg_winrate:.3f}"
        print(log_str)

        os.makedirs(f"saved_results/{self.task_name}", exist_ok=True)
        with open(f"saved_results/{self.task_name}/step{step}.json", "w") as f:
            json.dump(results, f, indent=2)

        env.close()

    def train(self):
        n_epochs = self.args.n_epochs
        n_evals = self.args.n_evals
        n_workers = self.args.n_workers
        
        log_interval = len(self.offline_data) * n_epochs // n_evals

        self.eval(self.model.actor, 0, verbose=True)

        with Pool(n_workers) as p:
            tasks = []
            logs = []
            while len(tasks) < n_evals:
                for data in self.offline_data:
                    if len(tasks) >= n_evals:
                        break
                    
                    losses = self.update(data.to(self.device))
                    logs.append(losses)

                    eval_step = self.global_step // log_interval
                    
                    if self.global_step % log_interval == 0:
                        loss_vals = torch.stack(logs).mean(0)
                        q_loss = loss_vals["q_loss"].item()
                        v_loss = loss_vals["v_loss"].item()
                        pi_loss = loss_vals["pi_loss"].item()
                        logits_high = loss_vals["logits_high"].item()
                        logits_low = loss_vals["logits_low"].item()

                        self.writer.add_scalar("losses/q_loss", q_loss, eval_step)
                        self.writer.add_scalar("losses/v_loss", v_loss, eval_step)
                        self.writer.add_scalar("losses/pi_loss", pi_loss, eval_step)
                        self.writer.add_scalar("logits/high", logits_high, eval_step)
                        self.writer.add_scalar("logits/low", logits_low, eval_step)
                        logs = []

                        loss_str = f"{q_loss:.3f}/{v_loss:.3f}/{pi_loss:.3f} - logits: {logits_low:.3f}/{logits_high:.3f}"
                        print(f"Step: {eval_step} - loss: {loss_str}")

                        tasks.append(p.submit(self.eval, copy.deepcopy(self.model.actor).eval(), eval_step))

            for task in tasks:
                task.result()
        
        if self.args.use_llm:
            path = f"saved_models/{self.args.algo}/{self.env_name}_llm/model_seed{self._seed}.pt"
        else:
            path = f"saved_models/{self.args.algo}/{self.env_name}/model_seed{self._seed}.pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        self.writer.close()
