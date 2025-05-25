import os
import json
import tyro
from config import Args, SMACV1_ENV_NAMES, SMACV2_ENV_NAMES, MAMUJOCO_ENV_NAMES


def check_exists(task_name, step):
    path = f"saved_results/{task_name}/step{step}.json"
    try:
        with open(path, "r") as f:
            data = json.load(f)
            assert "returns" in data
        print(f"Already evaluated: {path}")
        return True
    except:
        print("Running:", path)
        return False


def load_env(actor):
    if args.env_name in MAMUJOCO_ENV_NAMES:
        from envs.mamujoco.env import MaMujocoWrapper as EnvWrapper
        from rollouts.continuous import RolloutWorkerContinuous as RolloutWorker
        env = EnvWrapper(args.env_name, args.seed)
        n_agents = env.env.envs[0].n_agents
        rollout_worker = RolloutWorker(actor, n_agents, args.action_scale, args.action_bias, args.device)
    elif args.env_name in SMACV1_ENV_NAMES:
        from envs.smacv1.env import SMACWrapper as EnvWrapper
        from rollouts.discrete import RolloutWorkerDiscrete as RolloutWorker
        env = EnvWrapper(args.env_name, args.seed)
        rollout_worker = RolloutWorker(actor, env.env.n_agents, args.device)
    elif args.env_name in SMACV2_ENV_NAMES:
        from envs.smacv2.env import SMACWrapper as EnvWrapper
        from rollouts.discrete import RolloutWorkerDiscrete as RolloutWorker
        env = EnvWrapper(args.env_name, args.seed)
        rollout_worker = RolloutWorker(actor, env.n_agents, args.device)
    else:
        raise NotImplementedError
    return env, rollout_worker


def evaluate(verbose=False):
    if args.use_llm:
        task_name = f"{args.algo}/{args.env_name}_llm/seed{args.seed}"
    else:
        task_name = f"{args.algo}/{args.env_name}/seed{args.seed}"
    model_path = f"saved_models/{task_name}/actor_step{args.eval_step}.pt"
    result_path = f"saved_results/{task_name}/step{args.eval_step}.json"

    try:
        with open(result_path, "r") as f:
            return json.load(f)
    except:
        pass
    print(f"Evaluating: {model_path}")

    import torch
    import numpy as np
    actor = torch.jit.load(model_path, args.device).eval()
    env, rollout_worker = load_env(actor)

    results = rollout_worker.rollout(env, args.n_episodes, verbose=verbose)
    log_str = f"Step: {args.eval_step} - env: {task_name}"
    if "returns" in results:
        returns = results["returns"]
        avg_return = np.mean(returns)
        log_str += f" - return: {avg_return:.3f}"
    if "winrates" in results:
        winrates = results["winrates"]
        avg_winrate = np.mean(winrates)
        log_str += f" - winrate: {avg_winrate:.3f}"
    print(log_str)

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    env.close()
    return results


if __name__ == "__main__":
    args = tyro.cli(Args)
    evaluate()
    
