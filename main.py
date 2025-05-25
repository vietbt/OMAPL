import os
import tyro
import shutil
from pprint import pprint
from utils import set_seed
from config import Args, SMACV1_ENV_NAMES, SMACV2_ENV_NAMES, MAMUJOCO_ENV_NAMES


def main():
    set_seed(args.seed)
    if args.use_llm:
        logdir = f"logs/{args.algo}/{args.env_name}_llm/seed{args.seed}"
    else:
        logdir = f"logs/{args.algo}/{args.env_name}/seed{args.seed}"
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir, exist_ok=True)

    if args.env_name in MAMUJOCO_ENV_NAMES:
        from utils import load_mujoco_dataset as load_dataset
        is_discrete = False
    elif args.env_name in SMACV1_ENV_NAMES + SMACV2_ENV_NAMES:
        from utils import load_smac_dataset as load_dataset
        is_discrete = True
    else:
        raise NotImplementedError
    
    if args.algo == "OMAPL" and is_discrete:
        from algos.omarl import PolicyDiscrete as Agent
        from trainers.omarl import TrainerDiscrete as Trainer
    elif args.algo == "OMAPL" and not is_discrete:
        from algos.omarl import PolicyContinuous as Agent
        from trainers.omarl import TrainerContinuous as Trainer
    elif args.algo == "IPLVDN" and is_discrete:
        from algos.iplvdn import PolicyDiscrete as Agent
        from trainers.iplvdn import TrainerDiscrete as Trainer
    elif args.algo == "IPLVDN" and not is_discrete:
        from algos.iplvdn import PolicyContinuous as Agent
        from trainers.iplvdn import TrainerContinuous as Trainer
    elif args.algo == "IIPL" and is_discrete:
        from algos.iipl import PolicyDiscrete as Agent
        from trainers.iipl import TrainerDiscrete as Trainer
    elif args.algo == "IIPL" and not is_discrete:
        from algos.iipl import PolicyContinuous as Agent
        from trainers.iipl import TrainerContinuous as Trainer
    elif args.algo == "BC" and is_discrete:
        from algos.bc import PolicyDiscrete as Agent
        from trainers.bc import TrainerDiscrete as Trainer
    elif args.algo == "BC" and not is_discrete:
        from algos.bc import PolicyContinuous as Agent
        from trainers.bc import TrainerContinuous as Trainer
    elif args.algo == "SLMARL" and is_discrete:
        from algos.slmarl import PolicyDiscrete as Agent
        from trainers.slmarl import TrainerDiscrete as Trainer
    elif args.algo == "SLMARL" and not is_discrete:
        from algos.slmarl import PolicyContinuous as Agent
        from trainers.slmarl import TrainerContinuous as Trainer
    else:
        raise NotImplementedError

    dataset, st_dim, ob_dim, ac_dim, n_agents = load_dataset(args.env_name, args.batch_size, args.use_llm)
    agent = Agent(st_dim, ob_dim, ac_dim, n_agents).to(args.device)
    trainer = Trainer(agent, logdir, dataset, n_agents, args)
    trainer.train()


if __name__ == "__main__":
    args = tyro.cli(Args)
    pprint(vars(args))
    main()
    
