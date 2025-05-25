import h5py
import tyro
import json
import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from config import SMACV1_ENV_NAMES, SMACV2_ENV_NAMES
from envs.smacv1.env import SMACWrapper as SMACWrapperV1
from envs.smacv2.env import SMACWrapper as SMACWrapperV2


descriptions = {
    "5m_vs_6m": [
        "- Allied Team Agent Configuration : five Marines(Marines are ranged units in StarCraft 2).",
        "- Enemy Team Agent Configuration : six Marines(Marines are ranged units in StarCraft 2).",
        "- Situation Description : The situation involves the allied team and the enemy team engaging in combat, where victory is achieved by defeating all the enemies.",
        "- Objective : Defeat all enemy agents while ensuring as many allied agents as possible survive.",
        "* Important Notice : You should prefer the trajectory where our allies' health is preserved while significantly reducing the enemy's health. In similar situations, you should prefer shorter trajectory lengths.",
    ],

    "6h_vs_8z": [
        "- Allied Team Agent Configuration : six Hydras(Hydras are long-range attack units in StarCraft 2).",
        "- Enemy Team Agent Configuration : eight Zealots(Zealots are close-range attack units in StarCraft 2).",
        "- Situation Description : The situation involves the allied team and the enemy team engaging in combat, where victory is achieved by defeating all the enemies.",
        "- Objective : Defeat all enemy agents while ensuring as many allied agents as possible survive.",
        "* Important Notice : You should prefer the trajectory where our allies' health is preserved while significantly reducing the enemy's health. In similar situations, you should prefer shorter trajectory lengths.",
    ],

    "2c_vs_64zg": [
        "- Allied Team Agent Configuration : two Colossi(Colossi are powerful ranged units in StarCraft 2).",
        "- Enemy Team Agent Configuration : sixty-four Zerglings(Zerglings are fast melee units in StarCraft 2).",
        "- Situation Description : The situation involves the allied team and the enemy team engaging in combat, where victory is achieved by defeating all the enemies.",
        "- Objective : Defeat all enemy agents while ensuring as many allied agents as possible survive.",
        "* Important Notice : You should prefer the trajectory where our allies' health is preserved while significantly reducing the enemy's health. In similar situations, you should prefer shorter trajectory lengths.",
    ],

    "corridor": [
        "- Allied Team Agent Configuration : six Zealots(Zealots are close-range attack units in StarCraft 2).",
        "- Enemy Team Agent Configuration : twenty-four Zerglings(Zerglings are fast melee units in StarCraft 2).",
        "- Situation Description : The situation involves the allied team and the enemy team engaging in combat, where victory is achieved by defeating all the enemies.",
        "- Objective : Defeat all enemy agents while ensuring as many allied agents as possible survive.",
        "* Important Notice : You should prefer the trajectory where our allies' health is preserved while significantly reducing the enemy's health. In similar situations, you should prefer shorter trajectory lengths.",
    ],

    "protoss_5_vs_5": [
        "- Allied Team Agent Configuration : five units consisting of a mix of Stalkers, Zealots, and Colossi (Stalkers are versatile ranged units, Zealots are close-range melee units, and Colossi are powerful long-range units in StarCraft 2).",
        "- Enemy Team Agent Configuration : five units consisting of a mix of Stalkers, Zealots, and Colossi (same as the allied team).",
        "- Situation Description : The situation involves the allied team and the enemy team engaging in combat, where victory is achieved by defeating all the enemies.",
        "- Objective : Defeat all enemy agents while ensuring as many allied agents as possible survive.",
        "* Important Notice : You should prefer the trajectory where our allies' health is preserved while significantly reducing the enemy's health. In similar situations, you should prefer shorter trajectory lengths.",
    ],

    "protoss_10_vs_10": [
        "- Allied Team Agent Configuration : ten units consisting of a mix of Stalkers, Zealots, and Colossi (Stalkers are versatile ranged units, Zealots are close-range melee units, and Colossi are powerful long-range units in StarCraft 2).",
        "- Enemy Team Agent Configuration : ten units consisting of a mix of Stalkers, Zealots, and Colossi (same as the allied team).",
        "- Situation Description : The situation involves the allied team and the enemy team engaging in combat, where victory is achieved by defeating all the enemies.",
        "- Objective : Defeat all enemy agents while ensuring as many allied agents as possible survive.",
        "* Important Notice : You should prefer the trajectory where our allies' health is preserved while significantly reducing the enemy's health. In similar situations, you should prefer shorter trajectory lengths.",
    ],

    "protoss_10_vs_11": [
        "- Allied Team Agent Configuration : ten units consisting of a mix of Stalkers, Zealots, and Colossi (Stalkers are versatile ranged units, Zealots are close-range melee units, and Colossi are powerful long-range units in StarCraft 2).",
        "- Enemy Team Agent Configuration : eleven units consisting of a mix of Stalkers, Zealots, and Colossi (same as the allied team).",
        "- Situation Description : The situation involves the allied team and the enemy team engaging in combat, where victory is achieved by defeating all the enemies.",
        "- Objective : Defeat all enemy agents while ensuring as many allied agents as possible survive.",
        "* Important Notice : You should prefer the trajectory where our allies' health is preserved while significantly reducing the enemy's health. In similar situations, you should prefer shorter trajectory lengths.",
    ],

    "protoss_20_vs_20": [
        "- Allied Team Agent Configuration : twenty units consisting of a mix of Stalkers, Zealots, and Colossi (Stalkers are versatile ranged units, Zealots are close-range melee units, and Colossi are powerful long-range units in StarCraft 2).",
        "- Enemy Team Agent Configuration : twenty units consisting of a mix of Stalkers, Zealots, and Colossi (same as the allied team).",
        "- Situation Description : The situation involves the allied team and the enemy team engaging in combat, where victory is achieved by defeating all the enemies.",
        "- Objective : Defeat all enemy agents while ensuring as many allied agents as possible survive.",
        "* Important Notice : You should prefer the trajectory where our allies' health is preserved while significantly reducing the enemy's health. In similar situations, you should prefer shorter trajectory lengths.",
    ],

    "protoss_20_vs_23": [
        "- Allied Team Agent Configuration : twenty units consisting of a mix of Stalkers, Zealots, and Colossi (Stalkers are versatile ranged units, Zealots are close-range melee units, and Colossi are powerful long-range units in StarCraft 2).",
        "- Enemy Team Agent Configuration : twenty-three units consisting of a mix of Stalkers, Zealots, and Colossi (same as the allied team).",
        "- Situation Description : The situation involves the allied team and the enemy team engaging in combat, where victory is achieved by defeating all the enemies.",
        "- Objective : Defeat all enemy agents while ensuring as many allied agents as possible survive.",
        "* Important Notice : You should prefer the trajectory where our allies' health is preserved while significantly reducing the enemy's health. In similar situations, you should prefer shorter trajectory lengths.",
    ],

    "terran_5_vs_5": [
        "- Allied Team Agent Configuration : five units consisting of a mix of Marines, Marauders, and Medivacs (Marines are ranged units, Marauders are close-range units, and Medivacs are support units in StarCraft 2).",
        "- Enemy Team Agent Configuration : five units consisting of a mix of Marines, Marauders, and Medivacs (same as the allied team).",
        "- Situation Description : The situation involves the allied team and the enemy team engaging in combat, where victory is achieved by defeating all the enemies.",
        "- Objective : Defeat all enemy agents while ensuring as many allied agents as possible survive.",
        "* Important Notice : You should prefer the trajectory where our allies' health is preserved while significantly reducing the enemy's health. In similar situations, you should prefer shorter trajectory lengths.",
    ],

    "terran_10_vs_10": [
        "- Allied Team Agent Configuration : ten units consisting of a mix of Marines, Marauders, and Medivacs (Marines are ranged units, Marauders are close-range units, and Medivacs are support units in StarCraft 2).",
        "- Enemy Team Agent Configuration : ten units consisting of a mix of Marines, Marauders, and Medivacs (same as the allied team).",
        "- Situation Description : The situation involves the allied team and the enemy team engaging in combat, where victory is achieved by defeating all the enemies.",
        "- Objective : Defeat all enemy agents while ensuring as many allied agents as possible survive.",
        "* Important Notice : You should prefer the trajectory where our allies' health is preserved while significantly reducing the enemy's health. In similar situations, you should prefer shorter trajectory lengths.",
    ],

    "terran_10_vs_11": [
        "- Allied Team Agent Configuration : ten units consisting of a mix of Marines, Marauders, and Medivacs (Marines are ranged units, Marauders are close-range units, and Medivacs are support units in StarCraft 2).",
        "- Enemy Team Agent Configuration : eleven units consisting of a mix of Marines, Marauders, and Medivacs (same as the allied team).",
        "- Situation Description : The situation involves the allied team and the enemy team engaging in combat, where victory is achieved by defeating all the enemies.",
        "- Objective : Defeat all enemy agents while ensuring as many allied agents as possible survive.",
        "* Important Notice : You should prefer the trajectory where our allies' health is preserved while significantly reducing the enemy's health. In similar situations, you should prefer shorter trajectory lengths.",
    ],

    "terran_20_vs_20": [
        "- Allied Team Agent Configuration : twenty units consisting of a mix of Marines, Marauders, and Medivacs (Marines are ranged units, Marauders are close-range units, and Medivacs are support units in StarCraft 2).",
        "- Enemy Team Agent Configuration : twenty units consisting of a mix of Marines, Marauders, and Medivacs (same as the allied team).",
        "- Situation Description : The situation involves the allied team and the enemy team engaging in combat, where victory is achieved by defeating all the enemies.",
        "- Objective : Defeat all enemy agents while ensuring as many allied agents as possible survive.",
        "* Important Notice : You should prefer the trajectory where our allies' health is preserved while significantly reducing the enemy's health. In similar situations, you should prefer shorter trajectory lengths.",
    ],

    "terran_20_vs_23": [
        "- Allied Team Agent Configuration : twenty units consisting of a mix of Marines, Marauders, and Medivacs (Marines are ranged units, Marauders are close-range units, and Medivacs are support units in StarCraft 2).",
        "- Enemy Team Agent Configuration : twenty-three units consisting of a mix of Marines, Marauders, and Medivacs (same as the allied team).",
        "- Situation Description : The situation involves the allied team and the enemy team engaging in combat, where victory is achieved by defeating all the enemies.",
        "- Objective : Defeat all enemy agents while ensuring as many allied agents as possible survive.",
        "* Important Notice : You should prefer the trajectory where our allies' health is preserved while significantly reducing the enemy's health. In similar situations, you should prefer shorter trajectory lengths.",
    ],

    "zerg_5_vs_5": [
        "- Allied Team Agent Configuration : five units consisting of a mix of Zerglings, Roaches, and Hydras (Zerglings are fast melee units, Roaches are ranged units, and Hydras are long-range units in StarCraft 2).",
        "- Enemy Team Agent Configuration : five units consisting of a mix of Zerglings, Roaches, and Hydras (same as the allied team).",
        "- Situation Description : The situation involves the allied team and the enemy team engaging in combat, where victory is achieved by defeating all the enemies.",
        "- Objective : Defeat all enemy agents while ensuring as many allied agents as possible survive.",
        "* Important Notice : You should prefer the trajectory where our allies' health is preserved while significantly reducing the enemy's health. In similar situations, you should prefer shorter trajectory lengths.",
    ],

    "zerg_10_vs_10": [
        "- Allied Team Agent Configuration : ten units consisting of a mix of Zerglings, Roaches, and Hydras (Zerglings are fast melee units, Roaches are ranged units, and Hydras are long-range units in StarCraft 2).",
        "- Enemy Team Agent Configuration : ten units consisting of a mix of Zerglings, Roaches, and Hydras (same as the allied team).",
        "- Situation Description : The situation involves the allied team and the enemy team engaging in combat, where victory is achieved by defeating all the enemies.",
        "- Objective : Defeat all enemy agents while ensuring as many allied agents as possible survive.",
        "* Important Notice : You should prefer the trajectory where our allies' health is preserved while significantly reducing the enemy's health. In similar situations, you should prefer shorter trajectory lengths.",
    ],

    "zerg_10_vs_11": [
        "- Allied Team Agent Configuration : ten units consisting of a mix of Zerglings, Roaches, and Hydras (Zerglings are fast melee units, Roaches are ranged units, and Hydras are long-range units in StarCraft 2).",
        "- Enemy Team Agent Configuration : eleven units consisting of a mix of Zerglings, Roaches, and Hydras (same as the allied team).",
        "- Situation Description : The situation involves the allied team and the enemy team engaging in combat, where victory is achieved by defeating all the enemies.",
        "- Objective : Defeat all enemy agents while ensuring as many allied agents as possible survive.",
        "* Important Notice : You should prefer the trajectory where our allies' health is preserved while significantly reducing the enemy's health. In similar situations, you should prefer shorter trajectory lengths.",
    ],

    "zerg_20_vs_20": [
        "- Allied Team Agent Configuration : twenty units consisting of a mix of Zerglings, Roaches, and Hydras (Zerglings are fast melee units, Roaches are ranged units, and Hydras are long-range units in StarCraft 2).",
        "- Enemy Team Agent Configuration : twenty units consisting of a mix of Zerglings, Roaches, and Hydras (same as the allied team).",
        "- Situation Description : The situation involves the allied team and the enemy team engaging in combat, where victory is achieved by defeating all the enemies.",
        "- Objective : Defeat all enemy agents while ensuring as many allied agents as possible survive.",
        "* Important Notice : You should prefer the trajectory where our allies' health is preserved while significantly reducing the enemy's health. In similar situations, you should prefer shorter trajectory lengths.",
    ],

    "zerg_20_vs_23": [
        "- Allied Team Agent Configuration : twenty units consisting of a mix of Zerglings, Roaches, and Hydras (Zerglings are fast melee units, Roaches are ranged units, and Hydras are long-range units in StarCraft 2).",
        "- Enemy Team Agent Configuration : twenty-three units consisting of a mix of Zerglings, Roaches, and Hydras (same as the allied team).",
        "- Situation Description : The situation involves the allied team and the enemy team engaging in combat, where victory is achieved by defeating all the enemies.",
        "- Objective : Defeat all enemy agents while ensuring as many allied agents as possible survive.",
        "* Important Notice : You should prefer the trajectory where our allies' health is preserved while significantly reducing the enemy's health. In similar situations, you should prefer shorter trajectory lengths.",
    ],
}


def create_context(env_name, trajectory_x, trajectory_y):
    context = [
        "You are a helpful and honest judge of good game playing and progress in the StarCraft Multi-Agent Challenge game. Always answer as helpfully as possible, while being truthful.",
        "If you don't know the answer to a question, please don't share false information.",
        "I'm looking to have you evaluate a scenario in the StarCraft Multi-Agent Challenge. Your role will be to assess how much the actions taken by multiple agents in a given situation have contributed to achieving victory.",
        "",
        "The basic information for the evaluation is as follows.",
        "",
        f"- Scenario : {env_name}",
    ]
    context += descriptions[env_name]
    context += [
        "",
        "I will provide you with two trajectories, and you should select the better trajectory based on the outcomes of these trajectories. Regarding the trajectory, it will inform you about the final states, and you should select the better case based on these two trajectories.",
        "",
    ]

    ally_healths_x, enemy_healths_x, actives_x = trajectory_x
    last_step_x = torch.sum(actives_x)
    ally_healths_x = ally_healths_x[last_step_x-1]
    enemy_healths_x = enemy_healths_x[last_step_x-1]

    context += [
        "[Trajectory 1]",
        "1. Final State Information",
        f"    1) Allied Agents Health : {', '.join([f'{x:.3f}' for x in ally_healths_x])}",
        f"    2) Enemy Agents Health : {', '.join([f'{x:.3f}' for x in enemy_healths_x])}",
        f"    3) Number of Allied Deaths : {len(ally_healths_x) - torch.sum(ally_healths_x > 0)}",
        f"    4) Number of Enemy Deaths : {len(enemy_healths_x) - torch.sum(enemy_healths_x > 0)}",
        f"    5) Total Remaining Health of Allies : {torch.sum(ally_healths_x):.3f}",
        f"    6) Total Remaining Health of Enemies : {torch.sum(enemy_healths_x):.3f}",
        f"2. Total Number of Steps : {last_step_x}",
    ]

    ally_healths_y, enemy_healths_y, actives_y = trajectory_y
    last_step_y = torch.sum(actives_y)
    ally_healths_y = ally_healths_y[last_step_y-1]
    enemy_healths_y = enemy_healths_y[last_step_y-1]

    context += [
        "",
        "[Trajectory 2]",
        "1. Final State Information",
        f"    1) Allied Agents Health : {', '.join([f'{x:.3f}' for x in ally_healths_y])}",
        f"    2) Enemy Agents Health : {', '.join([f'{x:.3f}' for x in enemy_healths_y])}",
        f"    3) Number of Allied Deaths : {len(ally_healths_y) - torch.sum(ally_healths_y > 0)}",
        f"    4) Number of Enemy Deaths : {len(enemy_healths_y) - torch.sum(enemy_healths_y > 0)}",
        f"    5) Total Remaining Health of Allies : {torch.sum(ally_healths_y):.3f}",
        f"    6) Total Remaining Health of Enemies : {torch.sum(enemy_healths_y):.3f}",
        f"2. Total Number of Steps : {last_step_y}",
    ]

    context = "\n".join(context).strip()
    return context


def get_llm_output(env_name, id, context):
    prompt = [
        "Your task is to inform which one is better between [Trajectory1] and [Trajectory2] based on the information mentioned above. For example, if [Trajectory 1] seems better, output #1, and if [Trajectory 2] seems better, output #2. If it's difficult to judge or they seem similar, please output #0.",
        "* Important : Generally, it is considered better when fewer allied agents are killed or injured while inflicting more damage on the enemy.",
        "",
        "Omit detailed explanations and just provide the answer."
    ]
    prompt = "\n".join(prompt).strip()
    
    return {
        "custom_id": f"SMU-{env_name}-{id}", 
        "method": "POST", 
        "url": "/v1/chat/completions", 
        "body": {
            "model": "gpt-4o", 
            "messages": [
                {"role": "user", "content": context + "\n\n" + prompt}
            ],
        }
    }


def get_results(env_name, results):
    path = f"dataset/llm_{env_name}.jsonl"
    with open(path, "w") as f:
        f.write("\n".join([json.dumps(result) for result in results]))
    print(f"Saved results to {path}")
    

def load_h5py_data(env_name):
    with h5py.File(f"dataset/{env_name}.h5", "r") as f:
        states_x = f["states_x"]
        states_y = f["states_y"]
        actives_x = f["actives_x"]
        actives_y = f["actives_y"]
        
        obs_x = f["obs_x"]
        n_samples, n_steps, n_agents, _ = obs_x.shape

        if env_name in SMACV1_ENV_NAMES:
            states_x = torch.from_numpy(states_x[:]).reshape(n_samples, n_steps, n_agents, -1)
            states_y = torch.from_numpy(states_y[:]).reshape(n_samples, n_steps, n_agents, -1)
        else:
            states_x = torch.from_numpy(states_x[:])
            states_y = torch.from_numpy(states_y[:])

        actives_x = torch.from_numpy(actives_x[:])
        actives_y = torch.from_numpy(actives_y[:])

        states_x = states_x[:, :-1]
        states_y = states_y[:, :-1]

    return states_x, states_y, actives_x, actives_y


def read_batch_state_smacv1(states, env: SMACWrapperV1):
    n_samples, n_steps, n_agents, _ = states.shape

    self = env.env
    
    enemy_feats_dim = self.get_state_enemy_feats_size()
    ally_feats_dim = self.get_state_ally_feats_size()

    ally_state_pos = np.prod(ally_feats_dim)
    enemy_state_pos = ally_state_pos + np.prod(enemy_feats_dim)

    ally_state = states[..., :ally_state_pos].view(n_samples, n_steps, n_agents, *ally_feats_dim)
    enemy_state = states[..., ally_state_pos:enemy_state_pos].view(n_samples, n_steps, n_agents, *enemy_feats_dim)
    
    ally_healths = ally_state[..., 5]
    first_ally_healths = ally_healths[..., 0, :]
    last_ally_healths = ally_healths[..., -1, :1]
    ally_healths = torch.cat([last_ally_healths, first_ally_healths], dim=-1)
    enemy_healths = enemy_state[..., 5].mean(2)

    return ally_healths, enemy_healths


def read_batch_state_smacv2(states, env: SMACWrapperV2):
    n_samples, n_steps, _ = states.shape

    self = env.env.env
    
    nf_al = self.get_ally_num_attributes()
    nf_en = self.get_enemy_num_attributes()

    ally_state_pos = self.n_agents * nf_al
    enemy_state_pos = ally_state_pos + self.n_enemies * nf_en

    ally_state = states[..., :ally_state_pos].view(n_samples, n_steps, self.n_agents, nf_al)
    enemy_state = states[..., ally_state_pos:enemy_state_pos].view(n_samples, n_steps, self.n_enemies, nf_en)

    ally_healths = ally_state[..., 0]
    enemy_healths = enemy_state[..., 0]

    return ally_healths, enemy_healths


def main_smacv1(env_name, seed=0):
    states_x, states_y, actives_x, actives_y = load_h5py_data(env_name)
    env = SMACWrapperV1(env_name, seed)

    ally_healths_x, enemy_healths_x = read_batch_state_smacv1(states_x, env)
    ally_healths_y, enemy_healths_y = read_batch_state_smacv1(states_y, env)

    results = []
    for i in tqdm(range(len(states_x)), f"Reading {env_name} ...", ncols=80, mininterval=0.5):
        trajectory_x = (ally_healths_x[i], enemy_healths_x[i], actives_x[i])
        trajectory_y = (ally_healths_y[i], enemy_healths_y[i], actives_y[i])
        context = create_context(env_name, trajectory_x, trajectory_y)
        result = get_llm_output(env_name, i, context)
        results.append(result)
    
    get_results(env_name, results)
    env.close()


def main_smacv2(env_name, seed=0):
    states_x, states_y, actives_x, actives_y = load_h5py_data(env_name)
    env = SMACWrapperV2(env_name, seed)

    ally_healths_x, enemy_healths_x = read_batch_state_smacv2(states_x, env)
    ally_healths_y, enemy_healths_y = read_batch_state_smacv2(states_y, env)

    results = []
    for i in tqdm(range(len(states_x)), f"Reading {env_name} ...", ncols=80, mininterval=0.5):
        trajectory_x = (ally_healths_x[i], enemy_healths_x[i], actives_x[i])
        trajectory_y = (ally_healths_y[i], enemy_healths_y[i], actives_y[i])
        context = create_context(env_name, trajectory_x, trajectory_y)
        result = get_llm_output(env_name, i, context)
        results.append(result)
    
    get_results(env_name, results)
    env.close()


def main(env_name):
    if env_name in SMACV1_ENV_NAMES:
        main_smacv1(env_name)
    elif env_name in SMACV2_ENV_NAMES:
        main_smacv2(env_name)
    else:
        raise ValueError(f"Invalid environment name: {env_name}")


@dataclass
class Args:
    env_name: str = "protoss_5_vs_5"


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args.env_name)