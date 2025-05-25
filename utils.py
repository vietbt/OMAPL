import json
import h5py
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from tensordict.tensordict import TensorDict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_h5py_data(env_name):
    with h5py.File(f"dataset/{env_name}.h5", "r") as f:
        dataset = {}
        for k, v in f.items():
            dataset[k] = torch.from_numpy(v[:])
        dataset = TensorDict(dataset, batch_size=len(dataset["labels"]))
    return dataset


def load_mujoco_dataset(env_name, batch_size, use_llm=False):
    assert not use_llm
    dataset = load_h5py_data(env_name)
    
    st_dim = dataset["states_x"].shape[-1]
    ob_dim = dataset["obs_x"].shape[-1]
    ac_dim = dataset["actions_x"].shape[-1]
    n_agents = dataset["obs_x"].shape[-2]

    dataset = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: x, shuffle=True, drop_last=True, pin_memory=True)
    return dataset, st_dim, ob_dim, ac_dim, n_agents


def load_smac_dataset(env_name, batch_size, use_llm=False):
    dataset = load_h5py_data(env_name)
    
    st_dim = dataset["states_x"].shape[-1]
    ob_dim = dataset["obs_x"].shape[-1]
    ac_dim = dataset["avails_x"].shape[-1]
    n_agents = dataset["obs_x"].shape[-2]

    if use_llm:
        llm_labels = dataset["labels"].clone()
        llm_path = f"dataset/llm_{env_name}_out.jsonl"
        with open(llm_path, "r") as f:
            for line in f:
                data = json.loads(line)
                line_id = int(data["custom_id"].split("-")[-1])
                content = data["response"]["body"]["choices"][0]["message"]["content"]
                if content in ["#1", "#2"]:
                    llm_labels[line_id] = content == "#1"
                else:
                    print("line_id:", data["custom_id"], "content:", content)

        # accuracy = (llm_labels == dataset["labels"]).float().mean()
        # print("accuracy:", accuracy)
        dataset["labels"] = llm_labels


    dataset = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: x, shuffle=True, drop_last=True, pin_memory=True)
    return dataset, st_dim, ob_dim, ac_dim, n_agents


def is_trained(algo, env_name, seed):
    task_name = f"{algo}_{env_name}_seed{seed}"
    path = f"saved_models/{task_name}.pt"
    try:
        torch.load(path, map_location="cpu", weights_only=True)
        return True
    except:
        print(f"Failed to load {path} ...")
        return False