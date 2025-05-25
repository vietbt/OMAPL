import os
import json
from pprint import pprint
from config import SMACV1_ENV_NAMES, SMACV2_ENV_NAMES, MAMUJOCO_ENV_NAMES

# path = "dataset/llm_2c_vs_64zg_out.jsonl"

for env_name in SMACV1_ENV_NAMES + SMACV2_ENV_NAMES:
    path = f"dataset/llm_{env_name}_out.jsonl"
    total_completion_tokens = 0
    total_prompt_tokens = 0
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)["response"]["body"]["usage"]
            total_completion_tokens += data["completion_tokens"]
            total_prompt_tokens += data["prompt_tokens"]
    # print(f"{env_name}: {total_completion_tokens} completion tokens, {total_prompt_tokens} prompt tokens")
    print(f"{env_name}\t{total_completion_tokens}\t{total_prompt_tokens}")