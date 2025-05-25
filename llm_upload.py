import os
from openai import OpenAI
from config import SMACV1_ENV_NAMES, SMACV2_ENV_NAMES


def upload(env_name):
    path = f"dataset/llm_{env_name}.jsonl"
    
    batch_input_file = client.files.create(
        file=open(path, "rb"),
        purpose="batch"
    )
    print("batch_input_file:", batch_input_file)

    batch_input_file_id = batch_input_file.id
    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": path
        }
    )
    print("batch:", batch)

    batch = client.batches.retrieve(batch.id)
    print("batch:", batch)


def main():
    for env_name in SMACV1_ENV_NAMES + SMACV2_ENV_NAMES:
        # if "_10" in env_name or "_20" in env_name:
        print(f"Uploading {env_name} ...")
        upload(env_name)


if __name__ == "__main__":
    client = OpenAI(
        organization='org-J57WTuqjaC5PSIhlQhmfgmd9',
        project='proj_ehHc8tNJLqKqF69ZOzWTByr7',
    )
    
    main()