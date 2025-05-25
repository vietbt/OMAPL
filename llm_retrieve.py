import os
from openai import OpenAI


def retrieve(batch_id):
    batch = client.batches.retrieve(batch_id)
    original_file_path = batch.metadata["description"]
    print("Reading:", original_file_path)
    output_file_path = original_file_path.replace(".jsonl", "_out.jsonl")
    content = client.files.content(batch.output_file_id)
    content.write_to_file(output_file_path)


def main():
    batch_ids = [
        "batch_678ccd4130d4819091aebc30e4aed5d9",
        "batch_678ccd3ee74c819099b2274b46f0fe38",
        "batch_678ccd3c74688190b13f978fc17a0bf1",
        "batch_678ccd39c0188190920fdb434059678b",
        "batch_678ccd3754308190ac0514bfd572ef44",
        "batch_678ccd3512cc81909b3799407c2fae4a",
        "batch_678ccd326e0c8190bc03f27146013267",
        "batch_678ccd2ec2748190850a9aa45d7cd95c",
        "batch_678ccd2c0a5c8190987f3113bd7f6460",
        "batch_678ccd29c8bc8190bcffef025e0ed84d",
        "batch_678ccd2773048190ae39c5bf4d8050a4",
        "batch_678ccd24bf7081909fdbc8feddc9a1b2",
        "batch_678ccd221d348190a9017f941d38cf78",
        "batch_678ccd1f9f908190bd84d398027a8c82",
        "batch_678ccd1d0f648190b0162a8cac677b76",
        "batch_678ccd1a55a48190ab2674cbe89c6ebe",
        "batch_678ccd17bf9481908ce4f322c95bd4ea",
        "batch_678ccd1425188190b60eea24f520c4ee",
        "batch_678ccd11222881908c822b98aa1f788c",
    ]
    for batch_id in batch_ids:
        retrieve(batch_id)


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] =  "sk-proj-..."

    client = OpenAI(
        organization='org-J57WTuqjaC5PSIhlQhmfgmd9',
        project='proj_ehHc8tNJLqKqF69ZOzWTByr7',
    )
    
    main()