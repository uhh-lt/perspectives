from pathlib import Path
import pandas as pd
from tqdm import tqdm
import typer
from dotenv import load_dotenv
import os
from litellm import batch_completion
import re

load_dotenv("../.env")

BATCH_SIZE = 32
VLLM_API_KEY = os.getenv("VLLM_API_KEY")
VLLM_PORT = os.getenv("VLLM_EXPOSED")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME")
VLLM_MAX_TOKENS_TO_GENERATE = int(os.getenv("VLLM_MAX_TOKENS_TO_GENERATE", 1024))

if VLLM_API_KEY is None:
    raise ValueError("VLLM_API_KEY environment variable is not set.")
if VLLM_PORT is None:
    raise ValueError("VLLM_EXPOSED environment variable is not set.")
if VLLM_MODEL_NAME is None:
    raise ValueError("VLLM_MODEL_NAME environment variable is not set.")
if VLLM_MAX_TOKENS_TO_GENERATE <= 0:
    raise ValueError(
        "VLLM_MAX_TOKENS_TO_GENERATE must be a positive integer. "
        f"Current value: {VLLM_MAX_TOKENS_TO_GENERATE}"
    )


app = typer.Typer()


def parse_response(response: str) -> str:
    """
    Parse the response from the LLM to remove any kind of markdown formatting and newlines.
    """
    # Remove bold (**text** or __text__)
    response = re.sub(r"\*\*(.*?)\*\*|__(.*?)__", "\1\2", response)

    # Remove italic (*text* or _text_)
    response = re.sub(r"\*(.*?)\*|_(.*?)_", "\1\2", response)

    # Remove headings (# Heading)
    response = re.sub(r"^#+\s*(.*)$", "\1", response, flags=re.MULTILINE)

    # Remove excess newlines and whitespace
    response = re.sub(r"\s+", " ", response).strip()

    return response


def modify_dataset(
    dataset_path: Path,
    text_column: str,
    new_column_name: str,
    system_prompt: str,
    user_prompt: str,
):
    # Check if the dataset path exists
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    # Load the dataset
    dataset = pd.read_parquet(dataset_path)

    # Check if the text column exists
    if text_column not in dataset.columns:
        raise ValueError(f"Column '{text_column}' not found in the dataset.")

    # Create the new column if it does not exist
    if new_column_name not in dataset.columns:
        dataset[new_column_name] = None
    else:
        # If the column already exists, print a warning
        typer.echo(
            f"Warning: Column '{new_column_name}' already exists in the dataset. "
            "It will be overwritten."
        )

    # Convert to a python list
    dataset = dataset.to_dict(orient="records")

    # Check that user_prompt contains the placeholder for text
    if "{text}" not in user_prompt:
        raise ValueError("User prompt must contain the placeholder '{text}'.")

    # I need to iterate over the dataset in batches, then apply the prompt
    batch_size = BATCH_SIZE
    if len(dataset) < batch_size:
        batch_size = len(dataset)
    print(f"Processing dataset in batches of size {batch_size}")

    # Iterate batchwise over the dataset
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    new_dataset = []
    print(f"Total batches to process: {num_batches}")
    for i in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset[start_idx:end_idx]

        # prepare messages for batch processing
        batch_messages = []
        for idx, row in enumerate(batch):
            text = row[text_column]
            batch_messages.append(
                [
                    {
                        "role": "system",
                        "content": system_prompt.strip(),
                    },
                    {
                        "role": "user",
                        "content": user_prompt.format(text=text),
                    },
                ]
            )

        # Call the LLM in batch mode
        batch_responses = batch_completion(
            model=f"hosted_vllm/{VLLM_MODEL_NAME}",
            messages=batch_messages,
            max_tokens=VLLM_MAX_TOKENS_TO_GENERATE,
            base_url=f"http://localhost:{VLLM_PORT}/v1",
            api_key=VLLM_API_KEY,
        )

        # Apply the parsed responses to the new column in the batch
        for idx, response in enumerate(batch_responses):
            row = batch[idx]

            if response is None:
                row[new_column_name] = "Error: No response from LLM"
                continue

            # Ensure that the response has choices field
            if not hasattr(response, "choices") or not response.choices:
                row[new_column_name] = "Error: No choices in response"
                continue

            row[new_column_name] = parse_response(
                response.choices[0].message.content.strip()
            )

        # Append the modified batch to the new dataset
        new_dataset.extend(batch)

    # Save the modified dataset to a new file
    dataset = pd.DataFrame(new_dataset)
    dataset.to_parquet(dataset_path)
    print(f"Modified dataset saved to {dataset_path}")


@app.command()
def document_modification(
    dataset_path: Path = typer.Option(
        ...,
        help="Path to the dataset file",
    ),
    text_column: str = typer.Option(..., help="Name of the text column in the dataset"),
    column_name: str = typer.Option(..., help="Name of the new column to create"),
    system_prompt: str = typer.Option(..., help="System prompt for the LLM"),
    user_prompt: str = typer.Option(
        ..., help="User prompt for the LLM (must contain '{text}')"
    ),
):
    """
    Modify a dataset by applying an LLM to each document in a specified column.
    """
    modify_dataset(
        dataset_path=dataset_path,
        text_column=text_column,
        new_column_name=column_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt + "\n\n{text}",
    )


if __name__ == "__main__":
    app()
