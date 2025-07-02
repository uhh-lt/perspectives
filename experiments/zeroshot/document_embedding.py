from enum import Enum
from pathlib import Path
from typing import List, Optional
import typer
import pandas as pd
import os
import numpy as np

# Ensure that CUDA_VISIBLE_DEVICES is set to use the first GPU
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    raise EnvironmentError(
        "CUDA_VISIBLE_DEVICES environment variable is not set. Please set it to the GPU you want to use."
    )

app = typer.Typer()

BATCH_SIZE = 32  # Default batch size for embedding generation


class SupportedModel(Enum):
    MULTILINGUAL_E5_LARGE_INSTRUCT = "intfloat/multilingual-e5-large-instruct"


def generate_embeddings(
    model_name: SupportedModel, instruction: Optional[str], texts: List[str]
) -> List[List[float]]:
    if model_name == SupportedModel.MULTILINGUAL_E5_LARGE_INSTRUCT:
        return generate_embeddings_multi_e5_large_inst(instruction, texts)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def generate_embeddings_multi_e5_large_inst(
    instruction: Optional[str],
    texts: List[str],
) -> List[List[float]]:
    from sentence_transformers import SentenceTransformer

    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery: {query}"

    documents = texts
    if instruction is not None:
        documents = [get_detailed_instruct(instruction, text) for text in texts]

    model = SentenceTransformer(
        SupportedModel.MULTILINGUAL_E5_LARGE_INSTRUCT.value, device="cuda:0"
    )

    embeddings = model.encode(
        documents,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        device="cuda:0",
    )
    return embeddings.tolist()


@app.command()
def document_embedding(
    dataset_path: str = typer.Option(..., help="Path to the dataset file"),
    text_column: str = typer.Option(
        ..., help="Name of the column containing the text documents"
    ),
    output_path: str = typer.Option(..., help="Path to save the output embeddings"),
    model_name: SupportedModel = typer.Option(
        ..., help="Name of the embedding model to use"
    ),
    instruction: Optional[str] = typer.Option(
        None, help="Instruction for the embedding model (optional)"
    ),
):
    """
    Generate embeddings for each document in the dataset.
    """

    # Check if the dataset path exists
    if not Path(dataset_path).exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    # Check if the output path exists, if not create it
    op = Path(output_path)
    if not op.parent.exists():
        op.parent.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    dataset = pd.read_parquet(dataset_path)

    # Check if the text column exists
    if text_column not in dataset.columns:
        raise ValueError(f"Column '{text_column}' not found in the dataset.")

    # Convert to a python list
    texts = dataset[text_column].tolist()

    # Generate embeddings
    embeddings = generate_embeddings(
        model_name=model_name,
        instruction=instruction,
        texts=texts,
    )

    # Store embeddings in a new file
    np.savez_compressed(
        op.with_suffix(".npz"),
        embeddings=embeddings,
    )


if __name__ == "__main__":
    app()
