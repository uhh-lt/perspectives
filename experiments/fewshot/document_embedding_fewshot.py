from pathlib import Path
from typing import Optional
from setfit import SetFitModel, Trainer, TrainingArguments
import typer
import pandas as pd
import os
import numpy as np
from datasets import load_from_disk, Dataset
from peft import LoraConfig, TaskType, EvaConfig
from experiments.clustering_evaluation import (
    evaluate_2d_classification,
    evaluate_clustering,
)
from experiments.models import SupportedModel

# Ensure that CUDA_VISIBLE_DEVICES is set to use the first GPU
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    raise EnvironmentError(
        "CUDA_VISIBLE_DEVICES environment variable is not set. Please set it to the GPU you want to use."
    )

app = typer.Typer()


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description.strip()}\nQuery: {remove_linebreaks(query)}"


def remove_linebreaks(text: str) -> str:
    """Remove line breaks from text."""
    return " ".join(text.splitlines()).strip()


def get_embedding_output_path(
    output_dir: Path,
    train_dataset_path: Path,
    num_shots: int,
    uses_instruction: bool,
    run: int,
) -> Path:
    return (
        Path(output_dir)
        / f"{Path(train_dataset_path).name}_shot{num_shots}{'-inst' if uses_instruction else ''}_run{run}.npz"
    )


@app.command()
def document_embedding_fewshot(
    # train dataset params
    train_dataset_path: Path = typer.Option(..., help="Path to the dataset train file"),
    run: int = typer.Option(..., help="Run number for the experiment (0-9)"),
    num_shots: int = typer.Option(
        ..., help="Number of shots for few-shot learning (default: 1)"
    ),
    # test dataset params
    test_dataset_path: Path = typer.Option(..., help="Path to the dataset test file"),
    test_dataset_text_column: str = typer.Option(
        ..., help="Name of the text column in the test dataset"
    ),
    test_dataset_label_column: str = typer.Option(
        ..., help="Name of the label column in the test dataset"
    ),
    # model params
    model_name: SupportedModel = typer.Option(
        ..., help="Name of the embedding model to use"
    ),
    model_name_short: str = typer.Option(
        ..., help="Short Name of the embedding model to use"
    ),
    instruction: Optional[str] = typer.Option(
        None, help="Instruction for the embedding model (optional)"
    ),
    # training params
    batch_size: int = typer.Option(8, help="Batch size for training"),
    batch_size_head: int = typer.Option(
        16, help="Batch size for the head (default: 16)"
    ),
    num_epochs: int = typer.Option(
        1, help="Number of epochs for training (default: 1)"
    ),
    num_epochs_head: int = typer.Option(
        23, help="Number of epochs for the head (default: 23)"
    ),
    # adapter params
    r: int = typer.Option(8, help="Rank of the LoRA adapter (default: 8)"),
    lora_alpha: int = typer.Option(
        16, help="Alpha value for the LoRA adapter (default: 16)"
    ),
    lora_dropout: float = typer.Option(
        0.1, help="Dropout rate for the LoRA adapter (default: 0.1)"
    ),
    # learning rates
    body_learning_rate1: float = typer.Option(..., help="Learning rate for the model"),
    body_learning_rate2: float = typer.Option(
        ..., help="Learning rate for the model (alternative)"
    ),
    seed: int = typer.Option(..., help="Random seed for reproducibility"),
    # output params
    output_dir: Path = typer.Option(..., help="Path to the output directory"),
):
    """
    Generate embeddings for each document in the dataset.
    """

    # INPUT VALIDATION

    # Check if the training dataset path exists
    if not Path(train_dataset_path).exists():
        raise ValueError(f"Dataset path does not exist: {train_dataset_path}")

    if not Path(test_dataset_path).exists():
        raise ValueError(f"Test dataset path does not exist: {test_dataset_path}")

    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check if num_shots is a valid value
    if num_shots not in [2, 4, 8, 16]:
        raise ValueError(
            f"Invalid number of shots: {num_shots}. Must be one of [2, 4, 8, 16]."
        )

    # Check if run is a valid value
    if run not in range(10):
        raise ValueError(f"Invalid run number: {run}. Must be one of [0-9].")

    # DATASET LOADING

    # Load the test dataset
    test_df = pd.read_parquet(test_dataset_path)
    assert test_dataset_text_column in test_df.columns, (
        f"Column '{test_dataset_text_column}' not found in the test dataset."
    )
    # filter out rows where text is None, empty or starts with "Error:"
    mask = test_df[test_dataset_text_column].isnull() | (test_df[test_dataset_text_column].str.strip() == "") | test_df[test_dataset_text_column].str.startswith("Error:")
    test_df = test_df[~mask]

    label_names = test_df[test_dataset_label_column].unique().tolist()
    label2id = {name: idx for idx, name in enumerate(label_names)}
    eval_dataset = Dataset.from_dict(
        {
            "text": test_df[test_dataset_text_column].tolist(),
            "label": test_df[test_dataset_label_column].tolist(),
        }
    )
    # prepend the instruction
    if instruction is not None:
        eval_dataset = eval_dataset.map(
            lambda x: {"text": get_detailed_instruct(instruction, x["text"])}
        )
    # convert the labels str -> int
    eval_dataset = eval_dataset.map(
        lambda x: {"label": label2id[x["label"]]},
    )

    # Load the training dataset
    training_data = load_from_disk(train_dataset_path)
    train_dataset = training_data[f"shot={num_shots}_run={run}"]
    assert isinstance(train_dataset, Dataset)
    assert list(train_dataset.features.keys()) == ["text", "label"], (
        f"Expected train_dataset features to be ['text', 'label'], got {train_dataset.features}"
    )
    # filter out rows where text is None, empty or starts with "Error:"
    train_dataset = train_dataset.filter(lambda x: x["text"] is not None and x["text"].strip() != "" and not x["text"].startswith("Error:"))
    # prepend the instruction
    if instruction is not None:
        train_dataset = train_dataset.map(
            lambda x: {"text": get_detailed_instruct(instruction, x["text"])}
        )
    # convert the labels str -> int
    train_dataset = train_dataset.map(
        lambda x: {"label": label2id[x["label"]]},
    )

    # MODEL TRAINING

    # load model
    model = SetFitModel.from_pretrained(
        model_name.value,
        device="cuda:0",
        trust_remote_code=True,
        use_differentiable_head=True,
        head_params={"out_features": len(label_names)},
    )

    eva_config = EvaConfig()

    # add adapter to the model
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        use_rslora=True,  # Use RSLora for better performance
        init_lora_weights="eva",
        eva_config=eva_config,
    )
    assert model.model_body is not None, "Model body is not initialized."
    model.model_body.add_adapter(lora_config)

    # init training
    args = TrainingArguments(
        batch_size=(batch_size, batch_size_head),
        num_epochs=(num_epochs, num_epochs_head),
        output_dir=str(output_dir),
        sampling_strategy="undersampling",  # options: oversampling, undersampling, unique
        save_strategy="no",  # options: no epoch steps
        eval_strategy="no",  # options: no epoch steps
        seed=seed,
        end_to_end=True,
        load_best_model_at_end=False,
        body_learning_rate=(body_learning_rate1, body_learning_rate2),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )

    # train model
    trainer.train()

    # CLASSIFICATION EVALUATION

    setfit_evaluation_results = trainer.evaluate(dataset=eval_dataset)

    # TEST DATA EMBEDDING

    embedding_model = model.model_body
    embedding_model.eval()

    test_texts = test_df[test_dataset_text_column].tolist()
    if instruction is not None:
        test_texts = [get_detailed_instruct(instruction, text) for text in test_texts]

    embeddings = embedding_model.encode(
        test_texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        device="cuda:0",
    )

    # Store embeddings in a new file
    output_file = get_embedding_output_path(
        output_dir=output_dir,
        train_dataset_path=train_dataset_path,
        num_shots=num_shots,
        uses_instruction=instruction is not None,
        run=run,
    )
    np.savez_compressed(
        output_file,
        embeddings=embeddings,
    )
    print(f"Embeddings saved to {output_file}")

    # CLUSTERING EVALUATION

    labels = test_df[test_dataset_label_column].map(label2id).astype(int).tolist()

    # Check if embeddings and labels have the same length
    if len(embeddings) != len(labels):
        raise ValueError(
            "The number of embeddings does not match the number of labels. "
            f"Embeddings: {len(embeddings)}, Labels: {len(labels)}"
        )

    # Evaluate clustering
    print("Evaluating clustering...")
    clustering_results, best_params = evaluate_clustering(embeddings, np.array(labels))

    # Evaluate 2D classification
    print("Evaluating 2D classification...")
    classification_results = evaluate_2d_classification(embeddings, np.array(labels))

    # Save evaluation results
    results = {
        "dataset": train_dataset_path.stem.split("_")[0],
        "label_column": test_dataset_label_column,
        "text_column": test_dataset_text_column.split("-")[-1]
        if "-" in test_dataset_text_column
        else test_dataset_text_column,
        "model": model_name_short,
        "num_shots": num_shots,
        "run": run,
        "instruction": instruction is not None,
        **classification_results,
        **clustering_results,
        **setfit_evaluation_results,
    }
    pd.DataFrame([results]).to_csv(
        output_dir / output_file.with_suffix(".csv"), index=False
    )


if __name__ == "__main__":
    app()
