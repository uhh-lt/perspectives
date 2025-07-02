from enum import Enum
from pathlib import Path
from typing import Literal
import pandas as pd
import typer
from setfit import sample_dataset
from datasets import Dataset

app = typer.Typer()


class SupportedDatasets(Enum):
    SPOTIFY = "spotify"
    AMAZON = "amazon"
    NEWSGROUPS = "newsgroups"

    def dataset_path(self, variant: Literal["train", "test"]) -> Path:
        base_path = Path("../datasets")
        if self == SupportedDatasets.SPOTIFY:
            return base_path / f"spotify/data/spotify_{variant}.parquet"
        elif self == SupportedDatasets.AMAZON:
            return base_path / f"amazon/data/amazon_{variant}.parquet"
        elif self == SupportedDatasets.NEWSGROUPS:
            return base_path / f"20/newsgroups/data/newsgroups_{variant}.parquet"
        else:
            raise ValueError(f"Unsupported dataset: {self}")


@app.command()
def sample_fewshot_examples(
    dataset: SupportedDatasets = typer.Option(
        ..., help="Dataset to modify (spotify, amazon, newsgroup)"
    ),
    text_column: str = typer.Option(..., help="Name of the text column"),
    category_column: str = typer.Option(..., help="Name of the category column"),
    output_dir: Path = typer.Option(..., help="Path to save the prepared dataset"),
):
    """
    Sample a dataset for few-shot training.
    This function samples N examples from each category in the specified dataset.
    This is done 10 times for for multiple training runs.
    """

    # Ensure the output path is a directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load dataset
    if not dataset.dataset_path("train").exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset.dataset_path('train')}"
        )
    df = pd.read_parquet(dataset.dataset_path("train"))
    if category_column not in df.columns:
        raise ValueError(f"Column '{category_column}' not found in the dataset.")
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the dataset.")

    # prepare dataset
    train_dataset = Dataset.from_dict(
        {
            "text": df[text_column].tolist(),
            "label": df[category_column].tolist(),
        }
    )

    # Create splits for different shot sizes
    for N in [2, 4, 8, 16]:
        # Create splits for 10 runs
        datasets = []
        for seed in [7, 42, 69, 96, 123, 404, 500, 666, 1024, 1337]:
            sampled_dataset = sample_dataset(
                train_dataset, label_column="label", num_samples=N, seed=seed
            )
            datasets.append(sampled_dataset)

        # Combine datasets into a single pandas DataFrame
        combined_texts = []
        combined_labels = []
        for sampled_dataset in datasets:
            combined_texts.extend(sampled_dataset["text"])
            combined_labels.extend(sampled_dataset["label"])

        output_df = pd.DataFrame(
            {
                text_column: combined_texts,
                category_column: combined_labels,
            }
        )

        # Save the dataframe as parquet
        output_file = output_dir / f"{dataset.value}_shot={N}_{category_column}.parquet"
        output_df.to_parquet(output_file, index=False)

    typer.echo(f"Sampled datasets for {dataset.value}!")


if __name__ == "__main__":
    app()
