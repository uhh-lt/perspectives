from pathlib import Path
import pandas as pd
import typer
from datasets import Dataset, DatasetDict

app = typer.Typer()


@app.command()
def convert_to_dataset(
    dataset_dir: Path = typer.Option(
        ..., help="Path to the dataset file (Parquet format)"
    ),
    text_column: str = typer.Option(..., help="Name of the text column"),
    label_column: str = typer.Option(..., help="Name of the label column"),
    output_path: Path = typer.Option(..., help="Path to save the prepared dataset"),
):
    """
    Sample a dataset for few-shot training.
    This function samples N examples from each category in the specified dataset.
    This is done 10 times for for multiple training runs.
    """

    # Detect all datasets ending with {label_column}.parquet in the given directory
    input_dir = Path(dataset_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input path must be a directory: {input_dir}")

    dataset_paths = list(input_dir.glob(f"*{label_column}.parquet"))
    if not dataset_paths:
        raise FileNotFoundError(
            f"No dataset files found in {input_dir} matching '*{label_column}.parquet'"
        )

    print(
        f"Found {len(dataset_paths)} dataset files matching '*{label_column}.parquet' in {input_dir}"
    )

    training_dataset = DatasetDict()
    for dataset_path in dataset_paths:
        # parse the shots from the filename which includes shot={N}
        try:
            N = int(dataset_path.stem.split("_shot=")[-1].split("_")[0])
        except (IndexError, ValueError):
            raise ValueError(
                f"Could not parse the number of shots from the filename: {dataset_path.stem}"
            )
        print(f"Processing dataset: {dataset_path} with {N}-shot learning")

        # load dataset
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        df = pd.read_parquet(dataset_path)
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in the dataset.")
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the dataset.")

        # Ensure the dataset has the correct length
        unique_labels = df[label_column].unique()
        expected_length = len(unique_labels) * N * 10
        if len(df) != expected_length:
            raise ValueError(
                f"Dataset length {len(df)} does not match expected length {expected_length}!"
            )

        # split the dataset into 10 chunks, where each chunk has the length of N * len(unique_labels)
        start_index = 0
        datasets = []
        for i in range(10):
            end_index = start_index + N * len(unique_labels)
            chunk = df.iloc[start_index:end_index]

            train_dataset = Dataset.from_dict(
                {
                    "text": chunk[text_column].tolist(),
                    "label": chunk[label_column].tolist(),
                }
            )
            datasets.append(train_dataset)

            start_index = end_index

        # Combine splits into a single dataset with multiple splits
        for i, dataset in enumerate(datasets):
            training_dataset[f"shot={N}_run={i}"] = dataset

    # Save the dataset dict
    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_dataset.save_to_disk(str(output_path))


if __name__ == "__main__":
    app()
