from pathlib import Path
import typer
import os

from experiments.combine_result_csvs import combine_result_csvs
from experiments.fewshot.document_embedding_fewshot import (
    SupportedModel,
    document_embedding_fewshot,
)

# Ensure that CUDA_VISIBLE_DEVICES is set to use the first GPU
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    raise EnvironmentError(
        "CUDA_VISIBLE_DEVICES environment variable is not set. Please set it to the GPU you want to use."
    )

app = typer.Typer()

repo_path = Path("/ltstorage/home/tfischer/Development/interactive-clustering")


def run_fewshot_experiment(
    model_name: SupportedModel,
    model_name_short: str,
    dataset_name: str,
    label_column: str,
    instructions: list[str],
):
    output_dir = repo_path / f"experiments/{dataset_name}_{label_column}_fewshoteva"
    text_columns = ["text", f"{label_column}-summary", f"{label_column}-keyphrases"]
    test_dataset_path = (
        repo_path / f"datasets/{dataset_name}/data/{dataset_name}_test.parquet"
    )

    for insts in [instructions, [None, None, None]]:
        assert len(text_columns) == len(insts), (
            "Number of text columns must match number of inst."
        )

        for text_column, instruction in zip(text_columns, insts):
            train_dataset_path = (
                repo_path
                / f"document_modification_fewshot/{dataset_name}/{dataset_name}_{label_column}_{text_column}"
            )

            for num_shots in [2, 4, 8, 16]:
                for run in range(10):
                    # for run in [0, 1, 2]:
                    document_embedding_fewshot(
                        train_dataset_path=train_dataset_path,
                        run=run,
                        num_shots=num_shots,
                        test_dataset_path=test_dataset_path,
                        test_dataset_text_column=text_column,
                        test_dataset_label_column=label_column,
                        model_name=model_name,
                        model_name_short=model_name_short,
                        instruction=instruction,
                        # training params
                        batch_size=16,
                        batch_size_head=8,
                        num_epochs=1,
                        num_epochs_head=16,
                        # adapter
                        r=24,
                        lora_alpha=8,
                        lora_dropout=0.1,
                        # learning rate
                        body_learning_rate1=2e-5,
                        body_learning_rate2=1e-5,
                        seed=42,
                        output_dir=output_dir,
                    )

    combine_result_csvs(
        results_dir=output_dir,
        output_dir=output_dir,
    )


@app.command()
def spotify_emotion():
    print("Running Spotify Emotion Fewshot Experiment")

    model_name = SupportedModel.MULTILINGUAL_E5_LARGE_INSTRUCT
    model_name_short = "multie5li"
    dataset_name = "spotify"
    label_column = "emotion"
    instructions = [
        "Identify the main emotion expressed in the given song text",
        "Identify the main emotion described by the given summary of a song text",
        "Identify the main emotion described by the given keyphrases of a song text",
    ]

    run_fewshot_experiment(
        model_name=model_name,
        model_name_short=model_name_short,
        dataset_name=dataset_name,
        label_column=label_column,
        instructions=instructions,
    )


@app.command()
def spotify_genre():
    print("Running Spotify Genre Fewshot Experiment")

    model_name = SupportedModel.MULTILINGUAL_E5_LARGE_INSTRUCT
    model_name_short = "multie5li"
    dataset_name = "spotify"
    label_column = "genre"
    instructions = [
        "Identify the main genre of the given song text",
        "Identify the main genre described by the given summary of a song text",
        "Identify the main genre described by the given keyphrases of a song text",
    ]

    run_fewshot_experiment(
        model_name=model_name,
        model_name_short=model_name_short,
        dataset_name=dataset_name,
        label_column=label_column,
        instructions=instructions,
    )


@app.command()
def amazon_stars():
    print("Running Amazon Stars Fewshot Experiment")

    model_name = SupportedModel.MULTILINGUAL_E5_LARGE_INSTRUCT
    model_name_short = "multie5li"
    dataset_name = "amazon"
    label_column = "stars"
    instructions = [
        "Identify the sentiment of the given Amazon review",
        "Identify the sentiment of the given summary of an Amazon review",
        "Identify the sentiment of the given keyphrases of an Amazon review",
    ]

    run_fewshot_experiment(
        model_name=model_name,
        model_name_short=model_name_short,
        dataset_name=dataset_name,
        label_column=label_column,
        instructions=instructions,
    )


@app.command()
def amazon_product_category():
    print("Running Amazon Product Category Fewshot Experiment")

    model_name = SupportedModel.MULTILINGUAL_E5_LARGE_INSTRUCT
    model_name_short = "multie5li"
    dataset_name = "amazon"
    label_column = "product_category"
    instructions = [
        "Identify the main product category of an Amazon product based on the given review",
        "Identify the main product category of an Amazon product based on the given review summary",
        "Identify the main product category of an Amazon product based on the given review keyphrases",
    ]

    run_fewshot_experiment(
        model_name=model_name,
        model_name_short=model_name_short,
        dataset_name=dataset_name,
        label_column=label_column,
        instructions=instructions,
    )


@app.command()
def newsgroups_topic():
    print("Running Newsgroups Topic Fewshot Experiment")

    model_name = SupportedModel.MULTILINGUAL_E5_LARGE_INSTRUCT
    model_name_short = "multie5li"
    dataset_name = "newsgroups"
    label_column = "topic"
    instructions = [
        "Identify the topic or theme of the given news articles",
        "Identify the main topic described by the given summary of a news article",
        "Identify the main topic described by the given keyphrases of a news article",
    ]

    run_fewshot_experiment(
        model_name=model_name,
        model_name_short=model_name_short,
        dataset_name=dataset_name,
        label_column=label_column,
        instructions=instructions,
    )


if __name__ == "__main__":
    app()
