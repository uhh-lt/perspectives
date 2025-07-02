from pathlib import Path
import typer
import os

from experiments.combine_result_csvs import combine_result_csvs
from experiments.clustering_evaluation import clustering_evaluation
from experiments.zeroshot.document_embedding import SupportedModel, document_embedding

# Ensure that CUDA_VISIBLE_DEVICES is set to use the first GPU
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    raise EnvironmentError(
        "CUDA_VISIBLE_DEVICES environment variable is not set. Please set it to the GPU you want to use."
    )

app = typer.Typer()

repo_path = Path("/ltstorage/home/tfischer/Development/interactive-clustering")


def run_experiment(
    model_name: SupportedModel,
    model_name_short: str,
    dataset_name: str,
    label_column: str,
    instructions: list[str],
):
    dataset_path = (
        repo_path / f"datasets/{dataset_name}/data/{dataset_name}_test.parquet"
    )
    output_dir = repo_path / f"experiments/{dataset_name}_{label_column}"
    text_columns = ["text", f"{label_column}-summary", f"{label_column}-keyphrases"]

    for inst in [instructions, [None, None, None]]:
        assert len(text_columns) == len(inst), (
            "Number of text columns must match number of inst."
        )

        for text_column, instruction in zip(text_columns, inst):
            emb_output_path = (
                output_dir
                / f"{dataset_name}_{label_column}_{text_column}_{model_name_short}_shot0{'-inst' if instruction else ''}.npz"
            )

            document_embedding(
                dataset_path=str(dataset_path),
                text_column=text_column,
                output_path=str(emb_output_path),
                model_name=model_name,
                instruction=instruction,
            )

            clustering_evaluation(
                dataset_path=dataset_path,
                embedding_path=emb_output_path,
                label_column=label_column,
                text_column=text_column,
                output_dir=output_dir,
            )

    combine_result_csvs(
        results_dir=output_dir,
        output_dir=output_dir,
    )


@app.command()
def spotify_emotion():
    print("Running Spotify Emotion Experiment")

    model_name = SupportedModel.MULTILINGUAL_E5_LARGE_INSTRUCT
    model_name_short = "multie5li"
    dataset_name = "spotify"
    label_column = "emotion"
    instructions = [
        "Identify the main emotion expressed in the given song text",
        "Identify the main emotion described by the given summary of a song text",
        "Identify the main emotion described by the given keyphrases of a song text",
    ]

    run_experiment(
        model_name=model_name,
        model_name_short=model_name_short,
        dataset_name=dataset_name,
        label_column=label_column,
        instructions=instructions,
    )


@app.command()
def spotify_genre():
    print("Running Spotify Genre Experiment")

    model_name = SupportedModel.MULTILINGUAL_E5_LARGE_INSTRUCT
    model_name_short = "multie5li"
    dataset_name = "spotify"
    label_column = "genre"
    instructions = [
        "Identify the main genre of the given song text",
        "Identify the main genre described by the given summary of a song text",
        "Identify the main genre described by the given keyphrases of a song text",
    ]

    run_experiment(
        model_name=model_name,
        model_name_short=model_name_short,
        dataset_name=dataset_name,
        label_column=label_column,
        instructions=instructions,
    )


@app.command()
def amazon_stars():
    print("Running Amazon Stars Experiment")

    model_name = SupportedModel.MULTILINGUAL_E5_LARGE_INSTRUCT
    model_name_short = "multie5li"
    dataset_name = "amazon"
    label_column = "stars"
    instructions = [
        "Identify the sentiment of the given Amazon review",
        "Identify the sentiment of the given summary of an Amazon review",
        "Identify the sentiment of the given keyphrases of an Amazon review",
    ]

    run_experiment(
        model_name=model_name,
        model_name_short=model_name_short,
        dataset_name=dataset_name,
        label_column=label_column,
        instructions=instructions,
    )


@app.command()
def amazon_product_category():
    print("Running Amazon Product Category Experiment")

    model_name = SupportedModel.MULTILINGUAL_E5_LARGE_INSTRUCT
    model_name_short = "multie5li"
    dataset_name = "amazon"
    label_column = "product_category"
    instructions = [
        "Identify the main product category of an Amazon product based on the given review",
        "Identify the main product category of an Amazon product based on the given review summary",
        "Identify the main product category of an Amazon product based on the given review keyphrases",
    ]

    run_experiment(
        model_name=model_name,
        model_name_short=model_name_short,
        dataset_name=dataset_name,
        label_column=label_column,
        instructions=instructions,
    )


@app.command()
def newsgroups_topic():
    print("Running Newsgroups Topic Experiment")

    model_name = SupportedModel.MULTILINGUAL_E5_LARGE_INSTRUCT
    model_name_short = "multie5li"
    dataset_name = "newsgroups"
    label_column = "topic"
    instructions = [
        "Identify the topic or theme of the given news article",
        "Identify the main topic described by the given summary of a news article",
        "Identify the main topic described by the given keyphrases of a news article",
    ]

    run_experiment(
        model_name=model_name,
        model_name_short=model_name_short,
        dataset_name=dataset_name,
        label_column=label_column,
        instructions=instructions,
    )


if __name__ == "__main__":
    app()
