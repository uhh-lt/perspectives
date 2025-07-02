import typer
from pathlib import Path
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score, v_measure_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
import os
from itertools import product

# Ensure that CUDA_VISIBLE_DEVICES is set to use the first GPU
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    raise EnvironmentError(
        "CUDA_VISIBLE_DEVICES environment variable is not set. Please set it to the GPU you want to use."
    )

app = typer.Typer()


def load_embeddings(embedding_path: Path) -> np.ndarray:
    """Load embeddings from a .npz file."""
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    data = np.load(embedding_path)["embeddings"]
    return data


def load_dataset(
    dataset_path: Path, label_column: str, text_column: str
) -> tuple[pd.Series, list[str], list[int], list[str]]:
    """Load dataset from a parquet file and return the specified label column."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in the dataset.")

    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in the dataset.")

    # create a filter mask to remove rows where text_colum starts with "Error:"
    mask = df[text_column].str.startswith("Error:")
    df = df[~mask]
    print(f"Filtered out {mask.sum()} rows with 'Error:' in '{text_column}' column.")

    label_names = df[label_column].unique().tolist()
    label2id = {name: idx for idx, name in enumerate(label_names)}
    labels = df[label_column].map(label2id).astype(int).tolist()
    texts = df[text_column].tolist()

    return mask, label_names, labels, texts


def calculate_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Calculate the purity of clustering."""
    contingency_matrix = pd.crosstab(labels_pred, labels_true)
    purity = np.sum(np.max(contingency_matrix.values, axis=1)) / np.sum(
        contingency_matrix.values
    )
    return purity.item()


def evaluate_clustering(
    embeddings: np.ndarray, labels: np.ndarray
) -> tuple[dict, dict]:
    """Evaluate clustering using grid search over UMAP and HDBSCAN parameters."""

    param_grid = {
        "n_neighbors": [15, 20, 40, 80],  # Number of neighbors for UMAP
        "n_components": [2, 4, 8, 16, 32, 64, 128],  # UMAP dimension
        "min_samples": [5, 10, 20, 40, 80],  # Minimum samples for HDBSCAN
    }

    # best_score = -float("inf")
    best_params = {}
    best_metrics = {}

    best_silhouette = -float("inf")
    best_ari = -float("inf")
    best_v_measure = -float("inf")
    best_purity = -float("inf")

    silhouettes = []
    aris = []
    v_measures = []
    purities = []

    # Iterate over all parameter combinations
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))

        # Reduce dimensionality using UMAP
        try:
            umap = UMAP(
                n_neighbors=param_dict["n_neighbors"],
                n_components=param_dict["n_components"],
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            )
            reduced_embeddings = umap.fit_transform(embeddings)
        except Exception:
            continue

        # Cluster embeddings using HDBSCAN
        hdbscan = HDBSCAN(
            min_samples=param_dict["min_samples"],
            gen_min_span_tree=True,
            metric="euclidean",
        )
        cluster_labels = hdbscan.fit_predict(reduced_embeddings)

        # Filter outliers
        mask = cluster_labels != -1
        reduced_embeddings = reduced_embeddings[mask]
        cluster_labels = cluster_labels[mask]
        filtered_labels = labels[mask]

        # Skip if no clusters are formed
        if len(set(cluster_labels)) <= 1:
            continue

        # Calculate evaluation metrics
        silhouette = silhouette_score(reduced_embeddings, cluster_labels)
        ari = adjusted_rand_score(filtered_labels, cluster_labels)
        v_measure = v_measure_score(filtered_labels, cluster_labels)
        purity = calculate_purity(filtered_labels, cluster_labels)

        silhouettes.append(silhouette)
        aris.append(ari)
        v_measures.append(v_measure)
        purities.append(purity)

        if silhouette > best_silhouette:
            best_silhouette = silhouette
        if ari > best_ari:
            best_ari = ari
        if v_measure > best_v_measure:
            best_v_measure = v_measure
        if purity > best_purity:
            best_purity = purity

    best_metrics = {
        "silhouette_score": round(best_silhouette, 4),
        "adjusted_rand_index": round(best_ari, 4),
        "v_measure_score": round(best_v_measure, 4),
        "purity": round(best_purity, 4),
        "silhouette_score_avg": round(np.mean(silhouettes), 4),
        "adjusted_rand_index_avg": round(np.mean(aris), 4),
        "v_measure_score_avg": round(np.mean(v_measures), 4),
        "purity_avg": round(np.mean(purities), 4),
    }

    return best_metrics, best_params


def evaluate_2d_classification(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Evaluate clustering using silhouette score and adjusted rand index."""

    # Perform 10 runs to calculate average accuracy
    accuracies = []
    for seed in [7, 42, 69, 96, 123, 404, 500, 666, 1024, 1337]:
        # Reduce dimensionality using UMAP
        try:
            umap = UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.0,
                metric="cosine",
                random_state=seed,
            )
            embeddings_2d = umap.fit_transform(embeddings)
        except Exception:
            try:
                print("Retrying with different UMAP parameters...")
                umap = UMAP(
                    n_components=2,
                    n_neighbors=20,
                    min_dist=0.0,
                    metric="cosine",
                    random_state=seed,
                )
                embeddings_2d = umap.fit_transform(embeddings)
            except Exception:
                try:
                    print("Retrying with different UMAP parameters...")
                    umap = UMAP(
                        n_components=2,
                        n_neighbors=40,
                        min_dist=0.0,
                        metric="cosine",
                        random_state=seed,
                    )
                    embeddings_2d = umap.fit_transform(embeddings)
                except Exception:
                    return {
                        "knn_accuracy": -1,
                    }

        # Train and evaluate a k-NN classifier using 5-fold cross-validation
        knn = KNeighborsClassifier()
        scores = cross_val_score(
            knn, X=embeddings_2d, y=labels, cv=5, scoring="accuracy"
        )
        accuracies.append(scores.mean())

    # Calculate the average accuracy
    print(accuracies)
    average_accuracy = np.mean(accuracies).round(4).item()

    return {
        "knn_accuracy": average_accuracy,
    }


def parse_embedding_str(embedding_str: str, category: str):
    embedding_str = embedding_str.replace(f"{category}-", "")
    embedding_str = embedding_str.replace(f"{category}", "")
    embedding_str = embedding_str.replace("__", "_")

    is_inst = embedding_str.endswith("-inst")
    if is_inst:
        embedding_str = embedding_str[:-5]  # Remove "-inst" suffix

    splitted = embedding_str.split("_")
    dataset = splitted[0]
    text_column = splitted[1]
    model_name_short = splitted[2]
    shot = int(splitted[3].replace("shot", ""))

    return dataset, text_column, model_name_short, shot, is_inst


@app.command()
def clustering_evaluation(
    dataset_path: Path = typer.Option(..., help="Path to the dataset file"),
    embedding_path: Path = typer.Option(
        ..., help="Path to the numpy npz file containing embeddings"
    ),
    label_column: str = typer.Option(
        ..., help="Name of the column containing the labels"
    ),
    text_column: str = typer.Option(..., help="Name of the column containing the text"),
    output_dir: Path = typer.Option(
        ..., help="Path to directory to save the evaluation results"
    ),
):
    """
    Perform document clustering and evaluate the results.
    """
    # Ensure the output directory exists
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings and dataset
    print("Loading embeddings and dataset...")
    embeddings = load_embeddings(embedding_path)
    filter_mask, label_names, labels, texts = load_dataset(
        dataset_path, label_column, text_column
    )
    # Filter embeddings based on the filter mask
    embeddings = embeddings[~filter_mask]

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

    dataset, text_column, model_name_short, shot, is_inst = parse_embedding_str(
        embedding_path.stem, label_column
    )

    # Combine results
    results = {
        "dataset": dataset,
        "label_column": label_column,
        "text_column": text_column,
        "model": model_name_short,
        "num_shots": shot,
        "run": 0,
        "instruction": is_inst,
        **classification_results,
        **clustering_results,
        "accuracy": -1,
    }

    # Save evaluation results
    pd.DataFrame([results]).to_csv(
        output_dir / embedding_path.with_suffix(".csv"), index=False
    )


if __name__ == "__main__":
    app()
