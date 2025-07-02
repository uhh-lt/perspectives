# Experiments / Evaluation

This folder contains scripts to run clustering evaluation of embeddings.
The evaluation metrics are computed in `clustering_evaluation.py`:
- silhouette score
- adjusted rand index
- v measure score
- purity
- knn accuracy

In our paper, we only report knn accuracy metric.

We conduct two types of experiments: unsupervised (zero-shot) and supervised (few-shot).
In the unsupervised setting, the document embeddings are computed by the instruction-tuned emedding model as is and evaluated.
In the few-shot, supervised setting, the embedding model is first trained with a couple of examples (2, 4, 8, 16) and then document embeddings are computed and evaluated.

See the scripts `run_experiments.sh` and `run_fewshot_experiment.sh`.