# DATS Perspectives
This repo contains the evaluation experiments of the paper "DATS Perspectives – Interactive Document Clustering in the Discourse Analysis Tool Suite"

Have a look at the DATS repo here: https://github.com/uhh-lt/dats

## Goals

In the paper, we introduce changes to the typcial topic discovery pipeline in order to develop a more customizable aspect-focused document clustering pipeline:
1. optional document rewriting with an LLM
2. document embedding with instruction fine-tuned embedding model
3. dimensionality reduction with UMAP
4. clustering with HDBSCAN

This enables users to define two kinds of prompts to steer the document clustering:
- document rewriting prompts (e.g. summarize the main topic of this document)
- embedding instruction prompts (e.g. identify the main topic)

Further, we allow few-shot embeding model fine-tuning in order to align the clustering even more with user intent.
In this repo, we evaluate 2, 4, 8, and 16-shot few-shot learning.

## Evaluation

We evaluate this pipeline on three datasets:
- Spotify Songtexts
- Amazon Reviews
- 20 Newsgroups posts

We compute the following metrics, but only report knn accuracy in the paper:
- silhouette score
- adjusted rand index
- v measure score
- purity
- knn accuracy

## Results

These are the same results that we report in the paper: 

|         	| emoti 	| genre 	| prod  	| stars 	| topic 	|
|---------	|-------	|-------	|-------	|-------	|-------	|
| text    	| 45.10 	| 27.20 	| 36.72 	| 47.54 	| 71.17 	|
| +inst   	| 50.60 	| 27.50 	| 55.76 	| 58.86 	| 71.24 	|
| keyp    	| 49.15 	| 34.85 	| 54.64 	| 46.34 	| 65.70 	|
| +inst   	| 49.51 	| 34.90 	| 60.64 	| 49.68 	| 65.40 	|
| summ    	| 47.69 	| 32.59 	| 48.16 	| 52.14 	| 60.62 	|
| +inst   	| 48.04 	| 32.64 	| 61.18 	| 54.02 	| 61.91 	|
| 2-shot  	| 50.70 	| 36.63 	| 62.98 	| 59.71 	| 71.36 	|
| 4-shot  	| 50.92 	| 36.60 	| 63.54 	| 60.24 	| 71.49 	|
| 8-shot  	| 51.64 	| 36.66 	| 63.62 	| 60.09 	| 71.58 	|
| 16-shot 	| 52.07 	| 37.02 	| 64.04 	| 61.27 	| 72.15 	|


![Evaluation results](https://github.com/uhh-lt/perspectives/blob/main/visualizations/output.png "Evaluation results")