#!/bin/bash

GPU_ID=2

# SET ENV VARIABLES
export HF_HOME=/ltstorage/home/tfischer/.cache/huggingface
export PYTHONPATH="/ltstorage/home/tfischer/Development/interactive-clustering"

# SPOTIFY
CUDA_VISIBLE_DEVICES=$GPU_ID uv run run_fewshot_experiment.py spotify-emotion
CUDA_VISIBLE_DEVICES=$GPU_ID uv run run_fewshot_experiment.py spotify-genre

# AMAZON
CUDA_VISIBLE_DEVICES=$GPU_ID uv run run_fewshot_experiment.py amazon-stars
CUDA_VISIBLE_DEVICES=$GPU_ID uv run run_fewshot_experiment.py amazon-product-category

# 20 NEWSGROUPS
CUDA_VISIBLE_DEVICES=$GPU_ID uv run run_fewshot_experiment.py newsgroups-topic

# NEWS BIAS 2
CUDA_VISIBLE_DEVICES=$GPU_ID uv run run_fewshot_experiment.py newsbias2-bias

# GVFC
CUDA_VISIBLE_DEVICES=$GPU_ID uv run run_fewshot_experiment.py gvfc-frame

# GERMEVAL
CUDA_VISIBLE_DEVICES=$GPU_ID uv run run_fewshot_experiment.py germeval-category

# REDDIT CONFLICT
CUDA_VISIBLE_DEVICES=$GPU_ID uv run run_fewshot_experiment.py redditconflict-sentiment