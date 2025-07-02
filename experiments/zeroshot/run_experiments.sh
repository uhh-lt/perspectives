#!/bin/bash

GPU_ID=4

# SPOTIFY
CUDA_VISIBLE_DEVICES=$GPU_ID uv run run_experiment.py spotify-emotion
CUDA_VISIBLE_DEVICES=$GPU_ID uv run run_experiment.py spotify-genre

# AMAZON
CUDA_VISIBLE_DEVICES=$GPU_ID uv run run_experiment.py amazon-stars
CUDA_VISIBLE_DEVICES=$GPU_ID uv run run_experiment.py amazon-product-category

# 20 NEWSGROUPS
CUDA_VISIBLE_DEVICES=$GPU_ID uv run run_experiment.py newsgroups-topic