#!/bin/bash
# This script runs the training data preparation
# Ensure the script is executable: chmod +x run_fewshot_sampling.sh
# Usage: ./run_fewshot_sampling.sh

# Spotify Dataset Preparation

uv run sample_fewshot_examples.py \
--dataset spotify \
--text-column text \
--category-column emotion \
--output-dir spotify

uv run sample_fewshot_examples.py \
--dataset spotify \
--text-column text \
--category-column genre \
--output-dir spotify


# Amazon Dataset Preparation

uv run sample_fewshot_examples.py \
--dataset amazon \
--text-column text \
--category-column stars \
--output-dir amazon

uv run sample_fewshot_examples.py \
--dataset amazon \
--text-column text \
--category-column product_category \
--output-dir amazon


# 20 Newsgroups Dataset Modification

uv run sample_fewshot_examples.py \
--dataset newsgroups \
--text-column text \
--category-column label \
--output-dir newsgroups
