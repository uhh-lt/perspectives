#!/bin/bash
# This script runs the dataset creation process
# Ensure the script is executable: chmod +x run_fewshot_dataset.sh
# Usage: ./run_fewshot_dataset.sh

DATASET=spotify
ASPECTS=("emotion" "genre")

for ASPECT in "${ASPECTS[@]}"; do
    COLUMNS=("text" "${ASPECT}-summary" "${ASPECT}-keyphrases")

    for COLUMN in "${COLUMNS[@]}"; do
        uv run convert_to_dataset.py \
            --dataset-dir "$DATASET" \
            --text-column "$COLUMN" \
            --label-column "$ASPECT" \
            --output-path "${DATASET}/${DATASET}_${ASPECT}_${COLUMN}"
    done
done

DATASET=amazon
ASPECTS=("stars" "product_category")

for ASPECT in "${ASPECTS[@]}"; do
    COLUMNS=("text" "${ASPECT}-summary" "${ASPECT}-keyphrases")

    for COLUMN in "${COLUMNS[@]}"; do
        uv run convert_to_dataset.py \
            --dataset-dir "$DATASET" \
            --text-column "$COLUMN" \
            --label-column "$ASPECT" \
            --output-path "${DATASET}/${DATASET}_${ASPECT}_${COLUMN}"
    done
done

DATASET=newsgroups
ASPECTS=("topic")

for ASPECT in "${ASPECTS[@]}"; do
    COLUMNS=("text" "${ASPECT}-summary" "${ASPECT}-keyphrases")

    for COLUMN in "${COLUMNS[@]}"; do
        uv run convert_to_dataset.py \
            --dataset-dir "$DATASET" \
            --text-column "$COLUMN" \
            --label-column "$ASPECT" \
            --output-path "${DATASET}/${DATASET}_${ASPECT}_${COLUMN}"
    done
done