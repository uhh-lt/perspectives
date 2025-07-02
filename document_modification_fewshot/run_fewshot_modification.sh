#!/bin/bash
# This script runs the document modification process
# Ensure the script is executable: chmod +x run_fewshot_modification.sh
# Usage: ./run_fewshot_modification.sh

# Spotify Dataset Modification

# Find all files in the spotify directory ending with emotion.parquet
spotify_emotion_files=$(find spotify -name "*emotion.parquet")
echo "Found $(echo "$spotify_emotion_files" | wc -l) datasets ending with 'emotion.parquet'"
for dataset in $spotify_emotion_files; do
    echo "Processing dataset: $dataset"

    echo "Summarizing ..."
    uv run ../document_modification/document_modification.py \
    --dataset-path "$dataset" \
    --text-column text \
    --column-name emotion-summary \
    --system-prompt "You are a helpful assistant that summarizes song texts (maximum 5 sentences) in a way that highlights their emotional tone." \
    --user-prompt "Write a concise summary (maximum 5 sentences) that focuses on the emotional tone of the following song lyrics. Analyze the lyrics to determine the main emotion being conveyed and describe how it is expressed. Conclude with an emotion categorization:"

    echo "Generating keyphrases ..."
    uv run ../document_modification/document_modification.py \
    --dataset-path "$dataset" \
    --text-column text \
    --column-name emotion-keyphrases \
    --system-prompt "You are a helpful assistant that identifies keyphrases (maximum 5 phrases) related to emotions conveyed in song lyrics." \
    --user-prompt "Generate keyphrases (maximum 5 phrases) that describe the emotional tone of the following song lyrics. Focus on phrases that reflect the emotional tone, mood, or feelings expressed in the lyrics. Output just the keyphrases in a comma-separated format:"
done

# Find all files in the spotify directory ending with genre.parquet
spotify_genre_files=$(find spotify -name "*genre.parquet")
echo "Found $(echo "$spotify_genre_files" | wc -l) datasets ending with 'genre.parquet'"
for dataset in $spotify_genre_files; do
    echo "Processing dataset: $dataset"

    echo "Summarizing ..."
    uv run ../document_modification/document_modification.py \
    --dataset-path "$dataset" \
    --text-column text \
    --column-name genre-summary \
    --system-prompt "You are a helpful assistant that summarizes song texts (maximum 5 sentences) in a way that highlights their musical genre." \
    --user-prompt "Write a concise summary (maximum 5 sentences) that focuses on the genre of the following song lyrics. Analyze the lyrics to determine the musical genre, referencing stylistic elements, themes, or influences. Conclude with a genre categorization:"

    echo "Generating keyphrases ..."
    uv run ../document_modification/document_modification.py \
    --dataset-path "$dataset" \
    --text-column text \
    --column-name genre-keyphrases \
    --system-prompt "You are a helpful assistant that identifies keyphrases (maximum 5 phrases) related to the musical genre of song lyrics." \
    --user-prompt "Generate keyphrases (maximum 5 phrases) that describe the musical genre of the following song lyrics. Focus on phrases that reflect the genre's characteristics, style, or influences. Output just the keyphrases in a comma-separated format:"
done

# Amazon Dataset Modification

# Find all files in the amazon directory ending with product_category.parquet
amazon_product_files=$(find amazon -name "*product_category.parquet")
echo "Found $(echo "$amazon_product_files" | wc -l) datasets ending with 'product_category.parquet'"
for dataset in $amazon_product_files; do
    echo "Processing dataset: $dataset"

    echo "Summarizing ..."
    uv run ../document_modification/document_modification.py \
    --dataset-path "$dataset" \
    --text-column text \
    --column-name product_category-summary \
    --system-prompt "You are a helpful assistant that summarizes amazon reviews (maximum 5 sentences) in a way that highlights the discussed product's categorization." \
    --user-prompt "Write a concise summary (maximum 5 sentences) that focuses on the product categorization of the following amazon review. Analyze the review to determine the discussed product's categorization, referencing its features, type, or purpose. Conclude with a product categorization:"

    echo "Generating keyphrases ..."
    uv run ../document_modification/document_modification.py \
    --dataset-path "$dataset" \
    --text-column text \
    --column-name product_category-keyphrases \
    --system-prompt "You are a helpful assistant that identifies keyphrases (maximum 5 phrases) related to the discussed product in amazon reviews." \
    --user-prompt "Generate keyphrases (maximum 5 phrases) that describe the product of the following amazon review. Focus on phrases that reflect the product's features, type, or category. Output just the keyphrases in a comma-separated format:"
done

# Find all files in the amazon directory ending with stars.parquet
amazon_stars_files=$(find amazon -name "*stars.parquet")
echo "Found $(echo "$amazon_stars_files" | wc -l) datasets ending with 'stars.parquet'"
for dataset in $amazon_stars_files; do
    echo "Processing dataset: $dataset"
    
    echo "Summarizing ..."
    uv run ../document_modification/document_modification.py \
    --dataset-path "$dataset" \
    --text-column text \
    --column-name stars-summary \
    --system-prompt "You are a helpful assistant that summarizes amazon reviews (maximum 5 sentences) in a way that highlights their sentiment." \
    --user-prompt "Write a concise summary (maximum 5 sentences) that focuses on the sentiment of the following amazon review. Analyze the review to determine the main sentiment being conveyed and describe how it is expressed. Conclude with a sentiment categorization:"

    echo "Generating keyphrases ..."
    uv run ../document_modification/document_modification.py \
    --dataset-path "$dataset" \
    --text-column text \
    --column-name stars-keyphrases \
    --system-prompt "You are a helpful assistant that identifies keyphrases (maximum 5 phrases) related to the sentiment of amazon reviews." \
    --user-prompt "Generate keyphrases (maximum 5 phrases) that describe the sentiment of the following amazon review. Focus on phrases that reflect the sentiment, mood, or feelings expressed in the review. Output just the keyphrases in a comma-separated format:"
done

# 20 Newsgroups Dataset Modification

# Find all files in the newsgroups directory ending with label.parquet
newsgroups_files=$(find newsgroups -name "*label.parquet")
echo "Found $(echo "$newsgroups_files" | wc -l) datasets ending with 'label.parquet'"
for dataset in $newsgroups_files; do
    echo "Processing dataset: $dataset"

    echo "Summarizing ..."
    uv run ../document_modification/document_modification.py \
    --dataset-path "$dataset" \
    --text-column text \
    --column-name topic-summary \
    --system-prompt "You are a helpful assistant that summarizes news articles (maximum 5 sentences) in a way that highlights the discussed topic or theme." \
    --user-prompt "Write a concise summary (maximum 5 sentences) that focuses on the main topic or theme of the following news article. Analyze the article to determine the main topic or theme being discussed. Conclude with a topic categorization:"

    echo "Generating keyphrases ..."
    uv run ../document_modification/document_modification.py \
    --dataset-path "$dataset" \
    --text-column text \
    --column-name topic-keyphrases \
    --system-prompt "You are a helpful assistant that identifies keyphrases (maximum 5 phrases) related the discussed topic or theme in news articles." \
    --user-prompt "Generate keyphrases (maximum 5 phrases) that describe the topic of the following news article. Focus on phrases that reflect the main topic or theme being discussed. Output just the keyphrases in a comma-separated format:"
done