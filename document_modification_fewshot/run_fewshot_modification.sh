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

# News Bias 2 Dataset Modification

# Find all files in the newsbias directory ending with bias.parquet
newsbias2_files=$(find newsbias2 -name "*bias.parquet")
echo "Found $(echo "$newsbias2_files" | wc -l) datasets ending with 'bias.parquet'"
for dataset in $newsbias2_files; do
    echo "Processing dataset: $dataset"

    echo "Summarizing ..."
    uv run ../document_modification/document_modification.py \
    --dataset-path "$dataset" \
    --text-column text \
    --column-name bias-summary \
    --system-prompt "You are a helpful assistant that summarizes news articles (maximum 5 sentences) in a way that highlights their political framing. The frames are left, center, and right." \
    --user-prompt "Write a concise summary (maximum 5 sentences) that focuses on the political framing of the following news article. Analyze the article to determine the main political framing. Conclude with a political categorization (one of left, center, right):"

    echo "Generating keyphrases ..."
    uv run ../document_modification/document_modification.py \
    --dataset-path "$dataset" \
    --text-column text \
    --column-name bias-keyphrases \
    --system-prompt "You are a helpful assistant that identifies keyphrases (maximum 5 phrases) related to the political framing in news articles. The frames are left, center, and right." \
    --user-prompt "Generate keyphrases (maximum 5 phrases) that describe the political framing of the following news article. Focus on phrases that reflect the main political framing (left, center, right) being discussed. Output just the keyphrases in a comma-separated format:"

done

# GVFC Dataset Modification

# Find all files in the gvfc directory ending with frame.parquet
gvfc_files=$(find gvfc -name "*frame.parquet")
echo "Found $(echo "$gvfc_files" | wc -l) datasets ending with 'frame.parquet'"
for dataset in $gvfc_files; do
    echo "Processing dataset: $dataset"

    echo "Summarizing ..."
    uv run ../document_modification/document_modification.py \
    --dataset-path "$dataset" \
    --text-column text \
    --column-name frame-summary \
    --system-prompt "You are a helpful assistant that summarizes news articles (maximum 5 sentences) in a way that highlights the discussed topic or theme." \
    --user-prompt "Write a concise summary (maximum 5 sentences) that focuses on the main topic or theme of the following news article. Analyze the article to determine the main topic or theme being discussed. Conclude with a topic categorization:"

    echo "Generating keyphrases ..."
    uv run ../document_modification/document_modification.py \
    --dataset-path "$dataset" \
    --text-column text \
    --column-name frame-keyphrases \
    --system-prompt "You are a helpful assistant that identifies keyphrases (maximum 5 phrases) related the discussed topic or theme in news articles." \
    --user-prompt "Generate keyphrases (maximum 5 phrases) that describe the topic of the following news article. Focus on phrases that reflect the main topic or theme being discussed. Output just the keyphrases in a comma-separated format:"
done

# Germeval Dataset Modification

# Find all files in the germeval directory ending with category.parquet
germeval_files=$(find germeval -name "*category.parquet")
echo "Found $(echo "$germeval_files" | wc -l) datasets ending with 'category.parquet'"
for dataset in $germeval_files; do
    echo "Processing dataset: $dataset"

    echo "Summarizing ..."
    uv run ../document_modification/document_modification.py \
    --dataset-path "$dataset" \
    --text-column text \
    --column-name category-summary \
    --system-prompt "Du bist ein hilfreicher Assistent, der Klappentexte von Büchern in maximal 5 Sätzen zusammenfasst, dass das Genre des Buches hervorgehoben wird. Die Genres sind Literatur & Unterhaltung, Ratgeber, Kinderbuch & Jugendbuch, Sachbuch, Ganzheitliches Bewusstsein, Glaube & Ethik, Künste, Architektur & Garten." \
    --user-prompt "Schreibe eine prägnante Zusammenfassung (maximal 5 Sätze), die sich auf das Genre des folgenden Klappentextes konzentriert. Analysiere den Text, um das Genre zu bestimmen. Schließe mit einer allgemeinen Genre-Kategorisierung ab:"

    echo "Generating keyphrases ..."
    uv run ../document_modification/document_modification.py \
    --dataset-path "$dataset" \
    --text-column text \
    --column-name category-keyphrases \
    --system-prompt "Du bist ein hilfreicher Assistent, der Schlüsselbegriffe (maximal 5 Begriffe) identifiziert, die mit dem Genre des Buches zusammenhängen. Die Genres sind Literatur & Unterhaltung, Ratgeber, Kinderbuch & Jugendbuch, Sachbuch, Ganzheitliches Bewusstsein, Glaube & Ethik, Künste, Architektur & Garten."  \
    --user-prompt "Generiere Schlüsselbegriffe (maximal 5 Begriffe), die das Genre des folgenden Klappentextes beschreiben. Konzentriere dich auf Begriffe, die das allgemeine Genre des Buches widerspiegeln. Gib nur die Schlüsselbegriffe in kommagetrennter Form aus:"
done

# Reddit Conflict Dataset Modification

# Find all files in the redditconflict directory ending with sentiment.parquet
redditconflict_files=$(find redditconflict -name "*sentiment.parquet")
echo "Found $(echo "$redditconflict_files" | wc -l) datasets ending with 'sentiment.parquet'"
for dataset in $redditconflict_files; do
    echo "Processing dataset: $dataset"

    echo "Summarizing ..."
    uv run ../document_modification/document_modification.py \
    --dataset-path "$dataset" \
    --text-column text \
    --column-name sentiment-summary \
    --system-prompt "You are a helpful assistant that summarizes reddit posts (maximum 5 sentences) in a way that highlights their stance towards the Israel-Palestine conflict. The stances are Pro-Israel, Pro-Palestine, and Neutral." \
    --user-prompt "Write a concise summary (maximum 5 sentences) that focuses on the sentiment of the following reddit post. Analyze the post to determine the wether it is Pro-Israel, Pro-Palestine, or Neutral. Conclude with a sentiment categorization:"

    echo "Generating keyphrases ..."
    uv run ../document_modification/document_modification.py \
    --dataset-path "$dataset" \
    --text-column text \
    --column-name sentiment-keyphrases \
    --system-prompt "You are a helpful assistant that identifies keyphrases (maximum 5 phrases) related to the stance towards the Israel-Palestine conflict in reddit posts. The stances are Pro-Israel, Pro-Palestine, and Neutral." \
    --user-prompt "Generate keyphrases (maximum 5 phrases) that describe the stance of the following reddit post. Focus on phrases that reflect the main stance (Pro-Israel, Pro-Palestine, Neutral) being discussed. Output just the keyphrases in a comma-separated format:"

done