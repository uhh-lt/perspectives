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

We evaluate this pipeline on various datasets:

- Amazon Reviews (product categorization & star/sentiment analysis)
- German Blurbs (blurb genre classification)
- Gun Violence Frame Corpus (news frame analysis)
- News Bias (news political bias analysis)
- 20 Newsgroups posts (topic modelling)
- Israel-Palestine conflict (stance detection)
- Spotify Songtexts (emotion & genre classification)

We compute the following metrics, but only report KNN Accuracy in the paper:

- silhouette score
- adjusted rand index
- v measure score
- purity
- knn accuracy

## Experiment setup

Models:

- LLM: https://huggingface.co/google/gemma-3-27b-it
- Embedding Model: https://huggingface.co/intfloat/multilingual-e5-large-instruct

These are the prompts we use for document rewriting:

| Dataset            | Prompt                                                                                                                                                                                                                                                                       |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| emotion-summary    | Write a concise summary (maximum 5 sentences) that focuses on the emotional tone of the following song lyrics. Analyze the lyrics to determine the main emotion being conveyed and describe how it is expressed. Conclude with an emotion categorization:                    |
| emotion-keyphrases | Generate keyphrases (max 5 phrases) that describe the emotional tone of the following song lyrics. Focus on phrases that reflect the emotional tone, mood, or feelings expressed in the lyrics.                                                                              |
| genre-summary      | Write a concise summary (maximum 5 sentences) that focuses on the genre of the following song lyrics. Analyze the lyrics to determine the musical genre, referencing stylistic elements, themes, or influences. Conclude with a genre categorization:                        |
| genre-keyphrases   | Generate keyphrases (max 5 phrases) that describe the musical genre of the following song lyrics. Focus on phrases that reflect the genre's characteristics, style, or influences.                                                                                           |
| stars-summary      | Write a concise summary (maximum 5 sentences) that focuses on the sentiment of the following Amazon review. Analyze the review to determine the main sentiment being conveyed and describe how it is expressed. Conclude with a sentiment categorization:                    |
| stars-keyphrases   | Generate keyphrases (max 5 phrases) that describe the sentiment of the following Amazon review. Focus on phrases that reflect the sentiment, mood, or feelings expressed in the review.                                                                                      |
| product-summary    | Write a concise summary (maximum 5 sentences) that focuses on the product categorization of the following Amazon review. Analyze the review to determine the discussed product's categorization, referencing its features, type, or purpose. Conclude with a categorization: |
| product-keyphrases | Generate keyphrases (max 5 phrases) that describe the product of the following Amazon review. Focus on phrases that reflect the product's type or category.                                                                                                                  |
| topic-summary      | Write a concise summary (maximum 5 sentences) that focuses on the topic or theme of the following news article. Analyze the article to determine the main topic or theme being discussed. Conclude with a topic categorization:                                              |
| topic-keyphrases   | Generate keyphrases (max 5 phrases) that describe the topic of the following article. Focus on phrases that reflect the main topic being discussed.                                                                                                                          |
| blurbs-summary     | Schreibe eine prägnante Zusammenfassung (maximal 5 Sätze), die sich auf das Genre des folgenden Klappentextes konzentriert. Analysiere den Text, um das Genre zu bestimmen. Schließe mit einer allgemeinen Genre-Kategorisierung ab.                                         |
| blurbs-keyphrases  | Generiere Schlüsselbegriffe (max 5 Begriffe), die das Genre des folgenden Klappentextes beschreiben. Fokusiere dich auf Begriffe, die das allgemeine Genre widerspiegeln.                                                                                                    |
| stance-summary     | Write a concise summary (maximum 5 sentences) that focuses on the stance the following Reddit post. Analyze the post to determine whether it is Pro-Israel, Pro-Palestine, or Neutral. Conclude with a stance categorization:                                                |
| stance-keyphrases  | Generate keyphrases (max 5 phrases) that describe the stance of the following Reddit post. Focus on phrases that reflect the stance being discussed.                                                                                                                         |
| bias-summary       | Write a concise summary (maximum 5 sentences) that focuses on the political leaning of the following news article. Analyze the article to determine the main political framing. Conclude with a political categorization:                                                    |
| bias-keyphrases    | Generate keyphrases (max 5 phrases) that describe the political leaning of following news article. Focus on phrases that reflect the political framing.                                                                                                                      |

These are the instructions we use for aspect-oriented embeddings:

| Dataset | Instruction                                                                                                        |
| ------- | ------------------------------------------------------------------------------------------------------------------ |
| emotion | Identify the main emotion expressed in the given (summary of a \| keyphrases of a) song text                       |
| genre   | Identify the main genre of the given (summary of a \| keyphrases of a) song text                                   |
| stars   | Identify the sentiment of the given (summary of an \| keyphrases of an) Amazon review                              |
| product | Identify the category of an Amazon product based on the given review (summary \| keyphrases)                       |
| topic   | Identify the topic or theme of the given (summary of a \| keyphrases of a) news article                            |
| frame   | Identify the framing of the given (summary of a \| keyphrases of a) news article                                   |
| blurb   | Identifiziere das Genre, dass durch die (Zusammenfassung \| Schlüsselbegriffe) des Klappentextes beschrieben wird. |
| stance  | Identify the stance towards the Israel-Palestine conflict of the given (summary \| keyphrases) of a news article   |
| bias    | Identify the political leaning of the given (summary of a \| keyphrases of a) news article                         |

## Results

These are the results that we report in the paper:

|           | emotion | genre | product | stars | topic | frames | blurbs | stance | bias  |
| --------- | ------- | ----- | ------- | ----- | ----- | ------ | ------ | ------ | ----- |
| text      | 45.10   | 27.20 | 36.72   | 47.54 | 71.17 | 59.78  | 73.84  | 51.19  | 47.20 |
| +inst     | 50.60   | 27.50 | 55.76   | 58.86 | 71.24 | 65.00  | 73.25  | 48.21  | 48.51 |
| keyphrase | 49.15   | 34.85 | 54.64   | 46.34 | 65.70 | 60.34  | 76.73  | 61.63  | 50.76 |
| +inst     | 49.51   | 34.90 | 60.64   | 49.68 | 65.40 | 58.31  | 76.66  | 63.14  | 50.69 |
| summary   | 47.69   | 32.59 | 48.16   | 52.14 | 60.62 | 64.34  | 76.46  | 62.47  | 48.41 |
| +inst     | 48.04   | 32.64 | 61.18   | 54.02 | 61.91 | 65.91  | 74.80  | 68.74  | 52.53 |
| 2-shot    | 50.70   | 36.63 | 62.98   | 59.71 | 71.36 | 65.10  | 76.79  | 70.17  | 52.72 |
| 4-shot    | 50.92   | 36.60 | 63.54   | 60.24 | 71.49 | 65.59  | 76.74  | 70.56  | 53.03 |
| 8-shot    | 51.64   | 36.66 | 63.62   | 60.09 | 71.58 | 66.92  | 76.92  | 71.06  | 53.09 |
| 16-shot   | 52.07   | 37.02 | 64.04   | 61.27 | 72.15 | 67.27  | 77.85  | 71.51  | 54.09 |

Here is a more detailed result breakdown, including all few-shot evaluation results.

![Evaluation results](https://github.com/uhh-lt/perspectives/blob/main/visualizations/output.png "Evaluation results")
