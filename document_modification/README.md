# Document Modification / Rewriting

This folder contains scripts to modify the original documents / test data.

We modify the documents in two ways:
- keyphrase generation
- summary generation

This way, we have three different representations of a document:
- the original text
- the keyphrases
- the summary

We used prompts like this:

```
Write a concise summary (maximum 5 sentences) that focuses on the emotional tone of the following song lyrics. Analyze the lyrics to determine the main emotion being conveyed and describe how it is expressed. Conclude with an emotion categorization:
```

```
Generate keyphrases (maximum 5 phrases) that describe the emotional tone of the following song lyrics. Focus on phrases that reflect the emotional tone, mood, or feelings expressed in the lyrics. Output just the keyphrases in a comma-separated format:
```

The `run_modifications.sh` script generates keyphrases and summaries for all datasets using such prompts and stores them in the dataframe as new columns.