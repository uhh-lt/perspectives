# Dataset Creation for Few-shot Training

This folder contains scripts to prepare the training data for few-shot sentence transformer finetuning.
It implements the following steps:

1. Sample few-shot examples from the dataset's training set (`run_fewshot_sampling.sh`).
    1. Sample N=2,4,8,16
    2. Do this 10 times (to prepare 10 splits for different training runs)
2. Rewrite the dataset's text with LLMs, write summary and generate keyphrases per aspect (`run_fewshot_modification.sh`)
3. Create datasets that are ready-to-use with SetFit library (`run_fewshot_dataset`)