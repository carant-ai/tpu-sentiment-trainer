# general-sentiment-trainer
Reference Paper: [Auxiliary-Sentence Construction for Implicit Aspect Learning in Sentiment Analysis](https://arxiv.org/abs/2203.11702)

In this repository, we aim to create a good bilinggual (English and Indonesian) sentiment model by finetuning XLM Roberta model on dedicated datasets.

# Dataset Cards
1. [carant-ai/compiled-absa-indonesian](https://huggingface.co/datasets/carant-ai/compiled-absa-indonesian)
2. [carant-ai/compiled-absa-english](https://huggingface.co/datasets/carant-ai/compiled-absa-english)
3. [carant-ai/indonesian_sentiment_dataset](https://huggingface.co/datasets/carant-ai/indonesian_sentiment_dataset)
4. [carant-ai/english_sentiment_dataset](https://huggingface.co/datasets/carant-ai/english_sentiment_dataset)

# Model Cards
1. [carant-ai/xlm-roberta-absa-base](https://huggingface.co/carant-ai/xlm-roberta-absa-base)
2. [carant-ai/xlm-roberta-sentiment-large](https://huggingface.co/carant-ai/xlm-roberta-sentiment-large)
3. [carant-ai/xlm-roberta-sentiment-base](https://huggingface.co/carant-ai/xlm-roberta-sentiment-base)
   
# Code
This repository contains:
1. ETL notebook: download various open source sentiment dataset, process into single table, and upload it as huggingface datasets.
2. Pytorch Lightning code to finetune Transformer encoder model (Roberta) on overall sentiment and aspect based sentiment task.
3. Prefilled environment variables for model replication.
