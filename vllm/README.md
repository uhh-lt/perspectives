# VLLM

We use vLLM to serve Large Language Models with fast inference.

It is configured through the `.env` file. Copy paste the `.env.example` file. Then, set the following parameters:

```
CUDA_VISIBLE_DEVICES=

VLLM_MODEL_NAME=
VLLM_EXPOSED=
VLLM_API_KEY=
VLLM_MAX_TOKENS_TO_GENERATE=

HUGGING_FACE_CACHE_DIR=
HUGGING_FACE_HUB_TOKEN=
```