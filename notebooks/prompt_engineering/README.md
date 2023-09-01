# Large Language Models (LLMs) & Prompt Engineering with Hugging Face, Databricks and MLflow

![hf](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MPT-red?style=for-the-badge) ![hf](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LLAMA2-Blue?style=for-the-badge) ![pt](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![db](https://camo.githubusercontent.com/bf9d06ea392c793c80e66ab19c3ef8a86cf9287ab2aa8fc7b2662d8cdcb7c8c0/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f44617461627269636b732d4646333632312e7376673f7374796c653d666f722d7468652d6261646765266c6f676f3d44617461627269636b73266c6f676f436f6c6f723d7768697465) ![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)

<img src="https://github.com/rafaelvp-db/databricks-llm-workshop/blob/main/img/header.png?raw=true" />

## Contents

The repo is structured per different use cases related to **Prompt Engineering** and **Large Language Models (LLMs)**.

As of 29/08/2023, you will find the following examples in the `notebooks` folder:

🙋🏻‍♂️ `customer_service`

| Notebook            | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|---------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `hf_mlflow_crash_course` <img width=500/>   | 🤓 Provides a basic example using [Hugging Face](https://huggingface.co/) for training an [intent classification model](https://research.aimultiple.com/intent-classification/) using `distilbert-qa`. Also showcases foundational concepts of MLflow, such as [experiment tracking](https://mlflow.org/docs/latest/tracking.html), [artifact logging](https://mlflow.org/docs/latest/python_api/mlflow.artifacts.html) and [model registration](https://mlflow.org/docs/latest/model-registry.html). |
| `primer`                   | 🎬 Mostly conceptual notebook. Contains explanations around Prompt Engineering, and foundational concepts such as **Top K** sampling, **Top p** sampling and **Temperature**.                                                                                                                                                                                                                                                                                                                         |
| `basic_prompt_evaluation`            | 🧪  Demonstrates basic Prompt Engineeering with lightweight LLM models. In addition to this, showcases [MLflow's newest LLM features](https://www.databricks.com/blog/announcing-mlflow-24-llmops-tools-robust-model-evaluation), such as `mlflow.evaluate()`.                                                                                                                                                                                                                                         |
| `few_shot_learning`        | 💉 Here we explore [Few Shot Learning](https://blog.paperspace.com/few-shot-learning/) with an [Instruction Based LLM](https://blog.gopenai.com/an-introduction-to-base-and-instruction-tuned-large-language-models-8de102c785a6) ([mpt-7b-instruct](https://huggingface.co/mosaicml/mpt-7b-instruct)).                                                                                                                                                                                               |
| `active_prompting`       | 🏃🏻‍♂️ In this notebook, we explore active learning techniques. Additionally, we demonstrate how to leverage [VLLM](https://vllm.readthedocs.io/en/latest/) in order to achieve 7X - 10X inference latency improvements.                                                                                                                                                                                                                                                                                  |
| `llama2_mlflow_logging_inference`       | 🚀 Here we show how to log, register and deploy a [LLaMA V2](https://huggingface.co/docs/transformers/main/model_doc/llama2) model into MLflow                                                                                                                                                                                                                                                                                  |
| `mpt_mlflow_logging_inference`       | 🚀 Here we show how to log, register and deploy an [MPT-Instruct](https://huggingface.co/docs/transformers/main/model_doc/mpt) model into MLflow                                                                                                                                                                                                                                                                                  |


## Getting Started

To start using this repo on Databricks, there are a few pre-requirements:

1. Create a [GPU Cluster](https://learn.microsoft.com/en-us/azure/databricks/clusters/gpu), minimally with [Databricks Machine Learning Runtime 13.2 GPU](https://docs.databricks.com/en/release-notes/runtime/13.2ml.html) and an [NVIDIA T4 GPU](https://www.nvidia.com/en-us/data-center/tesla-t4/) ([A100](https://www.nvidia.com/en-us/data-center/a100/) is required for the steps involving VLLM).
2. *(only if using Databricks MLR < 13.2)* Install CUDA additional dependencies
   * First, [clone this repo to your workspace](https://docs.databricks.com/en/repos/index.html)
   * Configure an **init script** in your cluster by pointing to the following path in the Init Script configuration: `/Repos/your_name@email.com/databricks-llm-prompt-engineering/init/init.sh`
4. *(only if using MPT models)* Install the following Python packages in your cluster:
```bash
accelerate==0.21.0
einops==0.6.1
flash-attn==v1.0.5
ninja
tokenizers==0.13.3
transformers==4.30.2
xformers==0.0.20
```
4. Once all dependencies finish installing and your cluster has successfully started, you should be good to go.
   
## Coming soon

🔎 [Retrieval Augmented Generation (RAG)](https://www.promptingguide.ai/techniques/rag)
<br/>
~~🚀 [Model Deployment and Real Time Inference](https://docs.databricks.com/en/machine-learning/model-serving/index.html)~~
<br/>
🛣️ [MLflow AI Gateway](https://mlflow.org/docs/latest/gateway/index.html)

## Credits & Reference

* [Rafael Pierre](https://github.com/rafaelvp-db)
* [Daniel Liden](https://github.com/djliden)
* [Getting Started with NLP using Hugging Face Transformers](https://www.databricks.com/blog/2023/02/06/getting-started-nlp-using-hugging-face-transformers-pipelines.html)
* [DAIR.ai - Prompt Engineering Guide](https://www.promptingguide.ai/)
* [Peter Cheng - Token Selection Strategies: Top-K, Top-p and Temperature](https://peterchng.com/blog/2023/05/02/token-selection-strategies-top-k-top-p-and-temperature/)
* [Databricks ML Examples & LLM Models](https://github.com/databricks/databricks-ml-examples)
