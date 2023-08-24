# Prompt Engineering with Hugging Face, Databricks and MLflow

<img src="https://github.com/rafaelvp-db/databricks-llm-workshop/blob/main/img/header.png?raw=true" />

## Contents

The repo is structured per different use cases related to **Prompt Engineering** and **Large Language Models (LLMs)**.

As of 18/08/2023, you will find the following examples in the `notebooks` folder:

🙋🏻‍♂️ `customer_service`

| Notebook            | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|---------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `hf_mlflow_crash_course` <img width=500/>   | 🤓 Provides a basic example using [Hugging Face](https://huggingface.co/) for training an [intent classification model](https://research.aimultiple.com/intent-classification/) using `distilbert-qa`. Also showcases foundational concepts of MLflow, such as [experiment tracking](https://mlflow.org/docs/latest/tracking.html), [artifact logging](https://mlflow.org/docs/latest/python_api/mlflow.artifacts.html) and [model registration](https://mlflow.org/docs/latest/model-registry.html). |
| `primer`                   | 🎬 Mostly conceptual notebook. Contains explanations around Prompt Engineering, and foundational concepts such as **Top K** sampling, **Top p** sampling and **Temperature**.                                                                                                                                                                                                                                                                                                                         |
| `basic_prompt_evaluation`            | 🧪  Demonstrates basic Prompt Engineeering with lightweight LLM models. In addition to this, showcases [MLflow's newest LLM features](https://www.databricks.com/blog/announcing-mlflow-24-llmops-tools-robust-model-evaluation), such as `mlflow.evaluate()`.                                                                                                                                                                                                                                         |
| `few_shot_learning`        | 💉 Here we explore [Few Shot Learning](https://blog.paperspace.com/few-shot-learning/) with an [Instruction Based LLM](https://blog.gopenai.com/an-introduction-to-base-and-instruction-tuned-large-language-models-8de102c785a6) ([mpt-7b-instruct](https://huggingface.co/mosaicml/mpt-7b-instruct)).                                                                                                                                                                                               |
| `active_prompting`       | 🏃🏻‍♂️ In this notebook, we explore active learning techniques. Additionally, we demonstrate how to leverage [VLLM](https://vllm.readthedocs.io/en/latest/) in order to achieve 7X - 10X inference latency improvements.                                                                                                                                                                                                                                                                                  |

## Getting Started

To start using this repo on Databricks, there are a few pre-requirements:

1. Create a [GPU Cluster](https://learn.microsoft.com/en-us/azure/databricks/clusters/gpu), minimally with [Databricks Machine Learning Runtime 13.2 GPU](https://docs.databricks.com/en/release-notes/runtime/13.2ml.html) and an [NVIDIA T4 GPU](https://www.nvidia.com/en-us/data-center/tesla-t4/) ([A100](https://www.nvidia.com/en-us/data-center/a100/) is required for the steps involving VLLM).
2. Install CUDA additional dependencies
   2.1 First, [clone this repo to your workspace](https://docs.databricks.com/en/repos/index.html)
   2.2 Then, configure a init script in your cluster by pointing to the following path in the Init Script configuration: `/Repos/your_name@email.com/databricks-llm-prompt-engineering/init/init.sh`
4. Install the following Python packages in your cluster:
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
🚀 [Model Deployment and Real Time Inference](https://docs.databricks.com/en/machine-learning/model-serving/index.html)
<br/>
🛣️ [MLflow AI Gateway](https://mlflow.org/docs/latest/gateway/index.html)

## Credits & Reference

* [Rafael Pierre](https://github.com/rafaelvp-db)
* [Daniel Liden](https://github.com/djliden)
* [Getting Started with NLP using Hugging Face Transformers](https://www.databricks.com/blog/2023/02/06/getting-started-nlp-using-hugging-face-transformers-pipelines.html)
* [DAIR.ai - Prompt Engineering Guide](https://www.promptingguide.ai/)
* [Peter Cheng - Token Selection Strategies: Top-K, Top-p and Temperature](https://peterchng.com/blog/2023/05/02/token-selection-strategies-top-k-top-p-and-temperature/)
