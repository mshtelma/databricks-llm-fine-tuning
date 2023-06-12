# Databricks notebook source
# MAGIC !wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb -O /tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
# MAGIC   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcublas-dev-11-7_11.10.1.25-1_amd64.deb -O /tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
# MAGIC   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb -O /tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
# MAGIC   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcurand-dev-11-7_10.2.10.91-1_amd64.deb -O /tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb && \
# MAGIC   dpkg -i /tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
# MAGIC   dpkg -i /tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
# MAGIC  dpkg -i /tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
# MAGIC   dpkg -i /tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %pip install torch==2.0.1

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------
# MAGIC %pip install bitsandbytes
# MAGIC %pip install -U git+https://github.com/huggingface/transformers.git
# MAGIC %pip install -U git+https://github.com/huggingface/peft.git
# MAGIC %pip install -U git+https://github.com/huggingface/accelerate.git
# COMMAND ----------

import os

os.environ["HF_HOME"] = "/local_disk0/hf"
os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# COMMAND ----------

import torch

torch.cuda.set_device("cuda:0")

# COMMAND ----------

from databricks_llm.fine_tune import ExtendedTrainingArguments
from databricks_llm.fine_tune import train

training_args = ExtendedTrainingArguments(
    dataset="timdettmers/openassistant-guanaco",
    model="tiiuae/falcon-40b",
    tokenizer="tiiuae/falcon-40b",
    use_lora=True,
    use_4bit=True,
    output_dir="/local_disk0/output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_checkpointing=False,
    gradient_accumulation_steps=1,
    learning_rate=2e-6,
    optim="paged_adamw_32bit",
    num_train_epochs=3,
    weight_decay=1,
    evaluation_strategy="epoch",
    fp16=True,
    bf16=False,
    save_strategy="steps",
    save_steps=20,
)

train(training_args)

# COMMAND ----------
