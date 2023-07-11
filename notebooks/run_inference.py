# Databricks notebook source
# MAGIC !wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb -O /tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
# MAGIC   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcublas-dev-11-7_11.10.1.25-1_amd64.deb -O /tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
# MAGIC   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb -O /tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
# MAGIC   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcurand-dev-11-7_10.2.10.91-1_amd64.deb -O /tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb && \
# MAGIC   dpkg -i /tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
# MAGIC   dpkg -i /tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
# MAGIC   dpkg -i /tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
# MAGIC   dpkg -i /tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb


# COMMAND ----------

# MAGIC %pip install torch==2.0.1

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# COMMAND ----------

# MAGIC %pip install pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python
# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# COMMAND ----------

import logging

import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("sh.command").setLevel(logging.ERROR)
# COMMAND ----------
from databricks_llm.inference import generate_text, generate_text_for_df
from databricks_llm.model_utils import get_model_and_tokenizer
from databricks_llm.notebook_utils import get_dbutils

# COMMAND ----------
DEFAULT_INPUT_MODEL = "mosaicml/mpt-7b-instruct"
SUPPORTED_INPUT_MODELS = ["mosaicml/mpt-30b-instruct", "mosaicml/mpt-7b-instruct"]
# COMMAND ----------
# get_dbutils().widgets.text("num_gpus", "4", "num_gpus")
get_dbutils().widgets.combobox(
    "pretrained_name_or_path",
    DEFAULT_INPUT_MODEL,
    SUPPORTED_INPUT_MODELS,
    "pretrained_name_or_path",
)
# COMMAND ----------
questions = [
    "Write a love letter to Edgar Allan Poe",
    "Write a tweet announcing a new language model called Dolly from Databricks",
]
# COMMAND ----------
pretrained_name_or_path = get_dbutils().widgets.get("pretrained_name_or_path")
model, tokenizer = get_model_and_tokenizer(pretrained_name_or_path, inference=True)
# COMMAND ----------

results = generate_text(model, tokenizer, questions)
for result in results:
    print(result)
# COMMAND ----------
q = questions + questions + questions + questions
q_df = pd.DataFrame(data={"txt": q}, columns=["txt"])

res_df = generate_text_for_df(model, tokenizer, q_df, "txt", "gen_txt", batch_size=2)
display(res_df)
