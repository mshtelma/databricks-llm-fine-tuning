# Databricks notebook source

# MAGIC %pip install torch==2.0.1

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# COMMAND ----------

# MAGIC %pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python
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
