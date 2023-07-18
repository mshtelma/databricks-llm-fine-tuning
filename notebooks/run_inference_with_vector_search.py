# Databricks notebook source

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
from databricks_llm.inference import (
    generate_text,
    generate_text_for_df,
    generate_text_with_context,
    load_vector_db,
)
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
vector_database_location = get_dbutils().widgets.text(
    "vector_database_location", "", "vector_database_location"
)
vector_database_collection_name = get_dbutils().widgets.text(
    "vector_database_collection_name", "", "vector_database_collection_name"
)

# COMMAND ----------
questions = [
    "Write a love letter to Edgar Allan Poe",
    "Write a tweet announcing a new language model called Dolly from Databricks",
]
# COMMAND ----------
pretrained_name_or_path = get_dbutils().widgets.get("pretrained_name_or_path")
model, tokenizer = get_model_and_tokenizer(pretrained_name_or_path, inference=True)
vector_database_location = get_dbutils().widgets.get("vector_database_location")
vector_database_collection_name = get_dbutils().widgets.get(
    "vector_database_collection_name"
)

# COMMAND ----------
prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

  Instruction: 
  Answer the question using only information in the following paragraphs to answer the question at the end. Explain the answer with reference to these paragraphs. If you don't know, say that you do not know.

  {context}
 
  Question: {question}

  Response: 
  """
# COMMAND ----------
doc_ctx = load_vector_db(
    vector_database_collection_name,
    vector_database_location,
    model_name="sentence-transformers/all-mpnet-base-v2",
)
# COMMAND ----------
question = ""
filter = ""
results = generate_text_with_context(
    model, tokenizer, prompt_template, question, doc_ctx, doc_ctx_filter={"": filter}
)
for result in results:
    print(result)
