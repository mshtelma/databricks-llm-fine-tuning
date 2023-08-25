# Databricks notebook source

# MAGIC %pip install torch==2.0.1

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt
# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# COMMAND ----------

import logging

from databricks_llm.data_prep.prepare_vector_db import create_vector_db, split_documents

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("sh.command").setLevel(logging.ERROR)
# COMMAND ----------
from databricks_llm.notebook_utils import get_dbutils

# COMMAND ----------

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
SUPPORTED_EMBEDDING_MODELS = ["sentence-transformers/all-mpnet-base-v2"]
# COMMAND ----------
get_dbutils().widgets.text(
    "dbfs_output_location", "/dbfs/llm/vectorsearch", "dbfs_output_location"
)
get_dbutils().widgets.combobox(
    "embedding_model_name_or_path",
    DEFAULT_EMBEDDING_MODEL,
    SUPPORTED_EMBEDDING_MODELS,
    "embedding_model_name_or_path",
)
get_dbutils().widgets.text(
    "input_documents_table",
    "",
    "input_documents_table",
)
get_dbutils().widgets.text(
    "input_documents_location",
    "",
    "input_documents_location",
)
get_dbutils().widgets.text(
    "collection_name",
    "documents",
    "collection_name",
)

# COMMAND ----------

embedding_model_name_or_path = get_dbutils().widgets.get("embedding_model_name_or_path")
input_documents_table = get_dbutils().widgets.get("input_documents_table")
input_documents_location = get_dbutils().widgets.get("input_documents_table")
dbfs_output_location = get_dbutils().widgets.get("dbfs_output_location")
collection_name = get_dbutils().widgets.get("collection_name")


# COMMAND ----------
if input_documents_table:
    src_df = spark.read.table(input_documents_table)
else:
    src_df = spark.read.load(input_documents_location)

splitted_df = split_documents(src_df, col_name="text")
display(splitted_df)
# COMMAND ----------

create_vector_db(
    splitted_df,
    value_col_name="text",
    metadata_col_name="file_name",
    collection_name=collection_name,
    path=dbfs_output_location,
    embedding_model_name=embedding_model_name_or_path,
)
