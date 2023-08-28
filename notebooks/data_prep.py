# Databricks notebook source
from huggingface_hub import notebook_login

notebook_login()

# COMMAND ----------

# MAGIC %md # e2e_nlg

# COMMAND ----------

from datasets import load_dataset

ds = load_dataset("e2e_nlg_cleaned")

# COMMAND ----------

from datasets import load_dataset


def process(v):
    human_reference = v["human_reference"]
    meaning_representation = v["meaning_representation"]
    return {
        "text": f"""<s>[INST] <<SYS>>Extract entities from the text given below.<</SYS>> {human_reference} [/INST] </s><s>[INST] {meaning_representation} [/INST]"""
    }


ds = (
    ds.filter(
        lambda v: len(v["human_reference"]) > 1 and len(v["meaning_representation"]) > 1
    )
    .map(process, remove_columns=["meaning_representation", "human_reference"])
    .shuffle()
)

# COMMAND ----------

# MAGIC  %sh rm -rf /dbfs/msh/llm/datasets/e2e_nlg

# COMMAND ----------

ds.save_to_disk("/dbfs/msh/llm/datasets/e2e_nlg")

# COMMAND ----------


# COMMAND ----------

from datasets import load_dataset

ds = load_dataset("e2e_nlg")["validation"].shuffle()
it = iter(ds)
for _ in range(10):
    print("'", next(it)["human_reference"], "',")

# COMMAND ----------

from datasets import load_dataset, load_from_disk

ds = load_from_disk("/dbfs/msh/llm/datasets/e2e_nlg")["train"].shuffle()
it = iter(ds)
for _ in range(10):
    print("'", next(it)["human_reference"], "',")
