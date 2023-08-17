# Databricks notebook source
from datasets import Dataset

def process(v):
  text = v["text"]
  arr = text.split("### Assistant:")
  question = arr[0].replace("### Human:", "").strip()
  answer = arr[1].strip()
  return {"text":f"""<s>[INST] <<SYS>>You are a helpful, respectful, and honest assistant. Your answers should not include any harmful, racist, sexist, or illegal content. If you don't know the answer to a question, avoid sharing false information.<</SYS>>{question} [/INST] </s><s>[INST] {answer} [/INST]"""}

slack_ds = Dataset.from_pandas(df).map(process)

# COMMAND ----------

from huggingface_hub import notebook_login
notebook_login()

# COMMAND ----------

# MAGIC %md # The Stack - ABAP

# COMMAND ----------

from datasets import load_dataset
stack_ds = load_dataset("bigcode/the-stack-dedup",  data_dir="data/abap").select_columns(["content"]).rename_column("content", "text")


# COMMAND ----------

stack_ds = stack_ds["train"].shuffle().train_test_split(test_size=0.05)
stack_ds

# COMMAND ----------

stack_ds.save_to_disk("/dbfs/msh/llm/datasets/the_stack_abap")

# COMMAND ----------

from datasets import Dataset, DatasetDict

DatasetDict.load_from_disk("/dbfs/msh/llm/datasets/the_stack_abap")

# COMMAND ----------

from datasets import load_dataset

def process(v):
  prompt = v["prompt"]
  response = v["response"]
  question = prompt.split("### Instruction:")[1].replace("### Response:", "").strip()
  return {"text":f"""<s>[INST] <<SYS>>You are a helpful, respectful, and honest assistant. Your answers should not include any harmful, racist, sexist, or illegal content. If you don't know the answer to a question, avoid sharing false information.<</SYS>> {question} [/INST] </s><s>[INST] {response} [/INST]"""}

dolly_ds = load_dataset("mosaicml/dolly_hhrlhf", split="train").map(process, remove_columns=["prompt", "response"])

# COMMAND ----------

# MAGIC %sh mkdir -p /dbfs/msh/llm/datasets/slack_qa

# COMMAND ----------

from datasets import concatenate_datasets
ds = concatenate_datasets([slack_ds, dolly_ds]).shuffle()
ds_dict = ds.train_test_split(test_size=0.1, shuffle=True)
train_ds = ds_dict["train"]
test_ds = ds_dict["test"]
ds_dict.save_to_disk("/dbfs/msh/llm/datasets/slack_qa")

# COMMAND ----------

for i in Dataset.load_from_disk("/dbfs/llm/datasets/slack_qa"):
  print(i)
