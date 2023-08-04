# Databricks notebook source
import pandas as pd
df = pd.read_csv("../slack_qa.csv")
display(df)

# COMMAND ----------

from datasets import Dataset

def process(v):
  text = v["text"]
  arr = text.split("### Assistant:")
  question = arr[0].replace("### Human:", "").strip()
  answer = arr[1].strip()
  return {"text":f"""<s>[INST] <<SYS>>You are a helpful, respectful, and honest assistant. Your answers should not include any harmful, racist, sexist, or illegal content. If you don't know the answer to a question, avoid sharing false information.<</SYS>>{question} [/INST] </s><s>[INST] {answer} [/INST]"""}

slack_ds = Dataset.from_pandas(df).map(process)

# COMMAND ----------

from datasets import load_dataset

def process(v):
  prompt = v["prompt"]
  response = v["response"]
  question = prompt.split("### Instruction:")[1].replace("### Response:", "").strip()
  return {"text":f"""<s>[INST] <<SYS>>You are a helpful, respectful, and honest assistant. Your answers should not include any harmful, racist, sexist, or illegal content. If you don't know the answer to a question, avoid sharing false information.<</SYS>> {question} [/INST] </s><s>[INST] {response} [/INST]"""}

dolly_ds = load_dataset("mosaicml/dolly_hhrlhf", split="train").map(process, remove_columns=["prompt", "response"])

# COMMAND ----------

from datasets import concatenate_datasets
ds = concatenate_datasets([slack_ds, dolly_ds]).shuffle()
ds.save_to_disk("/dbfs/llm/datasets/slack_qa")

# COMMAND ----------

for i in Dataset.load_from_disk("/dbfs/llm/datasets/slack_qa"):
  print(i)

# COMMAND ----------


