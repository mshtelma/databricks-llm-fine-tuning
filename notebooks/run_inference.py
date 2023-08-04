# Databricks notebook source
# MAGIC
# MAGIC %pip install torch==2.0.1

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from huggingface_hub import notebook_login

notebook_login()

# COMMAND ----------

import os

os.environ["HF_HOME"] = "/local_disk0/hf"
os.environ["HF_DATASETS_CACHE"] = "/local_disk0/hf"
os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"

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

DEFAULT_INPUT_MODEL = "meta-llama/Llama-2-7b-chat-hf"
SUPPORTED_INPUT_MODELS = [
    "mosaicml/mpt-30b-instruct",
    "mosaicml/mpt-7b-instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
]

# COMMAND ----------

get_dbutils().widgets.combobox(
    "pretrained_name_or_path",
    DEFAULT_INPUT_MODEL,
    SUPPORTED_INPUT_MODELS,
    "pretrained_name_or_path",
)

# COMMAND ----------

pretrained_name_or_path = get_dbutils().widgets.get("pretrained_name_or_path")

# COMMAND ----------

questions = [
    "Write a love letter to Edgar Allan Poe",
    "Write a tweet announcing a new language model called Dolly from Databricks",
    "Explain the novelty of GPT-3 in 5 bullet points",
    "PLease explain what is Machine Learning?",
    "How are you?",
]

# COMMAND ----------

model, tokenizer = get_model_and_tokenizer(
    pretrained_name_or_path,
    pretrained_name_or_path_tokenizer=pretrained_name_or_path,
    inference=True,
)

# COMMAND ----------

# MAGIC %md # Generation using fine-tuned Llama v2  & Llama v2 Prompt Structure

# COMMAND ----------

def get_prompt_llama(query: str) -> str:
    return f"""<s>[INST] <<SYS>>You are a helpful, respectful, and honest assistant. Your answers should not include any harmful, racist, sexist, or illegal content. If you don't know the answer to a question, avoid sharing false information.<</SYS>>{query} [/INST]###"""


def post_process(s: str) -> str:
    _idx = s.find("###")
    if _idx > 0:
        return s[_idx + 4 :].strip()
    else:
        return s


# COMMAND ----------

q_df = pd.DataFrame(data={"txt": questions}, columns=["txt"])

res_df = generate_text_for_df(
    model,
    tokenizer,
    q_df,
    "txt",
    "gen_txt",
    batch_size=2,
    gen_prompt_fn=get_prompt_llama,
    post_process_fn=post_process,
    max_new_tokens=512,
    temperature=0.1,
)
display(res_df)
