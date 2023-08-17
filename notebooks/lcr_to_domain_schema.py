# Databricks notebook source
# MAGIC
# MAGIC %pip install torch==2.0.1

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from huggingface_hub import login
login(token="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")

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




# COMMAND ----------



# COMMAND ----------

DEFAULT_INPUT_MODEL = "meta-llama/Llama-2-70b-chat-hf"
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

model, tokenizer = get_model_and_tokenizer(
    pretrained_name_or_path,
    pretrained_name_or_path_tokenizer=pretrained_name_or_path,
    inference=True,
    load_in_8bit=True
)

# COMMAND ----------

# MAGIC %md # Generation using fine-tuned Llama v2  & Llama v2 Prompt Structure

# COMMAND ----------

with open("/dbfs/msh/lcr/lcr_doc.txt", "r") as file:
    doc_text = file.read()
import re
doc_text = re.sub(r'\s+', ' ', doc_text)
doc_text = doc_text[doc_text.index("Field Definitions", 4000):]

# COMMAND ----------

doc_tokens = tokenizer(doc_text, padding=True, truncation=False, return_tensors="np").input_ids[0]

# COMMAND ----------

def get_prompt_llama(query: str) -> str:
    return f"""<s>[INST] <<SYS>>In the context you can find reporting requirements. Extract domain data model schema as SQL. Include enumeration items list for enumerated entities.  <</SYS>>Context: {query} [/INST]###"""

# COMMAND ----------

import numpy as np
import torch 

doc_tokens = tokenizer(doc_text, padding=True, truncation=True, return_tensors="np").input_ids[0]
start_idx = 0
step = 2000
for i in range(3):
  start = start_idx+step*i
  end = start_idx+step*(i+1)+500
  print(start, end)
  curr_tokens = doc_tokens[start:end]
  prompt_str = get_prompt_llama(tokenizer.decode(curr_tokens, skip_special_tokens=True))
  curr_prompt = tokenizer(prompt_str , padding=True, truncation=True, return_tensors="pt").input_ids
  with torch.no_grad():
      output_tokens = model.generate(
          use_cache=True,
          input_ids=curr_prompt,
          max_new_tokens=512,
          temperature=0.2,
          top_p=10,
          num_return_sequences=1,
          pad_token_id=tokenizer.eos_token_id,
          eos_token_id=tokenizer.eos_token_id,
      )[0]
      prompt_length = len(curr_prompt[0])
      output_tokens = np.trim_zeros(output_tokens.cpu().numpy())
      output_tokens = output_tokens[prompt_length:]
      res = tokenizer.decode(output_tokens, skip_special_tokens=True)
      print(res)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

start_idx = 35000
step = 2000
for i in range(10):
  ctx = doc_text[(start_idx+step*i):(start_idx+step*(i+1))]
  prompt = get_prompt_llama(ctx)
  res = generate_text(model, tokenizer, prompt)
  print(ctx)
  print(res)

# COMMAND ----------

q_df = pd.DataFrame(data={"txt": questions}, columns=["txt"])

res_df = generate_text_for_df(
    model,
    tokenizer,
    q_df,
    "txt",
    "gen_txt",
    batch_size=2,
    #gen_prompt_fn=get_prompt_llama,
    #post_process_fn=post_process,
    max_new_tokens=512,
    temperature=0.1,
)
display(res_df)

# COMMAND ----------


