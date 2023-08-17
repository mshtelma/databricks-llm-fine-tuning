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

domain_schema = """
1. Reporting Entity
	* Entity Name
	* Entity Type (e.g., bank, non-bank financial entity, government entity, etc.)
	* Entity ID (unique identifier)
2. Product
	* Product Name
	* Product Type (e.g., loan, deposit, security, etc.)
	* Product ID (unique identifier)
3. Counterparty
	* Counterparty Name
	* Counterparty Type (e.g., retail, small business, non-financial corporate, sovereign, etc.)
	* Counterparty ID (unique identifier)
4. Exposure
	* Exposure Amount
	* Exposure Type (e.g., outstanding loan balance, deposit balance, etc.)
	* Exposure ID (unique identifier)
5. Maturity
	* Maturity Date
	* Maturity Type (e.g., short-term, long-term, etc.)
6. Currency
	* Currency Code (e.g., USD, EUR, GBP, etc.)
	* Currency Amount
7. Risk Weight
	* Risk Weight
8. Conversion
	* Conversion Flag (True/False)
9. Sub-Product
	* Sub-Product Name
	* Sub-Product Type (e.g., revolving exposure, non-revolving exposure, etc.)
10. Comment
	* Comment Text

Enumeration Items List:

1. Counterparty Type
	* Retail
	* Small Business
	* Non-Financial Corporate
	* Sovereign
	* Central Bank
	* Government Sponsored Entity (GSE)
	* Public Sector Entity (PSE)
	* Multilateral Development Bank (MDB)
	* Other Supranational
	* Pension Fund
	* Bank
	* Broker-Dealer
	* Investment Company or Advisor
	* Financial Market Utility
	* Other Supervised Non-Bank Financial Entity
2. Exposure Type
	* Outstanding Loan Balance
	* Deposit Balance
	* Security
	*
"""


# COMMAND ----------



# COMMAND ----------

DEFAULT_INPUT_MODEL = "HuggingFaceH4/starchat-beta"
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
    load_in_8bit=False
)

# COMMAND ----------

def get_prompt_starchat(query: str) -> str:
  prompt_template = "<|system|>Create the type structure in Elm using the domain schema define below.\n<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>"
  prompt = prompt_template.format(query=query)
  return prompt


# COMMAND ----------

code = generate_text(model, tokenizer, get_prompt_starchat(domain_schema), max_new_tokens=1024)[0]
print(code)

# COMMAND ----------

code = generate_text(model, tokenizer, 
                     "<|system|>Create classes in ABAP using the domain schema define below.\n<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>".format(query=domain_schema), 
                     max_new_tokens=1024)[0]
print(code)

# COMMAND ----------



# COMMAND ----------


