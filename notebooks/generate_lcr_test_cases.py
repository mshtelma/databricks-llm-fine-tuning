# Databricks notebook source
# MAGIC %sh ./install_morphir.sh

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------



# COMMAND ----------

import random 
from datasets import Dataset
from morphir import morphir_get_test_result, morphir_test

product_classes = ["E-1-Q", "E-2-Q", "IG-1-Q", "IG-2-Q", "W1", "W2", "W3"]

samples = []
for i in range(25000):
  clazz = random.choice(product_classes)
  rec = {
    'currency': ['USD'],
    'converted': random.choice([True, False]),
    'reportingEntity': 'Foo',
    'product': ['capacity'],
    'subProduct': random.choice([None, "1", "2"]),
    'marketValue': random.randint(0, 10000),
    'lendableValue': str(random.randint(0, 10000)),
    'maturityBucket': ['open'],
    'forwardStartAmount': random.choice([None, "1", "2"]),
    'forwardStartBucket': random.choice([None, "1", "2"]),
    'collateralClass': clazz,
    'treasuryControl': random.choice([True, False]),
    'accountingDesignation': 'foo',
    'effectiveMaturityBucket': ['open'],
    'encumbranceType': None,
    'internalCounterparty': None,
    'businessLine': 'foo'
  }
  res = morphir_get_test_result(rec)
  samples.append({"class":clazz, "test_input": rec, "test_result":res})

tests_inpt_output_ds = Dataset.from_list(samples)

# COMMAND ----------

tests_inpt_output_ds.save_to_disk("/dbfs/tmp/msh/osff/morphir_lcr_raw_test_cases_v2_25k")

# COMMAND ----------

elm_rule_impl = """

isHQLALevel1 : CollateralClass -> Bool
isHQLALevel1 class =
    List.member class [ a_0_Q, a_1_Q, a_2_Q, a_3_Q, a_4_Q, a_5_Q, s_1_Q, s_2_Q, s_3_Q, s_4_Q, cB_1_Q, cB_2_Q ]


isHQLALevel2A : CollateralClass -> Bool
isHQLALevel2A class =
    List.member class [ g_1_Q, g_2_Q, g_3_Q, s_5_Q, s_6_Q, s_7_Q, cB_3_Q ]


isHQLALevel2B : CollateralClass -> Bool
isHQLALevel2B class =
    List.member class [ e_1_Q, e_2_Q, iG_1_Q, iG_2_Q ]


isHQLA : CollateralClass -> Bool
isHQLA class =
    isHQLALevel1 class || isHQLALevel2A class || isHQLALevel2B class


{-| (1) High-Quality Liquid Assets (Subpart C, §.20-.22)
-}
rule_1_section_20_c_1 : Assets -> Maybe Float
rule_1_section_20_c_1 flow =
    if
        List.member flow.product [ i_A_1, i_A_2 ]
            -- Sub-Product: Not Currency and Coin
            --&& (flow.subProduct |> Maybe.map (\subProduct -> not (SubProduct.isCurrencyAndCoin subProduct)) |> Maybe.withDefault True)
            && (flow.subProduct /= Just currency_and_coin)
            ---- Collateral Class: E-1-Q; E-2-Q; IG-1-Q; IG-2-Q
            && CollateralClass.isHQLALevel2B flow.collateralClass
            -- Forward Start Amount: NULL
            && (flow.forwardStartAmount == Nothing)
            -- Forward Start Bucket: NULL
            && (flow.forwardStartBucket == Nothing)
            -- Encumbrance Type: Null
            && (flow.encumbranceType == Nothing)
            -- Treasury Control: Y
            && (flow.treasuryControl == True)
    then
        Just flow.marketValue

    else
        Nothing

"""

# COMMAND ----------

def process(v):
    clazz = v["class"]
    test_input = v["test_input"]
    test_result = v["test_result"]
    rec = {"test_input": test_input, "test_result": test_result}
    prompt = f"""
<|system|>
You are a code generating assistant. Generate only JSON output.
<|end|>
<|user|>
Generate test case input and test result as json for a rule text given below and product class {clazz}:
Rule text in ELM:\n{elm_rule_impl}.
Generate test case input and test result as json for the product class {clazz}.
<|end|>
<|assistant|>
{rec}
<|end|>"""
    return {"text": prompt}


tests_with_prompt_ds = tests_inpt_output_ds.map(
    process, remove_columns=["class", "test_input", "test_result"]
).shuffle()

# COMMAND ----------

display(tests_with_prompt_ds.to_pandas())

# COMMAND ----------

print(tests_with_prompt_ds.to_pandas()["text"][2])

# COMMAND ----------

len(tests_with_prompt_ds.to_pandas()["text"][2])

# COMMAND ----------

tests_with_prompt_ds_dict = tests_with_prompt_ds.train_test_split(test_size=0.1, shuffle=True)
tests_with_prompt_ds_dict.save_to_disk("/dbfs/tmp/msh/osff/morphir_lcr_test_data_v4_25k")

# COMMAND ----------

!ls /dbfs/tmp/msh/osff/morphir_lcr_test_data_v3

# COMMAND ----------

from datasets import load_from_disk
ds = load_from_disk("/dbfs/tmp/msh/osff/morphir_lcr_test_data_v3")["train"]
it = iter(ds)
for _ in range(10):
  print(next(it)) 

# COMMAND ----------

test_input = {
  'currency': ['USD'],
  'converted': True,
  'reportingEntity': 'Foo',
  'product': ['capacity'],
  'subProduct': None,
  'marketValue': 200000,
  'lendableValue': '1000',
  'maturityBucket': ['open'],
  'forwardStartAmount': None,
  'forwardStartBucket': None,
  'collateralClass': 'E-2-Q',
  'treasuryControl': True,
  'accountingDesignation': 'foo',
  'effectiveMaturityBucket': ['open'],
  'encumbranceType': None,
  'internalCounterparty': None,
  'businessLine': 'foo'
}

# COMMAND ----------



from morphir import morphir_get_test_result, morphir_test
morphir_test_results = morphir_test({
  'currency': ['USD'],
  'converted': True,
  "product": ['capacity'],
  'reportingEntity': 'Foo',
  "subProduct": None,
  'lendableValue': '1000',
  'maturityBucket': ['open'],
  "collateralClass": "E-2-Q",
  "forwardStartAmount": None,
  "forwardStartBucket": None,
  "encumbranceType": None,
  "treasuryControl": True,
  "marketValue": 1000000,
  'accountingDesignation': 'foo',
  'effectiveMaturityBucket': ['open'],
  'encumbranceType': None,
  'internalCounterparty': None,
  'businessLine': 'foo'
}, 0)
morphir_test_results.stdout.decode('UTF-8').split('\n')

# COMMAND ----------

morphir_test_results.stdout.decode('UTF-8').split('\n')

# COMMAND ----------

from morphir import morphir_get_test_result
morphir_get_test_result(test_input)

# COMMAND ----------

morphir_test_results

# COMMAND ----------

morphir_test_results.stdout.decode('UTF-8').split('\n')

# COMMAND ----------

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

elm_rules = """

isHQLALevel1 : CollateralClass -> Bool
isHQLALevel1 class =
    List.member class [ a_0_Q, a_1_Q, a_2_Q, a_3_Q, a_4_Q, a_5_Q, s_1_Q, s_2_Q, s_3_Q, s_4_Q, cB_1_Q, cB_2_Q ]


isHQLALevel2A : CollateralClass -> Bool
isHQLALevel2A class =
    List.member class [ g_1_Q, g_2_Q, g_3_Q, s_5_Q, s_6_Q, s_7_Q, cB_3_Q ]


isHQLALevel2B : CollateralClass -> Bool
isHQLALevel2B class =
    List.member class [ e_1_Q, e_2_Q, iG_1_Q, iG_2_Q ]


isHQLA : CollateralClass -> Bool
isHQLA class =
    isHQLALevel1 class || isHQLALevel2A class || isHQLALevel2B class


{-| (1) High-Quality Liquid Assets (Subpart C, §.20-.22)
-}
rule_1_section_20_c_1 : Assets -> Maybe Float
rule_1_section_20_c_1 flow =
    if
        List.member flow.product [ i_A_1, i_A_2 ]
            -- Sub-Product: Not Currency and Coin
            --&& (flow.subProduct |> Maybe.map (\subProduct -> not (SubProduct.isCurrencyAndCoin subProduct)) |> Maybe.withDefault True)
            && (flow.subProduct /= Just currency_and_coin)
            ---- Collateral Class: E-1-Q; E-2-Q; IG-1-Q; IG-2-Q
            && CollateralClass.isHQLALevel2B flow.collateralClass
            -- Forward Start Amount: NULL
            && (flow.forwardStartAmount == Nothing)
            -- Forward Start Bucket: NULL
            && (flow.forwardStartBucket == Nothing)
            -- Encumbrance Type: Null
            && (flow.encumbranceType == Nothing)
            -- Treasury Control: Y
            && (flow.treasuryControl == True)
    then
        Just flow.marketValue

    else
        Nothing

"""

# COMMAND ----------

DEFAULT_INPUT_MODEL = "HuggingFaceH4/starchat-beta"
SUPPORTED_INPUT_MODELS = [
    "codellama/CodeLlama-7b-Instruct-hf",
    "HuggingFaceH4/starchat-beta",
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
  prompt_template = "<|system|You are a coding assistent. Generate records which can be used for testing the rule below.\n<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>"
  prompt = prompt_template.format(query=query)
  return prompt


# COMMAND ----------

prompt = f"ELM Rule implementation:\n {elm_rules}\nGenerate test record as json which can be used to test this rule mentioned above and expected output of the rule."
code = generate_text(model, tokenizer, get_prompt_starchat(prompt), max_new_tokens=1024)[0]
print(code)

# COMMAND ----------

test_output = 200001
test_input = {
  'currency': ['USD'],
  'converted': True,
  'reportingEntity': 'Foo',
  'product': ['capacity'],
  'subProduct': None,
  'marketValue': 200000,
  'lendableValue': '1000',
  'maturityBucket': ['open'],
  'forwardStartAmount': None,
  'forwardStartBucket': None,
  'collateralClass': 'E-2-Q',
  'treasuryControl': True,
  'accountingDesignation': 'foo',
  'effectiveMaturityBucket': ['open'],
  'encumbranceType': None,
  'internalCounterparty': None,
  'businessLine': 'foo'
}p-----------------------------------------------------------------------------------------------------------------===-

# COMMAND ----------

code = generate_text(model, tokenizer, 
                     "<|system|>Create classes in ABAP using the domain schema define below.\n<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>".format(query=domain_schema), 
                     max_new_tokens=1024)[0]
print(code)

# COMMAND ----------



# COMMAND ----------


