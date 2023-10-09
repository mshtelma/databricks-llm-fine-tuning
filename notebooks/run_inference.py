# Databricks notebook source
# MAGIC %pip install torch==2.0.1

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

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

model, tokenizer = get_model_and_tokenizer(
    "/dbfs/tmp/msh/osff/morphir_lcr_starchat_v4",
    pretrained_name_or_path_tokenizer="HuggingFaceH4/starchat-beta",
    inference=True,
)

# COMMAND ----------

prompt = """<|system|>Generate test case input  and test result as json for a rule text given below. Use defined product class in test data.
<|end|>
<|user|>
Product class: IG-2-Q
Rule text in ELM:


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


<|end|>
Generate test case input  and test result as json for a rule text given below and product class IG-2-Q.
<|assistant|>
"""

generate_text(model, tokenizer, prompt, max_new_tokens=256)

# COMMAND ----------

len(prompt)

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
