# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Prompt Engineering Techniques
# MAGIC ### Active Prompting & Chain-of-Thought (CoT)
# MAGIC <hr/>
# MAGIC
# MAGIC ### Recap
# MAGIC
# MAGIC * In the last notebook, we achieved good results with Few Shot Learning
# MAGIC * However, for some of the intents, performance was really bad.
# MAGIC * Another limitation of our previous experiment is that we had a really small sample when building our testing set. We will increase the size a bit, so our results are more statistically significant.

# COMMAND ----------

# MAGIC %pip install vllm triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

sdf = spark.read.table("prompt_engineering.customer_support_intent")
df = sdf.toPandas()

# COMMAND ----------

import pandas as pd
import numpy as np

def get_top_intents_samples(df, k = 10, random_state = 123, n_samples = 5):
  
  df_arr = []

  for intent in df.intent.unique()[:k]:
    sample = df.loc[df.intent == intent, :].sample(
      n = n_samples,
      random_state = random_state
    )
    df_arr.append(sample)

  df_sampled = pd.concat(df_arr, axis = 0)
  return df_sampled.sample(frac = 1.0)

df_train = get_top_intents_samples(
  df = df,
  random_state = 123,
  n_samples = 3
)

df_test = get_top_intents_samples(
  df = df[~np.isin(df.index, df_train.index)],
  random_state = 234,
  n_samples = 100
)

# COMMAND ----------

df_test.shape

# COMMAND ----------

from vllm import LLM, SamplingParams

llm = LLM(
    model="mosaicml/mpt-7b-instruct",
    tokenizer = "EleutherAI/gpt-neox-20b",
    trust_remote_code = True
)

# COMMAND ----------

prompt_template = """
    Below is an instruction that describes a task. Write a response that appropriately completes the request.
    
    ### Instruction:
    A dataset contains 'utterances' and 'intents'. An utterance can be classified with one of the following ten intents:
    {intents}
    Below you will find an array containing three examples of utterances for each of these intents. Each example is in JSON format:
    {examples}
    Assume you are an expert in intent classification. Based on the examples above, return a JSON object with the correct intent for the utterance below. Make sure to include only the intent in the JSON.
    '{utterance}'
    ### Response:
"""

prompts = []

for i, row in df_test.iterrows():
    sample_dict_arr = df_train.loc[:, ["utterance", "intent"]].to_dict(orient = "records")
    prompt = prompt_template.format(
        examples = sample_dict_arr,
        utterance = row['utterance'],
        intents = df_train.intent.unique()
    )
    prompts.append(prompt)

sampling_params = SamplingParams(temperature=0.0)

output = llm.generate(prompts, sampling_params)

# COMMAND ----------

predicted_arr

# COMMAND ----------

import json

def post_process(output):

    predicted = []

    for generated in output:
        pred = generated.outputs[0].text.lstrip().replace("'", '"')
        predicted.append(pred)

    return predicted

predicted_arr = post_process(output)
df_test["predicted"] = predicted_arr

# COMMAND ----------

# DBTITLE 1,Analysing Results
from sklearn.metrics import classification_report

report = classification_report(df_test.intent, df_test.predicted)
print(report)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Evaluation
# MAGIC
# MAGIC * By adding more examples we were able to sequentially improve individual performance per intent, as well as overall F1 score.
# MAGIC * We will probably get even better results if we simply increase the amount of samples globally.
# MAGIC * Of course, here we have the extra samples at our disposal. In a real-life setting, we would have to manually label these extra samples, which can be costly and time consuming. At the same time, improving individual performance of intents can already push F1 score up quite significantly - specially if the intents in question have bad performance and are quite frequent.
# MAGIC * Just for confirming our assumption about extra samples, let's increase the number of examples for all intents and see the results we get.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC * By adding more few show examples we improved our metrics significantly 🚀
# MAGIC * However, that came at the cost of increased inference latency - which makes sense, since we also increased the number of tokens for each of the prompts - due to including more examples as part of the prompts
# MAGIC * In the next notebook, we'll explore strategies for improving inference latency

# COMMAND ----------

#TODO: include MLflow logging
