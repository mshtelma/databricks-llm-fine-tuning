# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Few Shot Learning
# MAGIC
# MAGIC In Machine Learning, **Few Shot Learning** (FSL) is a Machine Learning framework that enables a pre-trained model to generalize over new categories of data (that the pre-trained model has not seen during training) using only a few labeled samples per class. It falls under the paradigm of meta-learning (meta-learning means learning to learn).
# MAGIC
# MAGIC Within **Prompt Engineering**, FSL is also known as **Few Shot Prompting** - you include some examples related to the task that you want the model to perform as part of your prompts.
# MAGIC
# MAGIC ### Few Shot Learning Example
# MAGIC
# MAGIC It can be tricky for LLMs to do sentiment analysis when there are aspects in the prompt such as *ambiguity* or *irony*.
# MAGIC
# MAGIC #### Sentiment Classification: Zero-Shot Learning
# MAGIC
# MAGIC Suppose you tried **Zero Shot Learning** for a **Sentiment Analysis** problem with an LLM of your choice. You noticed that the performance is below what you expect:
# MAGIC
# MAGIC * **Instruction**: *Please classify the following sentence as either POSITIVE or NEGATIVE, depending on the overall sentiment: "This movie is good, considering that you think it's good to sit in the movie theater sleeping for 2 hours."*
# MAGIC * **LLM Answer**: POSITIVE (expected classification: NEGATIVE)
# MAGIC
# MAGIC #### Sentiment Classification: Few-Shot Learning
# MAGIC
# MAGIC You then try to provide some classification examples containing *irony*, so that the model is able to distinguish those and properly classify similar ones:
# MAGIC <br/>
# MAGIC <br/>
# MAGIC
# MAGIC * **Instruction**: *Given the following sentiment classification examples:*
# MAGIC <br/>
# MAGIC
# MAGIC *'I liked this movie as much as I like standing in the rain' (NEGATIVE)*
# MAGIC <br/>
# MAGIC
# MAGIC *'If you would like to spend 10 USD for nothing then this movie is good for you' (NEGATIVE)*
# MAGIC
# MAGIC *Please classify the following sentence as either POSITIVE, NEUTRAL or NEGATIVE, depending on the overall sentiment.*
# MAGIC <br/>
# MAGIC
# MAGIC *"This movie is good, considering that you think it's good to sit in the movie theater sleeping for 2 hours."*
# MAGIC <br/>
# MAGIC <br/>
# MAGIC * **LLM Answer**: NEGATIVE (expected classification: NEGATIVE)
# MAGIC
# MAGIC In the cells below, we will try to apply the same concept to an **intent classification** dataset. For this purpose, we will experiment with MosaicML's `mpt-7b-instruct` model.
# MAGIC
# MAGIC A potential caveat with Few Shot Learning is given all the few shot examples that we provide, our prompt might be larger than the maximum **context length** for a particular model. 
# MAGIC
# MAGIC MPT family of models allows you to configure maximum context window size, which mitigates this issue. Of course, it is also possible to extend context window sizes by fine tuning LLMs, but we want to skip that step for this exercise.

# COMMAND ----------

# DBTITLE 1,Declaring and Configuring MPT-7b
import transformers
import torch

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of mpt-7b-instruct in https://huggingface.co/mosaicml/mpt-7b-instruct/commits/main

name = "mosaicml/mpt-7b-instruct"
config = transformers.AutoConfig.from_pretrained(
  name,
  trust_remote_code=True
)
config.attn_config['attn_impl'] = 'triton'
config.init_device = 'cuda'
config.max_seq_len = 4000

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.bfloat16,
  trust_remote_code=True,
  cache_dir="/local_disk0/.cache/huggingface/",
  revision="bbe7a55d70215e16c00c1825805b81e4badb57d7"
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
  "EleutherAI/gpt-neox-20b",
  padding_side="left"
)

generator = transformers.pipeline(
  "text-generation",
  model=model, 
  config=config, 
  tokenizer=tokenizer,
  torch_dtype=torch.bfloat16,
  device=0
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Output Formats
# MAGIC
# MAGIC Different LLMs have distinct behaviours when it comes to the output format.
# MAGIC
# MAGIC For instance, we might want to have our LLM return JSON-formatted outputs, so that we can easily transform these outputs into a Pandas or Spark DataFrame for persisting them somewhere later on and also evaluating the outputs.
# MAGIC
# MAGIC To simplify this, we will parse our LLM responses to JSON. To achieve this, we'll leverage [JSONFormer](https://github.com/1rgs/jsonformer), a nice, lightweight Python Package that parses JSON payloads from LLM outputs.
# MAGIC
# MAGIC Sample usage:
# MAGIC
# MAGIC ```python
# MAGIC from jsonformer import Jsonformer
# MAGIC from transformers import AutoModelForCausalLM, AutoTokenizer
# MAGIC
# MAGIC model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b")
# MAGIC tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")
# MAGIC
# MAGIC json_schema = {
# MAGIC     "type": "object",
# MAGIC     "properties": {
# MAGIC         "name": {"type": "string"},
# MAGIC         "age": {"type": "number"},
# MAGIC         "is_student": {"type": "boolean"},
# MAGIC         "courses": {
# MAGIC             "type": "array",
# MAGIC             "items": {"type": "string"}
# MAGIC         }
# MAGIC     }
# MAGIC }
# MAGIC
# MAGIC prompt = "Generate a person's information based on the following schema:"
# MAGIC jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
# MAGIC generated_data = jsonformer()
# MAGIC
# MAGIC print(generated_data)
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Declaring our Generation Wrapper Function
from jsonformer import Jsonformer

def generate_text(prompt, **kwargs):

  json_schema = {
    "type": "object",
    "properties": {
      "utterance": {"type": "string"},
      "intent": {"type": "string"}
    }
  }

  if "max_new_tokens" not in kwargs:
    kwargs["max_new_tokens"] = 100
  
  kwargs.update(
    {
      "pad_token_id": tokenizer.eos_token_id,
      "eos_token_id": tokenizer.eos_token_id,
    }
  )
  
  if isinstance(prompt, str):
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
    response = jsonformer()

  elif isinstance(prompt, list):
    response = []
    for prompt_ in prompt:
      jsonformer = Jsonformer(model, tokenizer, json_schema, prompt_)
      generated_text = jsonformer()
      response.append(generated_text)

  return response

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## The Dataset
# MAGIC <br/>
# MAGIC
# MAGIC * For this **intent classification** example, we will use a **customer support intent dataset** from Hugging Face.
# MAGIC * Once we have downloaded it, we will save it as Delta Table, so that we don't need to download it again. That will also allow us to have it in optimized format should we want to do any preprocessing, visualization etc.

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC CREATE DATABASE IF NOT EXISTS prompt_engineering

# COMMAND ----------

# DBTITLE 1,Downloading our Dataset from Hugging Face and Saving to Dclta
import datasets

ds = datasets.load_dataset("bitext/customer-support-intent-dataset")
df = ds["train"].to_pandas()

sdf = spark.createDataFrame(df)
sdf.write.saveAsTable("prompt_engineering.customer_support_intent", mode = "overwrite")

#Take a small, random sample from this dataset and display it
display(df.sample(frac=0.2, random_state = 123).head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Selecting our Few Shot Examples
# MAGIC <br/>
# MAGIC
# MAGIC * The dataset we're using contains multiple different intents.
# MAGIC * For our experiment to be minimally successful, we need to provide at least some few shot examples for each and every intent.
# MAGIC * For simplicity purposes, we will select the top 10 most frequent intents. Then, for each of these top 10 intents, we will randomly sample 5 different utterances/questions.
# MAGIC * Concretely, our dataframe containing few shot examples will be our **training set** (although we're not really *training* our model per se; this will just be our convention).
# MAGIC * The remaining part of our dataset will be our **testing set**, which we'll use to evaluate how our Few Shot experiment performs.

# COMMAND ----------

# DBTITLE 1,Creating a sampled dataset for Few Shot Examples
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
  n_samples = 5
)

# COMMAND ----------

# DBTITLE 1,Creating Prompts using Few Shot Learning
import tqdm

def train_predict(prompt_template, df_train, df_test):

  sample_dict_arr = df_train.to_dict(orient="records")
  examples = sample_dict_arr

  utterances = df_test.utterance.values
  intents = df_test.intent.values
  utterances_intents = list(zip(utterances, intents))
  result = []

  for utterance_intent in tqdm.tqdm(utterances_intents):

    formatted_prompt = prompt_template.format(
      intents = df_train.intent.unique(),
      examples = examples,
      utterance = utterance_intent[0]
    )

    # Setting temperature = 0 and do_sample = False
    # (to be as deterministic as possible)

    response = generate_text(
      prompt = formatted_prompt,
      temperature = 0.1,
      top_p = 0.15,
      top_k = 5,
      do_sample = True
    )
    response["actual_intent"] = utterance_intent[1]
    result.append(response)

  return result

prompt_template = """
    Below you will find an array containing three examples of utterances for each of these intents. Each example is in JSON format:
    {examples}
    Based on the examples above, return a JSON object with the actual utterance, and the right intent for the utterance below:
    '{utterance}'
  """

result = train_predict(prompt_template, df_train, df_test)

# COMMAND ----------

# DBTITLE 1,Analysing Results
from sklearn.metrics import classification_report

result_df = pd.DataFrame.from_dict(result)
report = classification_report(result_df.actual_intent, result_df.intent)
print(report)

# COMMAND ----------

result_df[result_df.actual_intent == "change_address"]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Results Evaluation
# MAGIC
# MAGIC <br/>
# MAGIC
# MAGIC * We were able to get a (weighted) F1 Score of 0.93, which is not bad at all!
# MAGIC * Still, for some classes/intents the score is really bad:
# MAGIC   * `change_address`
# MAGIC * If we look again at both our training and testing sets we'll realise that these intents are not even there.
# MAGIC * Let's change our prompt, so that our model becomes more assertive in that sense.
# MAGIC   * We'll surround our **utterances** with single quotes
# MAGIC   * We'll also add a bit of text to try to make our generations more assertive

# COMMAND ----------

prompt_template = """
    Below you will find an array containing three examples of utterances for each of these intents. Each example is in JSON format:
    {examples}
    Based on the examples above, return a JSON object with the actual utterance, and the right intent for the utterance below:
    '{utterance}'
    Make sure that your answer contains one of the intents below:
    {intents}
"""

result = train_predict(prompt_template, df_train, df_test)

# COMMAND ----------

result_df = pd.DataFrame.from_dict(result)
report = classification_report(result_df.actual_intent, result_df.intent)
print(report)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC * Of course, this look super good 😀
# MAGIC * We just need to have in mind that our testing set is quite limited - mainly for experimentation / latency purposes.
# MAGIC * In the next notebook we'll expand our testing set, and also introduce other techniques, such as **Chain of Thought (CoT)** and **Active Prompting**.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Reference
# MAGIC
# MAGIC <hr />
# MAGIC
# MAGIC * [Everything You Need to Know About Few-Shot Learning](https://blog.paperspace.com/few-shot-learning/)
# MAGIC * [Few-Shot Prompting](https://www.promptingguide.ai/techniques/fewshot)
