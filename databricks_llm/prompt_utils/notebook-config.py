# Databricks notebook source
import torch

# COMMAND ----------

if 'config' not in locals():
  config = {}

# COMMAND ----------

# DBTITLE 1,Use Case
config['use-case']="dss_few_shot_prompt_comparison"

# COMMAND ----------

# Define the model we would like to use
# config['model_id'] = 'openai'
# config['model_id'] = 'meta-llama/Llama-2-70b-chat-hf'
# config['model_id'] = 'meta-llama/Llama-2-13b-chat-hf'
config['model_id'] = 'mosaicml/mpt-30b-chat'
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
config['use_azure_formrecognizer'] = True

# COMMAND ----------

# DBTITLE 1,Create database
config['database_name'] = 'dss_fine_tune_session'

# create database if not exists
_ = spark.sql(f"create database if not exists {config['database_name']}")

# set current datebase context
_ = spark.catalog.setCurrentDatabase(config['database_name'])

# COMMAND ----------

# DBTITLE 1,Set Environmental Variables for tokens
import os

if config['model_id'] == 'openai':
  os.environ['OPENAI_API_KEY'] = '<add open AI key>'

if "Llama-2" in config['model_id']:
  config['HUGGING_FACE_HUB_TOKEN'] = '<add hf keys>'

# COMMAND ----------

# DBTITLE 1,Set document path
config['hf_dataset'] = f"e2e_nlg"
# config['vector_store_path'] = f"/dbfs/{username}/qabot/vector_store/{config['model_id']}/{config['use-case']}" # /dbfs/... is a local file system representation

# COMMAND ----------

if config['use_azure_formrecognizer'] == True:
  config['formendpoint'] = 'https://howden-test.cognitiveservices.azure.com/'
  config['formkey'] = 'bdbca92002404c2588c729a8a33c6e10'

# COMMAND ----------

# DBTITLE 1,mlflow settings
import mlflow
config['registered_model_name'] = f"{config['use-case']}"
config['model_uri'] = f"models:/{config['registered_model_name']}/production"
_ = mlflow.set_experiment('/Users/{}/{}'.format(username, config['registered_model_name']))

# COMMAND ----------

# DBTITLE 1,Set model configs
if config['model_id'] == "openai":
  # Set the embedding vector and model  ####
  config['embedding_model'] = 'text-embedding-ada-002'
  config['openai_chat_model'] = "gpt-3.5-turbo"
  # Setup prompt template ####
  config['template'] = """You are a helpful assistant built by Databricks, you are good at helping to answer a question based on the context provided, the context is a document. If the context does not provide enough relevant information to determine the answer, just say I don't know. If the context is irrelevant to the question, just say I don't know. If you did not find a good answer from the context, just say I don't know. If the query doesn't form a complete question, just say I don't know. If there is a good answer from the context, try to summarize the context to answer the question.
  Given the context: {context}. Answer the question {question}."""
  config['temperature'] = 0.15

elif config['model_id'] == 'mosaicml/mpt-30b-chat' :
  # Setup prompt template ####
  config['embedding_model'] = 'intfloat/e5-large-v2'

  # Model parameters
  config['model_kwargs'] = {}
  config['pipeline_kwargs']={"temperature":  0.10,
                            "max_new_tokens": 256}
  
  config['template'] = """<|im_start|>system\n- You are an assistant which helps extracts important entities.If the input is incorrect just say I cannot answer the questions.Do not add entities which are not explicitly mentioned \n<|im_end|>\n<|im_start|>user\n text: {context} \n <|im_end|><|im_start|>\n assistant""".strip()

elif config['model_id'] == 'meta-llama/Llama-2-13b-chat-hf' :
  # Setup prompt template ####
  config['embedding_model'] = 'intfloat/e5-large-v2'
  config['model_kwargs'] = {}
  
  # Model parameters
  config['pipeline_kwargs']={"temperature":  0.10,
                            "max_new_tokens": 256}
  
  config['template'] = """<s>[INST] <<SYS>>
You are an assistant which helps extracts important entities.If the text is incorrect just say I cannot answer the questions.Do not add entities which are not explicitly mentioned in the text
    <</SYS>> text: {context} \n
    [/INST]\n entities""".strip()

elif config['model_id'] == 'meta-llama/Llama-2-70b-chat-hf' :
  # Setup prompt template ####
  config['embedding_model'] = 'intfloat/e5-large-v2'
  
  config['model_kwargs'] = {"load_in_8bit" : True}

  # Model parameters
  config['pipeline_kwargs']={"temperature":  0.10,
                            "max_new_tokens": 256}
  
  config['template'] = """<s>[INST] <<SYS>>
You are an assistant which helps extracts important entities.If the text is incorrect just say I cannot answer the questions.Do not add entities which are not explicitly mentioned in the text
    <</SYS>> text: {context} \n
    [/INST]\n entities""".strip()





# COMMAND ----------

# DBTITLE 1,Set evaluation config
# config["eval_dataset_path"]= "./data/eval_data.tsv"