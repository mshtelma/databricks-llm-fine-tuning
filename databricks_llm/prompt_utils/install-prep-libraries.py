# Databricks notebook source
# MAGIC %pip install -U transformers==4.29.2 sentence-transformers==2.2.2 langchain==0.0.190 pypdf==3.9.1 pycryptodome==3.18.0 accelerate==0.19.0 unstructured==0.7.1 unstructured[local-inference]==0.7.1 sacremoses==0.0.53 ninja==1.11.1  tiktoken==0.4.0 openai==0.27.6 faiss-cpu==1.7.4 typing-inspect==0.8.0 typing_extensions==4.5.0 InstructorEmbedding text-generation ray[default]==2.5.1

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir -p /local_disk0/git/ 
# MAGIC cd /local_disk0/git/ 
# MAGIC git clone https://github.com/Azure/azure-sdk-for-python.git
# MAGIC cd azure-sdk-for-python/sdk/formrecognizer/azure-ai-formrecognizer
# MAGIC pip install -e .

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import nltk
nltk.download('punkt')