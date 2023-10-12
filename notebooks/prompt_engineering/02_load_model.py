# Databricks notebook source
# MAGIC %md 
# MAGIC ### This is the notebook to run the Open LLM's as an Web Service
# MAGIC

# COMMAND ----------

# MAGIC %run "../../databricks_llm/prompt_utils/notebook-config"

# COMMAND ----------

# MAGIC %md
# MAGIC **This notebook is not required if you want to run the OpenAI model**

# COMMAND ----------

import os 
if config['model_id'] == "openai":
  raise "Notebook note required , Use this notebook to run on when using open LLM. change the config"
else:
  os.environ['user'] = username

# COMMAND ----------

# MAGIC %pip install torch==2.0.1

# COMMAND ----------

# ! rm -rf /dbfs/$user/tgi/*

# COMMAND ----------

# MAGIC %sh
# MAGIC # install rust
# MAGIC curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > tmp.sh
# MAGIC sh tmp.sh -y
# MAGIC source "$HOME/.cargo/env"
# MAGIC
# MAGIC # install protoc
# MAGIC PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
# MAGIC curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
# MAGIC sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
# MAGIC sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
# MAGIC rm -f $PROTOC_ZIP
# MAGIC
# MAGIC # install text-generation-inference
# MAGIC rm -rf  /local_disk0/tmp/text-generation-inference
# MAGIC cd /local_disk0/tmp && git clone https://github.com/huggingface/text-generation-inference.git  
# MAGIC cd /local_disk0/tmp/text-generation-inference && make install

# COMMAND ----------

# MAGIC %sh 
# MAGIC FILE=/dbfs/$user/tgi/flash_attn-1.0.8-cp310-cp310-linux_x86_64.whl 
# MAGIC if test -f "$FILE"; then
# MAGIC     echo "$FILE exists."
# MAGIC else
# MAGIC     export flash_att_commit='3a9bfd076f98746c73362328958dbc68d145fbec'
# MAGIC     mkdir /dbfs/$user/tgi/  -p
# MAGIC     rm -rf  /local_disk0/tmp/flash-attention
# MAGIC     cd /local_disk0/tmp && git clone https://github.com/HazyResearch/flash-attention.git 
# MAGIC
# MAGIC     cd flash-attention && git fetch && git checkout ${flash_att_commit}
# MAGIC     python setup.py build
# MAGIC     python setup.py bdist_wheel
# MAGIC     cp  dist/flash_attn-1.0.8-cp310-cp310-linux_x86_64.whl /dbfs/$user/tgi/flash_attn-1.0.8-cp310-cp310-linux_x86_64.whl
# MAGIC     cd csrc/rotary && python setup.py build 
# MAGIC     python setup.py bdist_wheel
# MAGIC     cp  dist/rotary_emb-0.1-cp310-cp310-linux_x86_64.whl /dbfs/$user/tgi/rotary_emb-0.1-cp310-cp310-linux_x86_64.whl
# MAGIC     cd ..
# MAGIC     cd layer_norm && python setup.py build 
# MAGIC     python setup.py bdist_wheel
# MAGIC     cp  dist/dropout_layer_norm-0.1-cp310-cp310-linux_x86_64.whl /dbfs/$user/tgi/dropout_layer_norm-0.1-cp310-cp310-linux_x86_64.whl
# MAGIC fi

# COMMAND ----------

# MAGIC %sh 
# MAGIC FILE=/dbfs/$user/tgi/flash_attn-2.0.0.post1-cp310-cp310-linux_x86_64.whl
# MAGIC if test -f "$FILE"; then
# MAGIC     echo "$FILE exists."
# MAGIC else
# MAGIC     export flash_att_v2_commit='4f285b354796fb17df8636485b9a04df3ebbb7dc'
# MAGIC
# MAGIC     rm -rf  /local_disk0/tmp/flash-attention-v2
# MAGIC     cd /local_disk0/tmp && git clone https://github.com/HazyResearch/flash-attention.git flash-attention-v2
# MAGIC
# MAGIC     cd flash-attention-v2 && git fetch && git checkout ${flash_att_v2_commit}
# MAGIC     python setup.py build
# MAGIC     python setup.py bdist_wheel
# MAGIC     cp  dist/flash_attn-2.0.0.post1-cp310-cp310-linux_x86_64.whl /dbfs/$user/tgi/flash_attn-2.0.0.post1-cp310-cp310-linux_x86_64.whl
# MAGIC fi

# COMMAND ----------

# MAGIC %sh 
# MAGIC FILE=/dbfs/$user/tgi/vllm-0.0.0-cp310-cp310-linux_x86_64.whl
# MAGIC if test -f "$FILE"; then
# MAGIC     echo "$FILE exists."
# MAGIC else
# MAGIC     export vllm_commit='d284b831c17f42a8ea63369a06138325f73c4cf9'
# MAGIC
# MAGIC     rm -rf  /local_disk0/tmp/vllm
# MAGIC     cd /local_disk0/tmp && git clone https://github.com/OlivierDehaene/vllm.git
# MAGIC
# MAGIC     cd vllm && git fetch && git checkout ${vllm_commit}
# MAGIC     python setup.py build
# MAGIC     python setup.py bdist_wheel
# MAGIC     cp dist/vllm-0.0.0-cp310-cp310-linux_x86_64.whl /dbfs/$user/tgi/vllm-0.0.0-cp310-cp310-linux_x86_64.whl
# MAGIC fi

# COMMAND ----------

# MAGIC %pip install /dbfs/$user/tgi/flash_attn-2* /dbfs/$user/tgi/dropout_laye* /dbfs/$user/tgi/rotary_emb*  /dbfs/$user/tgi/vllm*  urllib3==1.25.4 protobuf==3.20.*

# COMMAND ----------

#  dbutils.library.restartPython() 

# COMMAND ----------

import os 
nodeid = spark.conf.get('spark.databricks.driverNodeTypeId')
if "A100" in nodeid:
  os.environ['sharded'] = 'false'
  os.environ['CUDA_VISIBLE_DEVICES'] = "0"
else:
  os.environ['sharded'] = 'true'
  os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

if "Llama-2" in config['model_id']: 
  os.environ['HUGGING_FACE_HUB_TOKEN'] = config['HUGGING_FACE_HUB_TOKEN']
os.environ['HUGGINGFACE_HUB_CACHE'] ='/local_disk0/tmp/'
os.environ['CUDA_MEMORY_FRACTION'] = "0.95"

# get model variables
os.environ['model_id'] = config['model_id']
if "load_in_8bit" in config['model_kwargs']:
  os.environ['quantize'] = "bitsandbytes"
if config['model_id'] != 'meta-llama/Llama-2-70b-chat-hf':
  os.environ['CUDA_MEMORY_FRACTION'] = ".9"


# COMMAND ----------

from dbruntime.databricks_repl_context import get_context
ctx = get_context()

port = "8880"
driver_proxy_api = f"https://{ctx.browserHostName}/driver-proxy-api/o/0/{ctx.clusterId}/{port}"

print(f"""
driver_proxy_api = '{driver_proxy_api}'
cluster_id = '{ctx.clusterId}'
port = {port}
""")

# COMMAND ----------

# MAGIC %sh
# MAGIC source "$HOME/.cargo/env"
# MAGIC
# MAGIC if [ -z ${quantize} ]; 
# MAGIC     then echo "quantize" && text-generation-launcher --model-id $model_id --port 8880 --trust-remote-code --sharded $sharded --max-input-length 2048 --max-total-tokens 4096 ;
# MAGIC else text-generation-launcher --model-id $model_id --port 8880 --trust-remote-code --sharded $sharded --max-input-length 2048 --max-total-tokens 4096 --quantize bitsandbytes  ;
# MAGIC fi
# MAGIC

# COMMAND ----------

! kill -9  $(ps aux | grep 'text-generation' | awk '{print $2}')

# COMMAND ----------

