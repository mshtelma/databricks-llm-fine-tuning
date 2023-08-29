# Databricks notebook source
# MAGIC %md
# MAGIC # Install VLLM Code

# COMMAND ----------

# MAGIC %pip install torch==2.0.1 safetensors==0.3.1 accelerate==0.20.3 ray[default]
# MAGIC # %pip install git+https://github.com/puneet-jain159/vllm.git

# COMMAND ----------

# MAGIC %run "../../databricks_llm/prompt_utils/notebook-config"

# COMMAND ----------

 os.environ['user'] = username

# COMMAND ----------

# DBTITLE 1,Option 2 : Source Compile VLLM and copy wheel file to dbfs if not exists
# MAGIC %sh 
# MAGIC FILE=/dbfs/$user/vllm/vllm-0.1.3-cp310-cp310-linux_x86_64.whl
# MAGIC if test -f "$FILE"; then
# MAGIC     echo "$FILE exists."
# MAGIC else
# MAGIC     pip uninstall vllm -y || true
# MAGIC     rm -rf /local_disk0/tmp/vllm/
# MAGIC     mkdir /local_disk0/tmp/vllm/ -p 
# MAGIC     cd /local_disk0/tmp/vllm/ && git clone https://github.com/puneet-jain159/vllm.git
# MAGIC     cd vllm && git fetch && git checkout
# MAGIC     python setup.py build
# MAGIC     python setup.py bdist_wheel 
# MAGIC     # python setup.py install
# MAGIC     mkdir /dbfs/$user/vllm/ -p
# MAGIC     cp  /local_disk0/tmp/vllm/vllm/dist/vllm-0.1.3-cp310-cp310-linux_x86_64.whl /dbfs/$user/vllm/vllm-0.1.3-cp310-cp310-linux_x86_64.whl
# MAGIC fi
# MAGIC
# MAGIC
# MAGIC # %sh
# MAGIC

# COMMAND ----------

# DBTITLE 1,Build from Wheel file 
# MAGIC %pip install /dbfs/$user/vllm/vllm-0.1.3-cp310-cp310-linux_x86_64.whl

# COMMAND ----------

# MAGIC %md
# MAGIC # Test and benchmark VLLM Code

# COMMAND ----------

# dbutils.library.restartPython()
import torch
import os
from vllm import LLM, SamplingParams
os.environ['HUGGING_FACE_HUB_TOKEN'] = config['HUGGING_FACE_HUB_TOKEN']
os.environ['HUGGINGFACE_HUB_CACHE'] ='/local_disk0/tmp/'

# COMMAND ----------

# MAGIC %run "./util/install_ray"

# COMMAND ----------

import torch
llm = None
torch.cuda.empty_cache()

# COMMAND ----------

llm = LLM(model=config['modelId'],
          dtype="half",
          gpu_memory_utilization = 0.95,
          trust_remote_code=True)

# COMMAND ----------

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8,
                                 top_p=0.95,
                                 max_tokens = 256)

# COMMAND ----------

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# COMMAND ----------


