# Databricks notebook source
!pip install --upgrade transformers
!pip install --upgrade accelerate
!pip install --upgrade mlflow
dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizerFast
import mlflow
import torch

# COMMAND ----------

import huggingface_hub
#skip this if you are already logged in to hugging face
huggingface_hub.login()

# COMMAND ----------

model = "meta-llama/Llama-2-7b-chat-hf"
repository = huggingface_hub.snapshot_download(repo_id=model, ignore_patterns="*.safetensors*")

# COMMAND ----------

# Define prompt template to get the expected features and performance for the chat versions. See our reference code in github for details: https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

# Define PythonModel to log with mlflow.pyfunc.log_model

class Llama2(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            context.artifacts['repository'], padding_side="left")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts['repository'], 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, 
            trust_remote_code=True,
            device_map="auto",
            pad_token_id=self.tokenizer.eos_token_id)
        self.model.eval()

    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """
        return f"""<s>[INST]<<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n\n{instruction}[/INST]\n"""

    def _generate_response(self, prompt, temperature, max_new_tokens):
        """
        This method generates prediction for a single input.
        """
        # Build the prompt
        prompt = self._build_prompt(prompt)

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(encoded_input, do_sample=True, temperature=temperature, max_new_tokens=max_new_tokens)
    
        # Decode the prediction to text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Removing the prompt from the generated text
        prompt_length = len(self.tokenizer.encode(prompt, return_tensors='pt')[0])
        generated_response = self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

        return generated_response
      
    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """

        outputs = []

        for i in range(len(model_input)):
          prompt = model_input["prompt"][i]
          temperature = model_input.get("temperature", [1.0])[i]
          max_new_tokens = model_input.get("max_new_tokens", [100])[i]

          outputs.append(self._generate_response(prompt, temperature, max_new_tokens))
      
        return outputs

# COMMAND ----------

with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=Llama2(),
        artifacts={'repository' : repository},
        pip_requirements=["torch", "transformers", "accelerate"],
        input_example=pd.DataFrame({"prompt":["what is ML?"],"max_tokens": [80]}),
        registered_model_name='llama2-7b-rvp-2',
        await_registration_for=0
    )

# COMMAND ----------

# Provide a name to the serving endpoint
endpoint_name = 'llama2-7b-example-rvp2'

# COMMAND ----------

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

import requests
import json

deploy_headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
deploy_url = f'{databricks_url}/api/2.0/serving-endpoints'

model_version = 1  # replace with latest model version
model_name = 'llama2-7b-rvp-2'

endpoint_config = {
  "name": endpoint_name,
  "config": {
    "served_models": [{
      "name": f'{model_name}_{model_version}',
      "model_name": model_name,
      "model_version": model_version,
      "workload_type": "GPU_MEDIUM",
      "workload_size": "Small",
      "scale_to_zero_enabled": "False"
    }]
  }
}
endpoint_json = json.dumps(endpoint_config, indent='  ')

# Send a POST request to the API
deploy_response = requests.request(method='POST', headers=deploy_headers, url=deploy_url, data=endpoint_json)

if deploy_response.status_code != 200:
  raise Exception(f'Request failed with status {deploy_response.status_code}, {deploy_response.text}')

# Show the response of the POST request
# When first creating the serving endpoint, it should show that the state 'ready' is 'NOT_READY'
# You can check the status on the Databricks model serving endpoint page, it is expected to take ~35 min for the serving endpoint to become ready
print(deploy_response.json())
