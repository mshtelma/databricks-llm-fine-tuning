# Databricks notebook source
!wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb -O /tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcublas-dev-11-7_11.10.1.25-1_amd64.deb -O /tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb -O /tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcurand-dev-11-7_10.2.10.91-1_amd64.deb -O /tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb && \
  dpkg -i /tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
  dpkg -i /tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
  dpkg -i /tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
  dpkg -i /tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb

# COMMAND ----------

# MAGIC %pip install torch==2.0.1

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

dbfs_output_location = "/dbfs/llm/falcon_7b_oas_guanac_v2"

# COMMAND ----------

import pathlib
from pyspark.ml.torch.distributor import TorchDistributor

curr_dir = pathlib.Path.cwd()

train_file = str(curr_dir / "fine_tune.py")
print(train_file)
deepspeed_config = str(
    (curr_dir / ".." / "ds_configs" / "ds_zero_3_cpu_offloading.json").resolve()
)
print(deepspeed_config)
args = [
    f"--final_model_output_path={dbfs_output_location}",
    "--output_dir=/local_disk0/output",
    "--dataset=timdettmers/openassistant-guanaco",
    "--model=tiiuae/falcon-7b",
    "--tokenizer=tiiuae/falcon-7b",
    "--use_lora=false",
    "--use_4bit=false",
    f"--deepspeed_config={deepspeed_config}",
    "--fp16=false",
    "--bf16=true",
    "--per_device_train_batch_size=4",
    "--per_device_eval_batch_size=4",
    "--gradient_checkpointing=true",
    "--gradient_accumulation_steps=1",
    "--learning_rate=2e-6",
    "--weight_decay=1 ",
    "--evaluation_strategy=steps",
    "--save_strategy=steps",
    "--save_steps=20",
    "--num_train_epochs=1",
]
distributor = TorchDistributor(num_processes=8, local_mode=False, use_gpu=True)
distributor.run(train_file, *args)

# COMMAND ----------

!ls -lah {dbfs_output_location}

# COMMAND ----------

import pandas as pd
import transformers
import mlflow
import torch

print(torch.__version__)

# COMMAND ----------

class FalconPyFuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "tiiuae/falcon-7b", padding_side="left", trust_remote_code=True
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts["repository"],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self.model.eval()

    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """
        INSTRUCTION_KEY = "Instruction:"
        RESPONSE_KEY = "Response:"

        return f"""{INSTRUCTION_KEY}
        {instruction}
        {RESPONSE_KEY}
        """

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        prompt = model_input["prompt"][0]
        temperature = model_input.get("temperature", [1.0])[0]
        max_tokens = model_input.get("max_tokens", [100])[0]

        # Build the prompt
        prompt = self._build_prompt(prompt)

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(
            encoded_input,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )

        # Decode the prediction to text
        # generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Removing the prompt from the generated text
        prompt_length = len(self.tokenizer.encode(prompt, return_tensors="pt")[0])
        generated_response = self.tokenizer.decode(
            output[0][prompt_length:], skip_special_tokens=True
        )

        return generated_response


# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define input and output schema
input_schema = Schema(
    [
        ColSpec(DataType.string, "prompt"),
        ColSpec(DataType.double, "temperature"),
        ColSpec(DataType.long, "max_tokens"),
    ]
)
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example = pd.DataFrame(
    {"prompt": ["what is ML?"], "temperature": [0.5], "max_tokens": [100]}
)

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=FalconPyFuncModel(),
        artifacts={"repository": dbfs_output_location},
        pip_requirements=[
            "torch==2.0.1",
            "transformers==4.28.1",
            "accelerate==0.18.0",
            "einops",
            "sentencepiece",
        ],
        input_example=input_example,
        signature=signature,
    )


# COMMAND ----------

# Load the logged model
loaded_model = mlflow.pyfunc.load_model("runs:/" + run.info.run_id + "/model")

# COMMAND ----------

# Make a prediction using the loaded model
input_example = pd.DataFrame(
    {"prompt": ["what is ML?"], "temperature": [0.5], "max_tokens": [100]}
)
loaded_model.predict(input_example)

# COMMAND ----------

# Register model in MLflow Model Registry
result = mlflow.register_model(
    "runs:/" + run.info.run_id + "/model", "falcon-7b-fine-tuned"
)
# Note: Due to the large size of the model, the registration process might take longer than the default maximum wait time of 300 seconds. MLflow could throw an exception indicating that the max wait time has been exceeded. Don't worry if this happens - it's not necessarily an error. Instead, you can confirm the registration status of the model by directly checking the model registry. This exception is merely a time-out notification and does not necessarily imply a failure in the registration process.
