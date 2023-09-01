# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Basic Prompting & Evaluation
# MAGIC
# MAGIC <br/>
# MAGIC
# MAGIC * Now that we know a bit more possible parameters used in LLMs, let's start with some basic Prompt Engineering experiments.
# MAGIC * For this first part, we will use relatively lightweight LLMs - 1b parameters or less:
# MAGIC   * [gpt2-large](https://huggingface.co/gpt2-large) (774M parameters)
# MAGIC   * [DialoGPT-large](https://huggingface.co/microsoft/DialoGPT-large?text=Hey+my+name+is+Julien%21+How+are+you%3F) (762M parameters)
# MAGIC   * [bloom-560m](https://huggingface.co/bigscience/bloom-560m) (560M parameters)
# MAGIC * We will leverage [Hugging Face](https://huggingface.co) for obtaining model weights as well as the `transformers` library.
# MAGIC * For prompt tracking, evaluation and LLM comparisons, we will leverage some of the new MLflow LLM features, such as `mlflow.evaluate()`.
# MAGIC * For the sake of simplicity, here we're dealing with a specific case of prompt engineering, where we provide some **context** - examples of questions and outputs we expect. Concretely, this can be described as [Few Shot Learning](https://en.wikipedia.org/w/index.php?title=Few-shot_learning_(natural_language_processing).
# MAGIC * For more details on Few Shot Learning and Generative AI, you can browse some of [Databricks' training offerings around the topic](https://www.databricks.com/blog/now-available-new-generative-ai-learning-offerings).

# COMMAND ----------

# DBTITLE 1,Installing dependencies
# MAGIC %pip install transformers accelerate torch mlflow xformers -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Instantiating our models
# Wrapper class for Transformers models
from models.transformer import PyfuncTransformer

gcfg = {
    "max_length": 180,
    "max_new_tokens": 10,
    "do_sample": False,
}

example = (
    "Q: Are elephants larger than mice?\nA: Yes.\n\n"
    "Q: Are mice carnivorous?\nA: No, mice are typically omnivores.\n\n"
    "Q: What is the average lifespan of an elephant?\nA: The average lifespan of an elephant in the wild is about 60 to 70 years.\n\n"
    "Q: Is Mount Everest the highest mountain in the world?\nA: Yes.\n\n"
    "Q: Which city is known as the 'City of Love'?\nA: Paris is often referred to as the 'City of Love'.\n\n"
    "Q: What is the capital of Australia?\nA: The capital of Australia is Canberra.\n\n"
    "Q: Who wrote the novel '1984'?\nA: The novel '1984' was written by George Orwell.\n\n"
)

gpt2large = PyfuncTransformer(
    "gpt2-large",
    gen_config_dict=gcfg,
    examples=example,
)

dialogpt = PyfuncTransformer(
    "microsoft/DialoGPT-large",
    gen_config_dict=gcfg,
    examples=example,
)

bloom560 = PyfuncTransformer(
    "bigscience/bloom-560m",
    gen_config_dict=gcfg,
    examples=example,
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Logging Models into MLflow
# MAGIC
# MAGIC <br />
# MAGIC
# MAGIC * Model logging in MLflow is essentially version control for Machine Learning models: it guarantees reproducibility by recording model and environment details.
# MAGIC * Model logging also allows us to track (and compare) model versions, so we can refer fack to older models and see what effects changes to the models have.
# MAGIC * More details about MLflow Experiment Tracking and Model Registry can be found in MLflow documentation:
# MAGIC   * [Experiment Tracking](https://mlflow.org/docs/latest/tracking.html)
# MAGIC   * [Model Registry](https://mlflow.org/docs/latest/model-registry.html)

# COMMAND ----------

import mlflow
import time

mlflow.set_experiment(experiment_name="/Shared/compare_small_models")
run_ids = []
artifact_paths = []
model_names = ["gpt2large", "dialogpt", "bloom560"]

for model, name in zip([gpt2large, dialogpt, bloom560], model_names):
  with mlflow.start_run(run_name=f"log_model_{name}_{time.time_ns()}"):
    pyfunc_model = model
    artifact_path = f"models/{name}"
    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=pyfunc_model,
        input_example="Q: What color is the sky?\nA:",
    )
    run_ids.append(mlflow.active_run().info.run_id)
    artifact_paths.append(artifact_path)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC By now you should be able to see our MLflow experiment, along with the different runs we created for each model:
# MAGIC
# MAGIC <hr />
# MAGIC
# MAGIC <img src="https://github.com/rafaelvp-db/databricks-llm-workshop/blob/main/img/experiment.png?raw=true" />

# COMMAND ----------

# DBTITLE 1,Defining our evaluation prompts
import pandas as pd

eval_df = pd.DataFrame(
  {
    "question": [
        "Q: What color is the sky?\nA:",
        "Q: Are trees plants or animals?\nA:",
        "Q: What is 2+2?\nA:",
        "Q: Who is Darth Vader?\nA:",
        "Q: What is your favorite color?\nA:",
    ]
  }
)

eval_df

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Evaluation
# MAGIC
# MAGIC Let's loop through our models and evaluate the answers to our prompts. We will leverage the `runs` that were created in the previous cells - this is not absolutely required, however by doing this we ensure all elements appear in MLflow's UI in a clean format.

# COMMAND ----------

    run_ids


# COMMAND ----------

for i in range(3):
  with mlflow.start_run(
    run_id=run_ids[i]
  ):  # reopen the run with the stored run ID
      f"runs:/{run_ids[i]}/{artifact_paths[i]}"
    evaluation_results = mlflow.evaluate(
      model=f"runs:/{run_ids[i]}/{artifact_paths[i]}",
      model_type="text",
      data=eval_df,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC * You’ll notice one other change we made this time around: we saved the evaluation data as an [MLflow Dataset](https://mlflow.org/docs/latest/python_api/mlflow.data.html#mlflow.data.dataset.Dataset) with `eval_data = mlflow.data.from_pandas(eval_df, name=”evaluate_configurations”)` and then referred to this dataset in our `evaluate()` call, explicitly associating the dataset with the evaluation.
# MAGIC * We can retrieve the dataset information from the run in the future if needed, ensuring that we don’t lose track of the data used in the evaluation.
# MAGIC * If we browse through our experiment page once again, we can see that we have the generation results for each of the models that we evaluated:
# MAGIC
# MAGIC <hr />
# MAGIC
# MAGIC <img src="https://github.com/rafaelvp-db/databricks-llm-workshop/blob/main/img/evaluate1.png?raw=true" />

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Keeping track of generation configurations
# MAGIC
# MAGIC <hr />
# MAGIC
# MAGIC * Our `PyfuncTransformerWithParams` class supports passing **generation configuration** parameters at inference time.
# MAGIC * Logging these configurations in addition to our inputs and outputs can be quite useful, as we can compare how different configurations affect our generation outputs and quality.
# MAGIC * Let's do another sequence of runs in order to properly track these parameters, along with inputs and outputs for each of our models.
# MAGIC * This time, we'll focus on `bloom560` and use different configurations values - we'll mainly change `do_sample` to `True`, and we'll experiment with `top_k` sampling.

# COMMAND ----------

import json

config_dict1 = {
  "do_sample": True,
  "top_k": 10,
  "max_length": 180,
  "max_new_tokens": 10
}

config_dict2 = {
  "do_sample": False,
  "max_length": 180,
  "max_new_tokens": 10
}

few_shot_examples_1 = (
  "Q: Are elephants larger than mice?\nA: Yes.\n\n"
  "Q: Are mice carnivorous?\nA: No, mice are typically omnivores.\n\n"
  "Q: What is the average lifespan of an elephant?\nA: The average lifespan of an elephant in the wild is about 60 to 70 years.\n\n"
)

few_shot_examples_2 = (
  "Q: Is Mount Everest the highest mountain in the world?\nA: Yes.\n\n"
  "Q: Which city is known as the 'City of Love'?\nA: Paris is often referred to as the 'City of Love'.\n\n"
  "Q: What is the capital of Australia?\nA: The capital of Australia is Canberra.\n\n"
  "Q: Who wrote the novel '1984'?\nA: The novel '1984' was written by George Orwell.\n\n"
)

few_shot_examples = [few_shot_examples_1, few_shot_examples_2]
config_dicts = [config_dict1, config_dict2]

questions = [
    "Q: What color is the sky?\nA:",
    "Q: Are trees plants or animals?\nA:",
    "Q: What is 2+2?\nA:",
    "Q: Who is the Darth Vader?\nA:",
    "Q: What is your favorite color?\nA:",
]

data = {
    "input_text": questions * len(few_shot_examples),
    "few_shot_examples": [
        example for example in few_shot_examples for _ in range(len(questions))
    ],
    "config_dict": [
        json.dumps(config)
        for config in config_dicts
        for _ in range(len(questions))
    ],
}

eval_df = pd.DataFrame(data)

# COMMAND ----------

from models.transformer import PyfuncTransformerWithParams

bloom560_with_params = PyfuncTransformerWithParams("bigscience/bloom-560m")

mlflow.set_experiment(experiment_name="/Shared/compare_generation_params")
model_name = "bloom560"

with mlflow.start_run(run_name=f"log_model_{model_name}_{time.time_ns()}"):
  # Define an input example
  input_example = pd.DataFrame(
    {
        "input_text": "Q: What color is the sky?\nA:",
        "few_shot_examples": example,  # Assuming 'example' is defined and contains your few-shot prompts
        "config_dict": {},  # Assuming an empty dict for the generation parameters in this example
    }
  )

  perplexity = #...
  mlflow.log_metric("perplexity", perplexity)

  # Define the artifact_path
  artifact_path = f"models/{model_name}"

  # log the data
  eval_data = mlflow.data.from_pandas(eval_df, name="evaluate_configurations")

  # Log the model
  mod = mlflow.pyfunc.log_model(
    artifact_path=artifact_path,
    python_model=bloom560_with_params,
    input_example=input_example,
  )

  # Define the model_uri
  model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"

  # Evaluate the model
  mlflow.evaluate(model=model_uri, model_type="text", data=eval_data)

# COMMAND ----------

# DBTITLE 1,Visualizing Evaluation Results: Few Shot Examples
# MAGIC %md
# MAGIC
# MAGIC For our new experiment, we are now able to visualize not only **inputs** and **outputs** for each of our model runs, but also our **few shot examples**:
# MAGIC
# MAGIC <br />
# MAGIC
# MAGIC <img src="https://github.com/rafaelvp-db/databricks-llm-workshop/blob/main/img/evaluate2.png?raw=true" />

# COMMAND ----------

# DBTITLE 1,Visualizing Evaluation Results: Generation Configurations
# MAGIC %md
# MAGIC
# MAGIC On top of different **few shot examples** used, we also have access to the different **generation configuration parameters**:
# MAGIC   
# MAGIC <img src="https://github.com/rafaelvp-db/databricks-llm-workshop/blob/main/img/evaluate3.png?raw=true" />

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Reference
# MAGIC
# MAGIC <hr />
# MAGIC
# MAGIC * [Comparing LLMs with MLflow](https://medium.com/@dliden/comparing-llms-with-mlflow-1c69553718df)
# MAGIC * [Announcing MLflow 2.4: LLMOps Tools for Robust Model Evaluation](https://www.databricks.com/blog/announcing-mlflow-24-llmops-tools-robust-model-evaluation)
