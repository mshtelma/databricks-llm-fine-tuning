# Databricks notebook source
# MAGIC %md
# MAGIC # Tune a text classification model with Hugging Face Transformers
# MAGIC This notebook trains a SMS spam classifier with "distillibert-base-uncased" as the base model on a single GPU machine
# MAGIC using the [🤗&nbsp;Transformers](https://huggingface.co/docs/transformers/index) library.
# MAGIC
# MAGIC ## Cluster setup
# MAGIC For this notebook, Databricks recommends a single GPU cluster, such as a `g4dn.xlarge` on AWS or `Standard_NC4as_T4_v3` on Azure. You can [create a single machine cluster](https://docs.databricks.com/clusters/configure.html) using the personal compute policy or by choosing "Single Node" when creating a cluster. This notebook works with Databricks Runtime ML GPU version 11.1 or greater. Databricks Runtime ML GPU versions 9.1 through 10.4 can be used by replacing the following command with `%pip install --upgrade transformers datasets evaluate`.
# MAGIC
# MAGIC The `transformers` library is installed by default on Databricks Runtime ML. This notebook also requires [🤗&nbsp;Datasets](https://huggingface.co/docs/datasets/index) and [🤗&nbsp;Evaluate](https://huggingface.co/docs/evakyate/index), which you can install using `%pip`.

# COMMAND ----------

# MAGIC %md
# MAGIC Set up any parameters for the notebook. 
# MAGIC - The base model [DistilBERT base model (uncased)](https://huggingface.co/distilbert-base-uncased) is a great foundational model that is smaller and faster than [BERT base model (uncased)](https://huggingface.co/bert-base-uncased), but still provides similar behavior. This notebook fine tunes this base model.
# MAGIC
# MAGIC

# COMMAND ----------

import datasets
from transformers import AutoTokenizer

base_model = "distilbert-base-uncased" 

tokenizer = AutoTokenizer.from_pretrained(base_model)

ds = datasets.load_dataset("bitext/customer-support-intent-dataset")
ds = ds.rename_columns({
  "utterance": "text",
  "intent": "labels"
}).remove_columns(["category", "tags"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Data preparation
# MAGIC
# MAGIC The datasets passed into the transformers trainer for text classification need to have integer labels. 

# COMMAND ----------

# MAGIC %md
# MAGIC Collect the labels and generate a mapping from labels to IDs and vice versa. `transformers` models need
# MAGIC these mappings to correctly translate the integer values into the human readable labels.

# COMMAND ----------

# MAGIC %md
# MAGIC Tokenize and shuffle the datasets for training. Since the [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer) does not need the untokenized `text` columns for training,
# MAGIC the notebook removes them from the dataset. This isn't necessary, but not removing the column results in a warning during training.
# MAGIC In this step, `datasets` also caches the transformed datasets on local disk for fast subsequent loading during model training.

# COMMAND ----------

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=True)

tokenized_df = ds.map(tokenize_function, batched=True).remove_columns(["text"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Model training
# MAGIC For model training, this notebook largely uses default behavior. However, you can use the full range of 
# MAGIC metrics and parameters available to the `Trainer` to adjust your model training behavior.

# COMMAND ----------

# MAGIC %md
# MAGIC Create the evaluation metric to log. Loss is also logged, but adding other metrics such as accuracy can make modeling performance easier to understand.

# COMMAND ----------

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# COMMAND ----------

import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

import numpy as np


def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  predictions = np.argmax(predictions, axis=1)
  return accuracy.compute(predictions=predictions, references=labels)

# COMMAND ----------

train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()

labels = list(set(list(train_df.labels.unique()) + list(test_df.labels.unique())))

# COMMAND ----------

labels

# COMMAND ----------

id2label = {}
label2id = {}

for id_, label in enumerate(labels):
  id2label[id_] = label
  label2id[label] = id_

def map_label(example):
  example["labels"] = label2id[example["labels"]]
  return example

tokenized_df = tokenized_df.map(map_label)

# COMMAND ----------

tokenized_df["train"][0]

# COMMAND ----------

# MAGIC %md
# MAGIC Construct default training arguments. This is where you would set many of your training parameters, such as the learning rate.
# MAGIC Refer to [transformers documentation](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) 
# MAGIC for the full range of arguments you can set.

# COMMAND ----------

from transformers import TrainingArguments, Trainer
training_output_dir = "intent_trainer"
training_args = TrainingArguments(
  output_dir=training_output_dir, evaluation_strategy="epoch",
  learning_rate=2e-5,
  per_device_train_batch_size = 64,
  per_device_eval_batch_size = 64,
  num_train_epochs = 2,
  weight_decay = 0.01
)

# COMMAND ----------

# MAGIC %md
# MAGIC Create the model to train from the base model, specifying the label mappings and the number of classes.

# COMMAND ----------

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=len(id2label.keys()), label2id=label2id, id2label=id2label)

# COMMAND ----------

# MAGIC %md
# MAGIC Using a [data collator](https://huggingface.co/docs/transformers/main_classes/data_collator) batches input
# MAGIC in training and evaluation datasets. Using the `DataCollatorWithPadding` with defaults gives good baseline
# MAGIC performance for text classification.

# COMMAND ----------

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer)

# COMMAND ----------

# MAGIC %md
# MAGIC Construct the trainer object with the model, arguments, datasets, collator, and metrics created above.

# COMMAND ----------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_df["train"],
    eval_dataset=tokenized_df["test"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# COMMAND ----------

# MAGIC %md
# MAGIC Construct the [MLflow](https://mlflow.org) wrapper class to store the model as a pipeline. When loading the pipeline, this model uses the GPU if CUDA is available. This model hardwires the batchsize to use with the `transformers` pipeline. You'll want to set this with the hardware you will use
# MAGIC for inference in mind.

# COMMAND ----------

import mlflow
from tqdm.auto import tqdm
import torch

pipeline_artifact_name = "pipeline"
class TextClassificationPipelineModel(mlflow.pyfunc.PythonModel):
  
  def load_context(self, context):
    device = 0 if torch.cuda.is_available() else -1
    self.pipeline = pipeline("text-classification", context.artifacts[pipeline_artifact_name], device=device)
    
  def predict(self, context, model_input): 
    texts = model_input[model_input.columns[0]].to_list()
    pipe = tqdm(self.pipeline(texts, truncation=True, batch_size=8), total=len(texts), miniters=10)
    labels = [prediction['intent'] for prediction in pipe]
    return pd.Series(labels)

# COMMAND ----------

# MAGIC %md
# MAGIC Train the model, logging metrics and results to MLflow. This task is very easy for BERT-based models. Don't be
# MAGIC surprised is the evaluation accuracy is 1 or close to 1.

# COMMAND ----------

from transformers import pipeline

model_output_dir = "/tmp/intent_model"
pipeline_output_dir = "/tmp/intent_pipeline"
model_artifact_path = "intent_classification_model"

with mlflow.start_run() as run:
  trainer.train()
  trainer.save_model(model_output_dir)
  pipe = pipeline(
    "text-classification",
    model=AutoModelForSequenceClassification.from_pretrained(model_output_dir),
    tokenizer=tokenizer
  )
  pipe.save_pretrained(pipeline_output_dir)
  mlflow.transformers.log_model(
    transformers_model=pipe,
    artifact_path=model_artifact_path, 
    input_example="Hi there!",
  )

# COMMAND ----------

# DBTITLE 1,Inference
import mlflow
logged_model = 'runs:/f53cb5f163c04b3a91174fa01336cb03/intent_classification_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(ds["validation"].to_pandas())

# COMMAND ----------


