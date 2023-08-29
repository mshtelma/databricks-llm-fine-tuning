# Databricks notebook source
# MAGIC %md
# MAGIC # 🤗 Pretraining and Finetuning with Hugging Face Models

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Cluster:** `g5.4xlarge` [64GB ; 1 GPU]

# COMMAND ----------

# MAGIC %md
# MAGIC Want to pretrain and finetune a Hugging Face model with Composer? No problem. Here, we'll walk through using Composer to pretrain and finetune a Hugging Face model.
# MAGIC
# MAGIC ### Recommended Background
# MAGIC
# MAGIC If you have already gone through our [tutorial][huggingface] on finetuning a pretrained Hugging Face model with Composer, many parts of this tutorial will be familiar to you, but it is not necessary to do that one first.
# MAGIC
# MAGIC This tutorial assumes you are familiar with transformer models for NLP and with Hugging Face.
# MAGIC
# MAGIC To better understand the Composer part, make sure you're comfortable with the material in our [Getting Started][getting_started] tutorial.
# MAGIC
# MAGIC ### Tutorial Goals and Concepts Covered
# MAGIC
# MAGIC The goal of this tutorial is to demonstrate how to pretrain and finetune a Hugging Face transformer using the Composer library!
# MAGIC
# MAGIC Inspired by [this paper][downstream] showing that performing unsupervised pretraining on the downstream dataset can be surprisingly effective, we will focus on pretraining and finetuning a small version of [Electra][electra] on the [AG News][agnews] dataset!
# MAGIC
# MAGIC Along the way, we will touch on:
# MAGIC
# MAGIC * Creating our Hugging Face model, tokenizer, and data loaders
# MAGIC * Wrapping the Hugging Face model as a `ComposerModel` for use with the Composer trainer
# MAGIC * Reloading the pretrained model with a new head for sequence classification
# MAGIC * Training with Composer
# MAGIC
# MAGIC Let's do this 🚀
# MAGIC
# MAGIC [huggingface]: https://docs.mosaicml.com/projects/composer/en/stable/examples/huggingface_models.html
# MAGIC [getting_started]: https://docs.mosaicml.com/projects/composer/en/stable/examples/getting_started.html
# MAGIC [downstream]: https://arxiv.org/abs/2209.14389
# MAGIC [agnews]: https://paperswithcode.com/sota/text-classification-on-ag-news
# MAGIC [electra]: https://arxiv.org/abs/2003.10555

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Composer
# MAGIC
# MAGIC To use Hugging Face with Composer, we'll need to install Composer *with the NLP dependencies*. If you haven't already, run:

# COMMAND ----------

# MAGIC %pip install 'mosaicml[nlp]' xformers

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Hugging Face Model
# MAGIC First, we import an Electra model and its associated tokenizer from the transformers library. We use Electra small in this notebook so that our model trains quickly.

# COMMAND ----------

import transformers
from composer.utils import reproducibility

# Create an Electra masked language modeling model using Hugging Face transformers
# Note: this is just loading the model architecture, and is using randomly initialized weights, so it is important to set
# the random seed here
reproducibility.seed_all(17)
config = transformers.AutoConfig.from_pretrained('google/electra-small-discriminator')
model = transformers.AutoModelForMaskedLM.from_config(config)
tokenizer = transformers.AutoTokenizer.from_pretrained('google/electra-small-discriminator')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating Dataloaders

# COMMAND ----------

# MAGIC %md
# MAGIC For the purpose of this tutorial, we are going to perform unsupervised pretraining (masked language modeling) on our downstream dataset, AG News. We are only going to train for one epoch here, but note that the [paper][downstream] that showed good performance from pretraining on the downstream dataset trained for much longer.
# MAGIC
# MAGIC [downstream]: https://arxiv.org/abs/2209.14389

# COMMAND ----------

import datasets
from torch.utils.data import DataLoader

# Load the AG News dataset from Hugging Face
agnews_dataset = datasets.load_dataset('ag_news')

# Split the dataset randomly into a train and eval set
split_dict = agnews_dataset['train'].train_test_split(test_size=0.2, shuffle=True, seed=17)
train_dataset = split_dict['train']
eval_dataset = split_dict['test']

text_column_name = 'text'

# Tokenize the datasets
def tokenize_function(examples):
    # Remove empty lines
    examples[text_column_name] = [
        line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
    ]
    return tokenizer(
        examples[text_column_name],
        padding='max_length',
        truncation=True,
        max_length=256,
        return_special_tokens_mask=True,
    )

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=[text_column_name, 'label'],
    load_from_cache_file=False,
)
tokenized_eval = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=[text_column_name, 'label'],
    load_from_cache_file=False,
)

# We use the language modeling data collator from Hugging Face which will handle preparing the inputs correctly
# for masked language modeling
collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Create the dataloaders
train_dataloader = DataLoader(tokenized_train, batch_size=64, collate_fn=collator)
eval_dataloader = DataLoader(tokenized_eval, batch_size=64, collate_fn=collator)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert model to `ComposerModel`
# MAGIC
# MAGIC Composer uses `HuggingFaceModel` as a convenient interface for wrapping a Hugging Face model (such as the one we created above) in a `ComposerModel`. Its parameters are:
# MAGIC
# MAGIC - `model`: The Hugging Face model to wrap.
# MAGIC - `tokenizer`: The Hugging Face tokenizer used to create the input data
# MAGIC - `metrics`: A list of torchmetrics to apply to the output of `eval_forward` (a `ComposerModel` method).
# MAGIC - `use_logits`: A boolean which, if True, flags that the model's output logits should be used to calculate validation metrics.
# MAGIC
# MAGIC See the [API Reference][api] for additional details.
# MAGIC
# MAGIC [api]: https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.models.HuggingFaceModel.html

# COMMAND ----------

from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.models.huggingface import HuggingFaceModel

metrics = [
    LanguageCrossEntropy(ignore_index=-100),
    MaskedAccuracy(ignore_index=-100)
]
# Package as a trainer-friendly Composer model
composer_model = HuggingFaceModel(model, tokenizer=tokenizer, metrics=metrics, use_logits=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimizers and Learning Rate Schedulers

# COMMAND ----------

# MAGIC %md
# MAGIC The last setup step is to create an optimizer and a learning rate scheduler. We will use Composer's [DecoupledAdamW][optimizer] optimizer and [LinearWithWarmupScheduler][scheduler].
# MAGIC
# MAGIC [optimizer]: https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.optim.DecoupledAdamW.html
# MAGIC [scheduler]: https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.optim.LinearWithWarmupScheduler.html

# COMMAND ----------

from composer.optim import DecoupledAdamW, LinearWithWarmupScheduler

optimizer = DecoupledAdamW(composer_model.parameters(), lr=1.0e-4, betas=[0.9, 0.98], eps=1.0e-06, weight_decay=1.0e-5)
lr_scheduler = LinearWithWarmupScheduler(t_warmup='250ba', alpha_f=0.02)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Composer Trainer

# COMMAND ----------

# MAGIC %md
# MAGIC We will now specify a Composer `Trainer` object and run our training! `Trainer` has many arguments that are described in our [documentation](https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.Trainer.html#trainer), so we'll discuss only the less-obvious arguments used below:
# MAGIC
# MAGIC - `max_duration` - a string specifying how long to train. This can be in terms of batches (e.g., `'10ba'` is 10 batches) or epochs (e.g., `'1ep'` is 1 epoch), [among other options][time].
# MAGIC - `save_folder` - a string specifying where to save checkpoints to
# MAGIC - `schedulers` - a (list of) PyTorch or Composer learning rate scheduler(s) that will be composed together.
# MAGIC - `device` - specifies if the training will be done on CPU or GPU by using `'cpu'` or `'gpu'`, respectively. You can omit this to automatically train on GPUs if they're available and fall back to the CPU if not.
# MAGIC - `train_subset_num_batches` - specifies the number of training batches to use for each epoch. This is not a necessary argument but is useful for quickly testing code.
# MAGIC - `precision` - whether to do the training in full precision (`'fp32'`) or mixed precision (`'amp_fp16'` or `'amp_bf16'`). Mixed precision can provide a ~2x training speedup on recent NVIDIA GPUs.
# MAGIC - `seed` - sets the random seed for the training run, so the results are reproducible!
# MAGIC
# MAGIC [time]: https://docs.mosaicml.com/projects/composer/en/stable/trainer/time.html

# COMMAND ----------

import torch
from composer import Trainer

# Create Trainer Object
trainer = Trainer(
    model=composer_model, # This is the model from the HuggingFaceModel wrapper class.
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration='1ep', # train for more epochs to get better performance
    save_folder='checkpoints/pretraining/',
    optimizers=optimizer,
    schedulers=[lr_scheduler],
    device='gpu' if torch.cuda.is_available() else 'cpu',
    # train_subset_num_batches=100, # uncomment this line to only run part of training, which will be faster
    precision='amp_bf16',  # mixed precision
    seed=17,
)
# Start training
trainer.fit()
trainer.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the pretrained model for finetuning

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have a pretrained Hugging Face model, we will load it in and finetune it on a sequence classification task. Composer provides utilities to easily reload a Hugging Face model and tokenizer from a composer checkpoint, and add a task specific head to the model so that it can be finetuned for a new task

# COMMAND ----------

from torchmetrics.classification import MulticlassAccuracy
from composer.metrics import CrossEntropy
from composer.models import HuggingFaceModel

# Note: this does not load the weights, just the right model/tokenizer class and config.
# The weights will be loaded by the Composer trainer
model, tokenizer = HuggingFaceModel.hf_from_composer_checkpoint(
    f'checkpoints/pretraining/latest-rank0.pt',
    model_instantiation_class='transformers.AutoModelForSequenceClassification',
    model_config_kwargs={'num_labels': 4})

metrics = [CrossEntropy(), MulticlassAccuracy(num_classes=4, average='micro')]
composer_model = HuggingFaceModel(model, tokenizer=tokenizer, metrics=metrics, use_logits=True)

# COMMAND ----------

# MAGIC %md
# MAGIC The next part should look very familiar if you have already gone through the [tutorial][huggingface], as it is exactly the same except using a different dataset and starting model!
# MAGIC
# MAGIC [huggingface]: https://docs.mosaicml.com/projects/composer/en/stable/examples/huggingface_models.html

# COMMAND ----------

# MAGIC %md
# MAGIC We will now finetune on the AG News dataset. We have already downloaded and split the dataset, so now we just need to prepare the dataset for finetuning.

# COMMAND ----------

import datasets

text_column_name = 'text'

def tokenize_function(sample):
    return tokenizer(
        text=sample[text_column_name],
        padding="max_length",
        max_length=256,
        truncation=True
    )

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text'],
    load_from_cache_file=False,
)
tokenized_eval = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text'],
    load_from_cache_file=False,
)

from torch.utils.data import DataLoader
data_collator = transformers.data.data_collator.default_data_collator
train_dataloader = DataLoader(tokenized_train, batch_size=32, shuffle=False, drop_last=False, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_eval, batch_size=32, shuffle=False, drop_last=False, collate_fn=data_collator)

# COMMAND ----------

# MAGIC %md
# MAGIC Next we will create our optimizer and learning rate scheduler for the finetuning task.

# COMMAND ----------

from composer.optim import DecoupledAdamW, LinearWithWarmupScheduler

optimizer = DecoupledAdamW(composer_model.parameters(), lr=1.0e-4, betas=[0.9, 0.98], eps=1.0e-06, weight_decay=3.0e-4)
lr_scheduler = LinearWithWarmupScheduler(t_warmup='0.06dur', alpha_f=0.02)

# COMMAND ----------

# MAGIC %md
# MAGIC Lastly we can make our finetuning trainer and train! The only new arguments to the trainer here are `load_path`, which tells Composer where to load the already trained weights from, and `load_weights_only`, which tells Composer that we only want to load the weights from the checkpoint, not any other state from the previous training run.

# COMMAND ----------

import torch
from composer import Trainer

# Create Trainer Object
trainer = Trainer(
    model=composer_model, # This is the model from the HuggingFaceModel wrapper class.
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration='1ep', # Again, training for more epochs is likely to lead to higher performance
    save_folder='checkpoints/finetuning/',
    load_path=f'checkpoints/pretraining/latest-rank0.pt',
    load_weights_only=True, # We're starting a new training run, so we just the model weights
    load_strict_model_weights=False, # We're going from the original model, which is for MaskedLM, to a new model, for SequenceClassification
    optimizers=optimizer,
    schedulers=[lr_scheduler],
    device='gpu' if torch.cuda.is_available() else 'cpu',
    precision='amp_bf16',
    seed=17,
)
# Start training
trainer.fit()
trainer.close()

# COMMAND ----------

# MAGIC %md
# MAGIC Not bad, we got up to 91.5% accuracy on our eval split! Note that this is considerably less than the state-of-the-art on this task, but we started from a randomly initialized model, and did not train for very long, either in pretraining or finetuning!

# COMMAND ----------

# MAGIC %md
# MAGIC There are many possibilities for how to improve performance. Using a larger model and training for longer are often the first thing to try to improve performance (given a fixed dataset). You can also tweak the hyperparameters, try a different model class, start from pretrained weights instead of randomly initialized, or try adding some of Composer's [algorithms][algorithms]. We encourage you to play around with these and other things to get familiar with training in Composer.
# MAGIC
# MAGIC [algorithms]: https://docs.mosaicml.com/projects/composer/en/stable/trainer/algorithms.html

# COMMAND ----------

# MAGIC %md
# MAGIC ## What next?
# MAGIC
# MAGIC You've now seen how to use the Composer `Trainer` to pretrain and finetune a Hugging Face model on the AG News dataset.
# MAGIC
# MAGIC If you want to keep learning more, try looking through some of the documents linked throughout this tutorial to see if you can form a deeper intuition for what's going on in these examples.
# MAGIC
# MAGIC In addition, please continue to explore our tutorials and examples! Here are a couple suggestions:
# MAGIC
# MAGIC * Explore more advanced applications of Composer like [applying image segmentation to medical images][image_segmentation_tutorial].
# MAGIC
# MAGIC * Learn about callbacks and how to apply [early stopping][early_stopping_tutorial].
# MAGIC
# MAGIC * Check out the [benchmarks][benchmarks] repo for full examples of training large language models like GPT and BERT, image segmentation models like DeepLab, and more!
# MAGIC
# MAGIC [benchmarks]: https://github.com/mosaicml/benchmarks
# MAGIC [image_segmentation_tutorial]: https://docs.mosaicml.com/projects/composer/en/stable/examples/medical_image_segmentation.html
# MAGIC [early_stopping_tutorial]: https://docs.mosaicml.com/projects/composer/en/stable/examples/early_stopping.html

# COMMAND ----------

# MAGIC %md
# MAGIC ## Come get involved with MosaicML!
# MAGIC
# MAGIC We'd love for you to get involved with the MosaicML community in any of these ways:
# MAGIC
# MAGIC ### [Star Composer on GitHub](https://github.com/mosaicml/composer)
# MAGIC
# MAGIC Help make others aware of our work by [starring Composer on GitHub](https://github.com/mosaicml/composer).
# MAGIC
# MAGIC ### [Join the MosaicML Slack](https://join.slack.com/t/mosaicml-community/shared_invite/zt-w0tiddn9-WGTlRpfjcO9J5jyrMub1dg)
# MAGIC
# MAGIC Head on over to the [MosaicML slack](https://join.slack.com/t/mosaicml-community/shared_invite/zt-w0tiddn9-WGTlRpfjcO9J5jyrMub1dg) to join other ML efficiency enthusiasts. Come for the paper discussions, stay for the memes!
# MAGIC
# MAGIC ### Contribute to Composer
# MAGIC
# MAGIC Is there a bug you noticed or a feature you'd like? File an [issue](https://github.com/mosaicml/composer/issues) or make a [pull request](https://github.com/mosaicml/composer/pulls)!
