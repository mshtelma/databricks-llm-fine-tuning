# Author: Daniel Liden (https://github.com/djliden)

import mlflow
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    GenerationConfig,
)
import torch
import json


class PyfuncTransformer(mlflow.pyfunc.PythonModel):
    """PyfuncTransformer is a class that extends the mlflow.pyfunc.PythonModel class
    and is used to create a custom MLflow model for text generation using Transformers.
    """

    def __init__(self, model_name, gen_config_dict=None, examples=""):
        """
        Initializes a new instance of the PyfuncTransformer class.

        Args:
            model_name (str): The name of the pre-trained Transformer model to use.
            gen_config_dict (dict): A dictionary of generation configuration parameters.
            examples: examples for multi-shot prompting, prepended to the input.
        """
        self.model_name = model_name
        self.gen_config_dict = (
            gen_config_dict if gen_config_dict is not None else {}
        )
        self.examples = examples
        super().__init__()

    def load_context(self, context):
        """
        Loads the model and tokenizer using the specified model_name.

        Args:
            context: The MLflow context.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto"
        )

        # Create a custom GenerationConfig
        gcfg = GenerationConfig.from_model_config(model.config)
        for key, value in self.gen_config_dict.items():
            if hasattr(gcfg, key):
                setattr(gcfg, key, value)

        # Apply the GenerationConfig to the model's config
        model.config.update(gcfg.to_dict())

        self.model = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
        )

    def predict(self, context, model_input):
        """
        Generates text based on the provided model_input using the loaded model.

        Args:
            context: The MLflow context.
            model_input: The input used for generating the text.

        Returns:
            list: A list of generated texts.
        """
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.values.flatten().tolist()
        elif not isinstance(model_input, list):
            model_input = [model_input]

        generated_text = []
        for input_text in model_input:
            output = self.model(
                self.examples + input_text, return_full_text=False
            )
            generated_text.append(
                output[0]["generated_text"],
            )

        return generated_text

class PyfuncTransformerWithParams(mlflow.pyfunc.PythonModel):
  """PyfuncTransformerWithParams is a class that extends the mlflow.pyfunc.PythonModel class
  and is used to create a custom MLflow model for text generation using Transformers.
  """

  def __init__(self, model_name):
    """
    Initializes a new instance of the PyfuncTransformer class.

    Args:
        model_name (str): The name of the pre-trained Transformer model to use.
        examples: examples for multi-shot prompting, prepended to the input.
    """
    self.model_name = model_name
    super().__init__()

  def load_context(self, context):
    """
    Loads the model and tokenizer using the specified model_name.

    Args:
        context: The MLflow context.
    """
    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    model = AutoModelForCausalLM.from_pretrained(
      self.model_name, device_map="auto"
    )

    self.model = pipeline(
      "text-generation",
      model=model,
      tokenizer=tokenizer,
      return_full_text=False,
    )

  def predict(self, context, model_input):
    """
    Generates text based on the provided model_input using the loaded model.

    Args:
        context: The MLflow context.
        model_input: The input used for generating the text.

    Returns:
        list: A list of generated texts.
    """

    if isinstance(model_input, pd.DataFrame):
      model_input = model_input.to_dict(orient="records")
    elif not isinstance(model_input, list):
      model_input = [model_input]

    generated_text = []
    has_config = ("config_dict" in model_input[0].keys())

    for record in model_input:
      input_text = record["input_text"]
      few_shot_examples = record["few_shot_examples"]
      
      # Update the GenerationConfig attributes with the provided config_dict
      gcfg = GenerationConfig.from_model_config(self.model.model.config)

      if has_config:
        config_dict = record["config_dict"]
        for key, value in json.loads(config_dict).items():
          if hasattr(gcfg, key):
            setattr(gcfg, key, value)

      output = self.model(
        few_shot_examples + input_text,
        generation_config=gcfg,
        return_full_text=False,
      )
      generated_text.append(output[0]["generated_text"])

    return generated_text