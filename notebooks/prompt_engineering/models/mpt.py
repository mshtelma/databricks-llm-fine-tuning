import pandas as pd
import numpy as np
import transformers
import mlflow
import torch

class MPTInstruct7B(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    """
    This method initializes the tokenizer and language model
    using the specified model repository.
    """
    # Initialize tokenizer and language model
    name = "mosaicml/mpt-7b-instruct"
    config = transformers.AutoConfig.from_pretrained(
      name,
      trust_remote_code=True
    )
    config.attn_config['attn_impl'] = 'triton'
    config.init_device = 'cuda'
    config.max_seq_len = 4000

    model = transformers.AutoModelForCausalLM.from_pretrained(
      name,
      config=config,
      torch_dtype=torch.bfloat16,
      trust_remote_code=True,
      revision="bbe7a55d70215e16c00c1825805b81e4badb57d7"
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
      "EleutherAI/gpt-neox-20b",
      padding_side="left"
    )

    self.generator = transformers.pipeline(
      "text-generation",
      model=model, 
      config=config, 
      tokenizer=tokenizer,
      torch_dtype=torch.bfloat16,
      device=0
    )

  def _build_prompt(self, instruction):
    """
    This method generates the prompt for the model.
    """

    instruction_key = "### Instruction:"
    response_key = "### Response:"
    intro_blurb = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    prompt = """{intro}
    {instruction_key}
    {instruction}
    {response_key}
    """.format(
        intro=INTRO_BLURB,
        instruction_key=INSTRUCTION_KEY,
        instruction=instruction,
        response_key=RESPONSE_KEY
    )

    return prompt

  def predict(self, context, model_input):
    """
    This method generates prediction for the given input.
    """
    prompt = model_input["prompt"][0]
    temperature = model_input.get("temperature", [0.1])[0]
    top_p = model_input.get("top_p", [0.15])[0]
    top_k = model_input.get("top_k", [5])[0]
    do_sample = model_input.get("do_sample", [True])[0]

    # Build the prompt
    prompt = self._build_prompt(prompt)

    generated_response = self.generator(
      prompt,
      do_sample = do_sample,
      temperature = temperature,
      top_p = top_p,
      top_k = top_k
    )

    return generated_response