from typing import Any, Tuple, List, Union, Callable
from itertools import islice

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoConfig,
)

import pandas as pd


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    prompt: Union[List[str], str],
    temperature: float = 0.7,
    top_k: float = 0.92,
    max_new_tokens: int = 200,
) -> List[str]:
    if isinstance(prompt, str):
        prompt = [prompt]
    batch = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
    batch = batch.to("cuda")

    with torch.no_grad():
        output_tokens_batch = model.generate(
            use_cache=True,
            input_ids=batch.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_k,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_responses = []
    for output_tokens, curr_inpt_ids in zip(output_tokens_batch, batch):
        prompt_length = len(curr_inpt_ids)
        generated_response = tokenizer.decode(
            output_tokens[prompt_length:], skip_special_tokens=True
        )
        generated_response = generated_response.replace("Assistant:", "")
        generated_responses.append(generated_response)

    return generated_responses


def batchify(iterable, batch_size):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]



def generate_text_for_df(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    df: pd.DataFrame,
    src_col: str,
    target_col: str,
    gen_prompt_fn: Callable = None,
    post_process_fn: Callable = None,
    batch_size: int = 1,
):
    src_col_values = []
    responses_list = []
    for batch in batchify(df.to_dict(orient="records"), batch_size):
        prompts = []
        for rec in batch:
            src_col_values.append(rec[src_col])
            if gen_prompt_fn:
                prompt = gen_prompt_fn(rec[src_col])
            else:
                prompt = rec[src_col]
            prompts.append(prompt)
        responses = generate_text(model, tokenizer, prompts)

        for response in responses:
            if post_process_fn:
                response = post_process_fn(response)
            responses_list.append(response)
    print(responses_list)
    df[target_col] = responses_list
    return df


def get_prompt_bulletppoints(query: str) -> str:
    return f"""### Human: Summarize this paragraph into bullet points including all figures. Don't include information that was not presented in the document.\n\n{query}.\n###"""


def get_prompt_recreate_finding(query: str) -> str:
    return f"""### Human: Convert the following bullet points into coherent text.\n\n{query}.\n###"""
