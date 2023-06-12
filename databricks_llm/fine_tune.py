import json
from dataclasses import field, dataclass
import logging
from typing import Optional

import os
import torch

from datasets import Dataset, load_dataset
import transformers
from peft import LoraConfig, get_peft_model
from ray.air import ScalingConfig, RunConfig, CheckpointConfig

# from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train.huggingface import HuggingFaceTrainer
from ray.train.huggingface import TransformersCheckpoint

import ray.data

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer, IntervalStrategy,
)


logger = logging.getLogger(__name__)


@dataclass
class ExtendedTrainingArguments:
    number_of_tasks: Optional[int] = field(default=2)
    dataset: Optional[str] = field(default=None)
    model: Optional[str] = field(default=None)
    tokenizer: Optional[str] = field(default=None)

    use_lora: Optional[bool] = field(default=True)
    use_4bit: Optional[bool] = field(default=False)

    final_model_output_path: Optional[str] = field(default="/local_disk0/final_model")

    deepspeed_config: Optional[str] = field(default=None)

    output_dir: Optional[str] = field(default=None)
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_checkpointing: Optional[bool] = field(default=True)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=1e-6)
    optim: Optional[str] = field(default="adamw_hf")
    num_train_epochs: Optional[int] = field(default=1)
    weight_decay: Optional[int] = field(default=1)
    logging_strategy: Optional[IntervalStrategy] = field(default=IntervalStrategy.STEPS)
    evaluation_strategy: Optional[str] = field(default=IntervalStrategy.STEPS)
    save_strategy: Optional[str] = field(default=IntervalStrategy.STEPS)
    fp16: Optional[bool] = field(default=False)
    bf16: Optional[bool] = field(default=True)
    save_steps: Optional[int] = field(default=100)
    logging_steps: Optional[int] = field(default=10)


def load_training_dataset(
    tokenizer,
    path_or_dataset: str,
    split: str,
    dataset_text_field: str,
    max_seq_len,
    formatting_func=None,
) -> Dataset:
    logger.info(f"Loading dataset from {path_or_dataset}")
    dataset = load_dataset(path_or_dataset, split=split)
    logger.info("Found %d rows", dataset.num_rows)

    use_formatting_func = formatting_func is not None and dataset_text_field is None

    # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
    def tokenize(element):
        outputs = tokenizer(
            element[dataset_text_field]
            if not use_formatting_func
            else formatting_func(element),
            truncation=True,
            max_length=max_seq_len,
            return_overflowing_tokens=False,
            return_length=True,
            return_tensors="np",
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == max_seq_len:
                input_batch.append(input_ids)

        return {"input_ids": input_batch}

    tokenized_dataset = dataset.map(
        tokenize, batched=True, remove_columns=dataset.column_names
    )

    return tokenized_dataset


def setup_model(args: ExtendedTrainingArguments) -> AutoModelForCausalLM:
    logger.info(f"Loading model: {args.model}")
    # config = AutoConfig.from_pretrained(
    #     args.model, trust_remote_code="true", torch_dtype=torch.bfloat16
    # )
    # config.attn_config['attn_impl'] = 'triton'

    if args.use_4bit:
        import bitsandbytes
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        # config=config,
        quantization_config=bnb_config,
        trust_remote_code="true",
        torch_dtype=torch.float16,
    )

    if args.use_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    if args.use_lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.config.use_cache = False

    return model


def get_tokenizer(
    pretrained_tokenizer_name_or_path: str,
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_tokenizer_name_or_path, trust_remote_code="true"
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def setup_hf_trainer(train_dataset, eval_dataset=None, **config) -> Trainer:
    args: ExtendedTrainingArguments = config["args"]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        optim=args.optim,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        logging_strategy=args.logging_strategy,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        fp16=args.fp16,
        bf16=args.bf16,
        deepspeed=config.get("deepspeed_config_dict", None),
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        push_to_hub=False,
        disable_tqdm=True,
        # group_by_length=True,
        # ddp_find_unused_parameters=False,
    )

    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = get_tokenizer(args.tokenizer)
    model = setup_model(args)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    return trainer


def train_ray(args: ExtendedTrainingArguments):
    tokenizer = get_tokenizer(args.tokenizer)
    train_dataset = load_training_dataset(
        tokenizer, args.dataset, "train", "text", 256, formatting_func=None
    )
    eval_dataset = load_training_dataset(
        tokenizer, args.dataset, "test", "text", 256, formatting_func=None
    )

    train_ray_dataset = ray.data.from_huggingface(train_dataset)
    eval_ray_dataset = ray.data.from_huggingface(eval_dataset)

    with open(args.deepspeed_config) as json_data:
        deepspeed_config_dict = json.load(json_data)

    trainer = HuggingFaceTrainer(
        trainer_init_per_worker=setup_hf_trainer,
        trainer_init_config={
            "args": args,
            "deepspeed_config_dict": deepspeed_config_dict,
        },
        scaling_config=ScalingConfig(
            num_workers=args.number_of_tasks,
            use_gpu=True,
            resources_per_worker={"GPU": 1, "CPU": 1},
        ),
        run_config=RunConfig(
            local_dir=f"/dbfs/data-mle/llm/msh/falcon_train/job/",
            # callbacks=[MLflowLoggerCallback(experiment_name=f"/Users/{username}/dolly_multi-gpu_setup",save_artifact=False)],
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="eval_loss",
                checkpoint_score_order="min",
            ),
            verbose=0,
        ),
        datasets={"train": train_ray_dataset, "evaluation": eval_ray_dataset},
        # preprocessor=preprocessor,
    )
    result = trainer.fit()

    checkpoint = TransformersCheckpoint.from_checkpoint(result.checkpoint)
    hf_trainer = checkpoint.get_model(
        model=AutoModelForCausalLM, trust_remote_code=True
    )
    hf_trainer.save_pretrained(args.final_model_output_path)

    return result


def train(args: ExtendedTrainingArguments):
    tokenizer = get_tokenizer(args.tokenizer)
    train_dataset = load_training_dataset(
        tokenizer, args.dataset, "train", "text", 256, formatting_func=None
    )
    eval_dataset = load_training_dataset(
        tokenizer, args.dataset, "test", "text", 256, formatting_func=None
    )
    trainer = setup_hf_trainer(train_dataset, eval_dataset, args=args)
    trainer.train()
    trainer.save_model(args.final_model_output_path)


def main():
    parser = HfArgumentParser(ExtendedTrainingArguments)

    parsed = parser.parse_args_into_dataclasses()
    args: ExtendedTrainingArguments = parsed[0]

    train(args)


if __name__ == "__main__":
    os.environ["HF_HOME"] = "/local_disk0/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        main()
    except Exception:
        logger.exception("main failed")
        raise
