import json
import logging

import os
import torch

from datasets import Dataset, load_dataset


from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)

from databricks_llm.model_utils import get_model_and_tokenizer, get_tokenizer
from databricks_llm.utils import ExtendedTrainingArguments

logger = logging.getLogger(__name__)


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
        input_batch = []
        attention_masks = []

        outputs = tokenizer(
            element[dataset_text_field]
            if not use_formatting_func
            else formatting_func(element),
            truncation=True,
            padding=True,
            max_length=max_seq_len,
            return_overflowing_tokens=False,
            return_length=True,
        )

        for length, input_ids, attention_mask in zip(
            outputs["length"], outputs["input_ids"], outputs["attention_mask"]
        ):
            if length == max_seq_len:
                input_batch.append(input_ids)
                attention_masks.append(attention_mask)

        return {"input_ids": input_batch, "attention_mask": attention_masks}

    tokenized_dataset = dataset.map(
        tokenize, batched=True, remove_columns=dataset.column_names
    )

    return tokenized_dataset


def setup_hf_trainer(train_dataset, eval_dataset=None, **config) -> Trainer:
    args: ExtendedTrainingArguments = config["args"]

    torch.backends.cuda.matmul.allow_tf32 = True

    training_args = TrainingArguments(
        local_rank=args.local_rank,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        optim=args.optim,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
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
        report_to=[],
        # group_by_length=True,
        ddp_find_unused_parameters=False,

        fsdp=["full_shard", "offload"]
    )

    model, tokenizer = get_model_and_tokenizer(args.model, args.use_4bit, args.use_lora)
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
    import ray.data
    from ray.air import ScalingConfig, RunConfig, CheckpointConfig
    from ray.train.huggingface import HuggingFaceTrainer
    from ray.train.huggingface import TransformersCheckpoint

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
    if args.deepspeed_config:
        with open(args.deepspeed_config) as json_data:
            deepspeed_config_dict = json.load(json_data)
    else:
        deepspeed_config_dict = None
    trainer = setup_hf_trainer(
        train_dataset,
        eval_dataset,
        args=args,
        deepspeed_config_dict=deepspeed_config_dict,
    )
    trainer.train()
    trainer.save_model(args.final_model_output_path)


def main():
    parser = HfArgumentParser(ExtendedTrainingArguments)

    parsed = parser.parse_args_into_dataclasses()
    args: ExtendedTrainingArguments = parsed[0]

    train(args)


if __name__ == "__main__":
    os.environ["HF_HOME"] = "/local_disk0/hf"
    os.environ["HF_DATASETS_CACHE"] = "/local_disk0/hf"
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
