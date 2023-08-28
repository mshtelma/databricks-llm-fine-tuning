import json

from transformers import AutoModelForCausalLM

from databricks_llm.fine_tune import load_training_dataset, setup_hf_trainer
from databricks_llm.model_utils import get_tokenizer
from databricks_llm.utils import ExtendedTrainingArguments


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
