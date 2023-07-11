from dataclasses import field, dataclass
import logging
from typing import Optional, Union


from transformers import (
    IntervalStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtendedTrainingArguments:
    local_rank: Optional[str] = field(default="-1")
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
    num_train_epochs: Optional[int] = field(default=None)
    max_steps: Optional[int] = field(default=-1)
    weight_decay: Optional[int] = field(default=1)
    logging_strategy: Optional[Union[str, IntervalStrategy]] = field(
        default=IntervalStrategy.STEPS
    )
    evaluation_strategy: Optional[Union[str, IntervalStrategy]] = field(
        default=IntervalStrategy.STEPS
    )
    save_strategy: Optional[Union[str, IntervalStrategy]] = field(
        default=IntervalStrategy.STEPS
    )
    fp16: Optional[bool] = field(default=False)
    bf16: Optional[bool] = field(default=True)
    save_steps: Optional[int] = field(default=100)
    logging_steps: Optional[int] = field(default=10)
