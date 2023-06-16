# Databricks notebook source
# MAGIC !wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb -O /tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
# MAGIC   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcublas-dev-11-7_11.10.1.25-1_amd64.deb -O /tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
# MAGIC   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb -O /tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
# MAGIC   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcurand-dev-11-7_10.2.10.91-1_amd64.deb -O /tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb && \
# MAGIC   dpkg -i /tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
# MAGIC   dpkg -i /tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
# MAGIC  dpkg -i /tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
# MAGIC   dpkg -i /tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %pip install torch==2.0.1

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------
# MAGIC %sh cd .. && pip install .

# COMMAND ----------
import ray
import ray.util.scheduling_strategies
from ray.util.spark import setup_ray_cluster

# COMMAND ----------
MODEL = "tiiuae/falcon-40b"
NUMBER_OF_NODES = 2
NUMBER_OF_GPUS_PER_NODE = 8

# COMMAND ----------

setup_ray_cluster(
    num_worker_nodes=NUMBER_OF_NODES,
    num_cpus_per_node=NUMBER_OF_GPUS_PER_NODE * 2,
    num_gpus_per_node=NUMBER_OF_GPUS_PER_NODE,
    collect_log_to_path="/dbfs/data-mle/llm/msh/ray_collected_logs",
)
# COMMAND ----------

runtime_env = {"env_vars": {"RAY_memory_monitor_refresh_ms": "0"}}
ray.init(runtime_env=runtime_env)


# COMMAND ----------
from databricks_llm.ray_utils import install_libraries

install_libraries()

# COMMAND ----------
from databricks_llm.ray_utils import pre_download_model

pre_download_model(MODEL)

# COMMAND ----------

from databricks_llm.fine_tune import ExtendedTrainingArguments, train_ray

training_args = ExtendedTrainingArguments(
    number_of_tasks=NUMBER_OF_NODES * NUMBER_OF_GPUS_PER_NODE,
    dataset="timdettmers/openassistant-guanaco",
    model=MODEL,
    tokenizer=MODEL,
    use_lora=False,
    use_4bit=False,
    deepspeed_config="../ds_configs/ds_zero_3_cpu_offloading.json",
    final_model_output_path="/dbfs/data-mle/llm/msh/falcon_40b_openassistant_guanac_v1",
    output_dir="/local_disk0/output",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_checkpointing=True,
    gradient_accumulation_steps=1,
    learning_rate=2e-6,
    num_train_epochs=1,
    weight_decay=1,
    logging_strategy="steps",
    evaluation_strategy="steps",
    save_strategy="steps",
    fp16=False,
    bf16=True,
    save_steps=20,
    logging_steps=10,
)

result = train_ray(training_args)

# COMMAND ----------
