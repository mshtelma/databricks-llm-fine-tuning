# Databricks notebook source
# MAGIC !wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb -O /tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
# MAGIC   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcublas-dev-11-7_11.10.1.25-1_amd64.deb -O /tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
# MAGIC   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb -O /tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
# MAGIC   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcurand-dev-11-7_10.2.10.91-1_amd64.deb -O /tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb && \
# MAGIC   dpkg -i /tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
# MAGIC   dpkg -i /tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
# MAGIC   dpkg -i /tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
# MAGIC   dpkg -i /tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb

# COMMAND ----------

# MAGIC %pip install torch==2.0.1

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %sh cd .. && pip install .

# COMMAND ----------

# MAGIC %sh cd .. && deepspeed --num_gpus=8 \
# MAGIC --module databricks_llm.fine_tune \
# MAGIC --final_model_output_path="/dbfs/data-mle/llm/msh/falcon_40b_lora_openassistant_guanac_v1" \
# MAGIC --output_dir="/local_disk0/output" \
# MAGIC --dataset="timdettmers/openassistant-guanaco" \
# MAGIC --model="tiiuae/falcon-7b" \
# MAGIC --tokenizer="tiiuae/falcon-7b" \
# MAGIC --use_lora=false \
# MAGIC --use_4bit=false \
# MAGIC --deepspeed_config="../ds_configs/ds_zero_3_cpu_offloading.json" \
# MAGIC --fp16=false \
# MAGIC --bf16=true \
# MAGIC --per_device_train_batch_size=1 \
# MAGIC --per_device_eval_batch_size=1 \
# MAGIC --gradient_checkpointing=true \
# MAGIC --gradient_accumulation_steps=1 \
# MAGIC --learning_rate=2e-6 \
# MAGIC --num_train_epochs=3 \
# MAGIC --weight_decay=1 \
# MAGIC --evaluation_strategy="steps" \
# MAGIC --save_strategy="steps" \
# MAGIC --save_steps=20

# COMMAND ----------
