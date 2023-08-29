# Databricks notebook source
# MAGIC %sh 
# MAGIC RAY_PORT=9339
# MAGIC ulimit -n 1000000 && ray stop --force &&  ray start  --head --min-worker-port=20000 --max-worker-port=25000 --temp-dir="/local_disk0/tmp/ray/job"  --port=$RAY_PORT  --dashboard-port=8501 --dashboard-host="0.0.0.0" --include-dashboard=true --num-cpus=2 --system-config='{"object_spilling_config":"{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"/local_disk0/tmp/spill\"}}"}'

# COMMAND ----------

base_url='https://' + spark.conf.get("spark.databricks.workspaceUrl")
workspace_id=spark.conf.get("spark.databricks.clusterUsageTags.orgId")
cluster_id=spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
dashboard_port='8501'

pathname_prefix='/driver-proxy/o/' + workspace_id + '/' + cluster_id + '/' + dashboard_port+"/" 

apitoken = dbutils.notebook().entry_point.getDbutils().notebook().getContext().apiToken().get()
dashboard_url=base_url + pathname_prefix  # ?token=' + apitoken[0:10] + " " + apitoken[10:]
dashboard_url
displayHTML(f'<a href="{dashboard_url}">Click to go to ray Dashboard!</a>')

# COMMAND ----------

import ray
ray.init(address="localhost:9339")