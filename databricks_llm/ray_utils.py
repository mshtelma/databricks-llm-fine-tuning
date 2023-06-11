import os
import ray
from huggingface_hub import snapshot_download


def force_on_node(node_id: str, remote_func_or_actor_class):
    scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node_id, soft=False
    )
    options = {"scheduling_strategy": scheduling_strategy}
    return remote_func_or_actor_class.options(**options)


def run_on_every_node(remote_func_or_actor_class, **remote_kwargs):
    refs = []
    for node in ray.nodes():
        if node["Alive"] and node["Resources"].get("GPU", None):
            refs.append(
                force_on_node(node["NodeID"], remote_func_or_actor_class).remote(
                    **remote_kwargs
                )
            )
    return ray.get(refs)


@ray.remote(num_gpus=4)
def download_model(
    pretrained_model_name_or_path: str, local_dir: str = "/local_disk0/tmp/"
):
    snapshot_download(
        pretrained_model_name_or_path, local_dir=local_dir, resume_download=True
    )


def pre_download_model(pretrained_model_name_or_path: str):
    _ = run_on_every_node(
        download_model, pretrained_model_name_or_path=pretrained_model_name_or_path
    )


cmd = """
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb -O /tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcublas-dev-11-7_11.10.1.25-1_amd64.deb -O /tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb -O /tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcurand-dev-11-7_10.2.10.91-1_amd64.deb -O /tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb && \
dpkg -i /tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
dpkg -i /tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
dpkg -i /tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
dpkg -i /tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb
"""


@ray.remote(num_gpus=4)
def install_libraries_remote():
    os.system(cmd)


def install_libraries():
    _ = run_on_every_node(install_libraries_remote)
