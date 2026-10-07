# SPDX-License-Identifier: MIT

"""Validate the released 6+2 layout before allocating training workers."""

import os
from typing import Any


def validate_skill_gpu_layout(config: Any) -> None:
    """Reject conflicting GPU, offload, and scheduling overrides."""
    actor = config.actor
    ref = config.ref
    rollout = config.rollout
    server = config.actor_server
    devices = str(server.cuda_visible_devices).replace(" ", "").split(",")
    train_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    checked_executor_devices = os.environ.get(
        "ACTOR_SERVER_CUDA_VISIBLE_DEVICES", ""
    ).split(",")
    checks = {
        "single node with two AReaL GPUs": config.cluster.n_nodes == 1
        and config.cluster.n_gpus_per_node == 2,
        "six distinct frozen executor GPUs": server.enabled
        and len(devices) == 6
        and len(set(devices)) == 6
        and all(device.isdigit() for device in devices),
        "executor GPUs match the validated launcher pool": set(devices)
        == set(checked_executor_devices),
        "disjoint two-GPU training pool": len(train_devices) == 2
        and len(set(train_devices)) == 2
        and all(device.isdigit() for device in train_devices)
        and not set(devices).intersection(train_devices),
        "executor DP6 TP1": server.data_parallel_size == 6
        and server.tensor_parallel_size == 1,
        "d2 backends": actor.backend == "fsdp:d2"
        and ref.backend == "fsdp:d2"
        and rollout.backend == "sglang:d2",
        "actor allocated independently": actor.scheduling_strategy.type == "separation",
        "reference directly colocated with actor": ref.scheduling_strategy.type
        == "colocation"
        and ref.scheduling_strategy.target == "actor"
        and ref.scheduling_strategy.fork,
        "rollout directly colocated with actor": rollout.scheduling_strategy.type
        == "colocation"
        and rollout.scheduling_strategy.target == "actor"
        and rollout.scheduling_strategy.fork,
        "offload and disk weight synchronization": config.enable_offload
        and actor.offload
        and ref.offload
        and actor.weight_update_mode == "disk",
        "colocated SGLang memory release enabled": config.sglang.enable_memory_saver,
        "strict on-policy rollout": rollout.max_head_offpolicyness == 0,
    }
    failed = [label for label, passed in checks.items() if not passed]
    if failed:
        raise ValueError("invalid 6+2 GPU layout: " + "; ".join(failed))
