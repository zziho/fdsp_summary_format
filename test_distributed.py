#!/usr/bin/env python3
"""Minimal NCCL connectivity check for this two-GPU host."""

import os

import torch
import torch.distributed as dist


dist.init_process_group("nccl")
rank = dist.get_rank()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
value = torch.tensor([rank + 1.0], device=device)
dist.all_reduce(value)
dist.barrier(device_ids=[device.index])
print(f"rank={rank} device={device} all_reduce={value.item()}", flush=True)
dist.destroy_process_group()
