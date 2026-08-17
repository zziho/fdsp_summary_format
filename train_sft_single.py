#!/usr/bin/env python3
"""Single-GPU BF16 LoRA SFT baseline matching the FSDP SFT hyperparameters."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, TrainerCallback
from trl import SFTConfig, SFTTrainer

from common import BASE_MODEL, MAX_LENGTH, load_tokenizer, lora_config, sft_dataset


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "single_outputs"
EVENTS = OUTPUT / "step_metrics.jsonl"


def append_event(row: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


class SingleGpuBenchmarkCallback(TrainerCallback):
    """Record synchronized step time and CUDA allocator peaks."""

    def __init__(self) -> None:
        self.step_started = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        append_event(
            {
                "event": "train_begin",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "total_vram_mib": torch.cuda.get_device_properties(0).total_memory
                / 1024**2,
            }
        )
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.step_started = time.perf_counter()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        append_event(
            {
                "event": "step",
                "step": state.global_step,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sec": time.perf_counter() - self.step_started,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            }
        )
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        append_event(
            {
                "event": "trainer_log",
                "step": state.global_step,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **(logs or {}),
            }
        )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        append_event(
            {
                "event": "train_end",
                "step": state.global_step,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--warmup-benchmark-steps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expected exactly one visible CUDA GPU; launch with CUDA_VISIBLE_DEVICES=0"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    EVENTS.unlink(missing_ok=True)
    append_event(
        {
            "event": "config",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_model": str(BASE_MODEL),
            "fsdp": False,
            "precision": "bf16",
            "lora_rank": 16,
            "max_length": MAX_LENGTH,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 32,
            "effective_batch_size": 32,
            "learning_rate": 2e-4,
            "optimizer": "adamw_torch_fused",
            "activation_checkpointing": True,
            "max_steps": cli.max_steps,
            "excluded_warmup_steps": cli.warmup_benchmark_steps,
        }
    )

    tokenizer = load_tokenizer(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    args = SFTConfig(
        output_dir=str(OUTPUT / "trainer"),
        num_train_epochs=3,
        max_steps=cli.max_steps,
        per_device_train_batch_size=1,
        # Match FSDP effective batch: 1 × 16 × world_size(2) = 32.
        gradient_accumulation_steps=32,
        learning_rate=2e-4,
        warmup_steps=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        # Single-GPU equivalent of the FSDP activation-checkpointing setting.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=MAX_LENGTH,
        completion_only_loss=True,
        packing=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=42,
        data_seed=42,
        dataloader_num_workers=2,
        fsdp=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=sft_dataset(),
        processing_class=tokenizer,
        peft_config=lora_config(),
        callbacks=[SingleGpuBenchmarkCallback()],
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    append_event(
        {
            "event": "model",
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": 100 * trainable / total,
        }
    )
    trainer.train()


if __name__ == "__main__":
    main()
