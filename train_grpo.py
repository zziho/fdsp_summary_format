#!/usr/bin/env python3
"""Stage 2: GRPO LoRA training from the merged SFT model using FSDP."""

from __future__ import annotations

import argparse
import faulthandler
import os
import signal

from trl import GRPOConfig, GRPOTrainer

from common import (
    FSDPProvenanceCallback,
    GRPO_ADAPTER,
    MAX_LENGTH,
    OUTPUT_ROOT,
    SFT_MERGED,
    fsdp_config,
    grpo_dataset,
    inspect_fsdp_model,
    load_model,
    load_tokenizer,
    log_distributed_env,
    log_fsdp_config_provenance,
    lora_config,
    save_fsdp_adapter,
    snapshot_nvidia_smi,
)
from rewards import format_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def _multiple_of_four(dataset):
    usable = len(dataset) - (len(dataset) % 4)
    return dataset.select(range(usable))


def main() -> None:
    cli = parse_args()
    faulthandler.register(signal.SIGUSR1)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("FSDP_CPU_RAM_EFFICIENT_LOADING", "true")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    stage = "grpo"
    log_distributed_env(stage)
    snapshot_nvidia_smi("before_model_load", stage)

    args = GRPOConfig(
        output_dir=str(GRPO_ADAPTER),
        num_train_epochs=1,
        max_steps=cli.max_steps,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=2e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.02,
        warmup_steps=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        max_grad_norm=0.5,
        bf16=True,
        tf32=True,
        gradient_checkpointing=False,
        num_generations=4,
        num_generations_eval=4,
        generation_batch_size=4,
        max_completion_length=700,
        temperature=1.0,
        top_p=1.0,
        beta=0.04,
        use_vllm=False,
        mask_truncated_completions=True,
        logging_steps=1 if cli.max_steps > 0 else 10,
        log_completions=True,
        num_completions_to_print=2,
        eval_strategy="no" if cli.max_steps > 0 else "steps",
        eval_steps=100,
        save_strategy="no",
        report_to="none",
        seed=42,
        data_seed=42,
        dataloader_drop_last=True,
        dataloader_num_workers=2,
        fsdp=True,
        fsdp_config=fsdp_config(),
    )
    log_fsdp_config_provenance(stage, args)

    print(f"[rank {os.getenv('RANK', '0')}] loading tokenizer/model", flush=True)
    tokenizer = load_tokenizer(SFT_MERGED)
    tokenizer.model_max_length = MAX_LENGTH
    model = load_model(SFT_MERGED)
    inspect_fsdp_model(model, stage, tag="after_base_load")
    snapshot_nvidia_smi("after_model_load", stage)

    dataset = grpo_dataset()
    if cli.limit:
        dataset = dataset.select(range(min(cli.limit, len(dataset))))
    split = dataset.train_test_split(test_size=0.15, seed=42)
    train_dataset = _multiple_of_four(split["train"])
    eval_dataset = _multiple_of_four(split["test"])

    print(f"[rank {os.getenv('RANK', '0')}] building GRPO trainer", flush=True)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=format_reward,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config(),
        callbacks=[FSDPProvenanceCallback(stage)],
    )
    inspect_fsdp_model(trainer.model, stage, tag="after_trainer_init")
    snapshot_nvidia_smi("after_trainer_init", stage)
    print(f"[rank {os.getenv('RANK', '0')}] entering trainer.train", flush=True)
    trainer.train(resume_from_checkpoint=cli.resume_from_checkpoint)
    save_fsdp_adapter(trainer, GRPO_ADAPTER, tokenizer)
    snapshot_nvidia_smi("after_adapter_save", stage)


if __name__ == "__main__":
    main()
