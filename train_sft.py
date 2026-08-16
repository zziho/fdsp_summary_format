#!/usr/bin/env python3
"""Stage 1: supervised LoRA fine-tuning with two-GPU FSDP."""

from __future__ import annotations

import argparse
import faulthandler
import os
import signal

from trl import SFTConfig, SFTTrainer

from common import (
    BASE_MODEL,
    FSDPProvenanceCallback,
    MAX_LENGTH,
    OUTPUT_ROOT,
    SFT_ADAPTER,
    fsdp_config,
    inspect_fsdp_model,
    load_model,
    load_tokenizer,
    log_distributed_env,
    log_fsdp_config_provenance,
    lora_config,
    save_fsdp_adapter,
    sft_dataset,
    snapshot_nvidia_smi,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    faulthandler.register(signal.SIGUSR1)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("FSDP_CPU_RAM_EFFICIENT_LOADING", "true")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    stage = "sft"
    log_distributed_env(stage)
    snapshot_nvidia_smi("before_model_load", stage)

    args = SFTConfig(
        output_dir=str(SFT_ADAPTER),
        num_train_epochs=3,
        max_steps=cli.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        warmup_steps=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=False,
        max_length=MAX_LENGTH,
        completion_only_loss=True,
        packing=False,
        logging_steps=1 if cli.max_steps > 0 else 10,
        save_strategy="no",
        report_to="none",
        seed=42,
        data_seed=42,
        dataloader_num_workers=2,
        fsdp=True,
        fsdp_config=fsdp_config(),
    )
    log_fsdp_config_provenance(stage, args)

    print(f"[rank {os.getenv('RANK', '0')}] loading tokenizer/model", flush=True)
    tokenizer = load_tokenizer(BASE_MODEL)
    model = load_model(BASE_MODEL)
    inspect_fsdp_model(model, stage, tag="after_base_load")
    snapshot_nvidia_smi("after_model_load", stage)
    dataset = sft_dataset()
    if cli.limit:
        dataset = dataset.select(range(min(cli.limit, len(dataset))))

    print(f"[rank {os.getenv('RANK', '0')}] building SFT trainer", flush=True)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config(),
        callbacks=[FSDPProvenanceCallback(stage)],
    )
    inspect_fsdp_model(trainer.model, stage, tag="after_trainer_init")
    snapshot_nvidia_smi("after_trainer_init", stage)
    print(f"[rank {os.getenv('RANK', '0')}] entering trainer.train", flush=True)
    trainer.train(resume_from_checkpoint=cli.resume_from_checkpoint)
    save_fsdp_adapter(trainer, SFT_ADAPTER, tokenizer)
    snapshot_nvidia_smi("after_adapter_save", stage)


if __name__ == "__main__":
    main()
