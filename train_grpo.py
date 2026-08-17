#!/usr/bin/env python3
"""Stage 2: GRPO LoRA training from the merged SFT model using FSDP."""

from __future__ import annotations

import argparse
import faulthandler
import os
import signal
from pathlib import Path

from transformers import TrainerCallback
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


def _multiple_of(dataset, factor: int):
    usable = len(dataset) - (len(dataset) % factor)
    return dataset.select(range(usable))


class PeriodicAdapterCheckpoint(TrainerCallback):
    """Snapshot the LoRA adapter mid-run so a crash costs one interval, not the run.

    `save_fsdp_adapter` gathers DTensors collectively, so this must execute on
    every rank.
    """

    def __init__(self, every_steps: int, path, tokenizer):
        self.every_steps = every_steps
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.trainer = None

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_steps <= 0 or self.trainer is None:
            return control
        if state.global_step <= 0 or state.global_step % self.every_steps:
            return control
        target = self.path.parent / f"{self.path.name}_step{state.global_step}"
        try:
            save_fsdp_adapter(self.trainer, target, self.tokenizer)
            print(f"[checkpoint] step {state.global_step} -> {target}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[checkpoint] step {state.global_step} failed: {exc}", flush=True)
        return control


def main() -> None:
    cli = parse_args()
    faulthandler.register(signal.SIGUSR1)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("FSDP_CPU_RAM_EFFICIENT_LOADING", "true")
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("NCCL_IB_DISABLE", "0")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    stage = "grpo"
    log_distributed_env(stage)
    snapshot_nvidia_smi("before_model_load", stage)

    num_generations = int(os.getenv("GRPO_NUM_GENERATIONS", "4"))
    per_device = int(os.getenv("GRPO_PER_DEVICE_BATCH", "2"))
    grad_accum = int(os.getenv("GRPO_GRAD_ACCUM", "1"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    # Accumulation shrinks the logprob forward (and its vocab-sized logits) without
    # changing how many prompts an optimizer step consumes.
    global_batch = per_device * world_size * grad_accum
    # One generation round per optimizer step keeps prompts/step = global/num_generations.
    generation_batch_size = int(
        os.getenv("GRPO_GENERATION_BATCH", str(global_batch))
    )

    print(f"[rank {os.getenv('RANK', '0')}] loading tokenizer/model", flush=True)
    tokenizer = load_tokenizer(SFT_MERGED)
    tokenizer.model_max_length = MAX_LENGTH
    original_eos_token = tokenizer.eos_token
    # TRL both stops generation on, and detects termination with, a single
    # `tokenizer.eos_token_id`. Gemma chat models close a turn with
    # <end_of_turn>, never <eos>, so leaving the default makes every completion
    # run to max_completion_length and get masked as truncated.
    turn_end = os.getenv("GRPO_EOS_TOKEN", "<end_of_turn>")
    turn_end_id = tokenizer.convert_tokens_to_ids(turn_end)
    if turn_end_id is not None and turn_end_id != tokenizer.unk_token_id:
        tokenizer.eos_token = turn_end
        print(
            f"[rank {os.getenv('RANK', '0')}] eos override: {original_eos_token!r} "
            f"-> {turn_end!r} (id={tokenizer.eos_token_id})",
            flush=True,
        )
    else:
        print(
            f"[rank {os.getenv('RANK', '0')}] eos override skipped: {turn_end} not in vocab",
            flush=True,
        )

    # Also pass both stop ids so generate itself can halt even if something
    # rewrites tokenizer.eos_token_id later.
    eos_ids = []
    for x in (tokenizer.eos_token_id, 1, 106):
        if x is not None and x not in eos_ids:
            eos_ids.append(x)
    generation_kwargs = {"eos_token_id": eos_ids}

    args = GRPOConfig(
        output_dir=str(GRPO_ADAPTER),
        num_train_epochs=1,
        max_steps=cli.max_steps,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
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
        num_generations=num_generations,
        num_generations_eval=num_generations,
        generation_batch_size=generation_batch_size,
        max_completion_length=int(os.getenv("GRPO_MAX_COMPLETION", "700")),
        temperature=1.0,
        top_p=1.0,
        beta=0.04,
        use_vllm=False,
        mask_truncated_completions=True,
        generation_kwargs=generation_kwargs,
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

    model = load_model(SFT_MERGED)
    inspect_fsdp_model(model, stage, tag="after_base_load")
    snapshot_nvidia_smi("after_model_load", stage)

    dataset = grpo_dataset()
    if cli.limit:
        dataset = dataset.select(range(min(cli.limit, len(dataset))))
    split = dataset.train_test_split(test_size=0.15, seed=42)
    prompts_per_step = max(1, generation_batch_size // num_generations)
    train_dataset = _multiple_of(split["train"], prompts_per_step)
    eval_dataset = _multiple_of(split["test"], prompts_per_step)
    print(
        f"[rank {os.getenv('RANK', '0')}] grpo batching: per_device={per_device} "
        f"world={world_size} grad_accum={grad_accum} num_generations={num_generations} "
        f"generation_batch={generation_batch_size} prompts/step={prompts_per_step} "
        f"train_prompts={len(train_dataset)} est_steps={len(train_dataset)//prompts_per_step}",
        flush=True,
    )

    ckpt_every = int(os.getenv("GRPO_CHECKPOINT_EVERY", "20"))
    periodic = PeriodicAdapterCheckpoint(ckpt_every, GRPO_ADAPTER, tokenizer)

    print(f"[rank {os.getenv('RANK', '0')}] building GRPO trainer", flush=True)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=format_reward,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config(),
        callbacks=[FSDPProvenanceCallback(stage), periodic],
    )
    periodic.trainer = trainer
    inspect_fsdp_model(trainer.model, stage, tag="after_trainer_init")
    snapshot_nvidia_smi("after_trainer_init", stage)
    print(f"[rank {os.getenv('RANK', '0')}] entering trainer.train", flush=True)
    trainer.train(resume_from_checkpoint=cli.resume_from_checkpoint)
    save_fsdp_adapter(trainer, GRPO_ADAPTER, tokenizer)
    snapshot_nvidia_smi("after_adapter_save", stage)


if __name__ == "__main__":
    main()
