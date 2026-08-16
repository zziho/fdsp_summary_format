#!/usr/bin/env python3
"""Merge a LoRA adapter into its full-precision base model."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    BASE_MODEL,
    GRPO_ADAPTER,
    GRPO_MERGED,
    SFT_ADAPTER,
    SFT_MERGED,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("sft", "grpo"))
    return parser.parse_args()


def normalize_fsdp_adapter_keys(key: str) -> str:
    """Map FSDP+activation-checkpoint LoRA names onto PEFT's expected keys."""
    key = key.replace("._checkpoint_wrapped_module", "")
    if ".lora_A.weight" in key and ".lora_A.default.weight" not in key:
        key = key.replace(".lora_A.weight", ".lora_A.default.weight")
    if ".lora_B.weight" in key and ".lora_B.default.weight" not in key:
        key = key.replace(".lora_B.weight", ".lora_B.default.weight")
    return key


def prepare_adapter_for_peft(adapter_path: Path) -> Path:
    """
    If the adapter was saved under FSDP/checkpoint wrappers, materialize a
    temporary PEFT-compatible copy. Leaves the original adapter untouched.
    """
    weights = adapter_path / "adapter_model.safetensors"
    sd = load_file(str(weights))
    needs_fix = any(
        "_checkpoint_wrapped_module" in k or k.endswith("lora_A.weight") or k.endswith("lora_B.weight")
        for k in sd
    )
    if not needs_fix:
        return adapter_path

    fixed = {}
    for k, v in sd.items():
        nk = normalize_fsdp_adapter_keys(k)
        if nk in fixed:
            raise RuntimeError(f"duplicate key after remap: {nk}")
        fixed[nk] = v.contiguous()

    tmp = Path(tempfile.mkdtemp(prefix="peft_adapter_"))
    for name in adapter_path.iterdir():
        if name.name == "adapter_model.safetensors":
            continue
        dest = tmp / name.name
        if name.is_dir():
            shutil.copytree(name, dest)
        else:
            shutil.copy2(name, dest)
    save_file(fixed, str(tmp / "adapter_model.safetensors"))
    print(
        f"Normalized FSDP adapter keys for PEFT load "
        f"({len(sd)} -> {len(fixed)} tensors) via {tmp}",
        flush=True,
    )
    return tmp


def main() -> None:
    stage = parse_args().stage
    if stage == "sft":
        base_path, adapter_path, output_path = BASE_MODEL, SFT_ADAPTER, SFT_MERGED
    else:
        base_path, adapter_path, output_path = SFT_MERGED, GRPO_ADAPTER, GRPO_MERGED

    adapter_path = Path(adapter_path)
    if not Path(adapter_path, "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter is missing: {adapter_path}")

    peft_adapter = prepare_adapter_for_peft(adapter_path)

    print(f"Loading base model: {base_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    print(f"Loading adapter: {peft_adapter}")
    model = PeftModel.from_pretrained(model, peft_adapter)
    print("Merging adapter weights")
    model = model.merge_and_unload(safe_merge=True)
    model.config.use_cache = True

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer_source = (
        adapter_path if Path(adapter_path, "tokenizer_config.json").exists() else base_path
    )
    AutoTokenizer.from_pretrained(tokenizer_source).save_pretrained(output_path)
    print(f"Merged model saved to {output_path}")

    if peft_adapter != adapter_path:
        shutil.rmtree(peft_adapter, ignore_errors=True)


if __name__ == "__main__":
    main()
