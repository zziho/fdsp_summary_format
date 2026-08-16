"""Shared configuration and helpers for the two-stage FSDP pipeline."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT.parent / "summary_format"

BASE_MODEL = Path(os.getenv("BASE_MODEL", ROOT.parent / "models/gemma-3-12b-it"))
DATA_PATH = Path(
    os.getenv(
        "DATA_PATH",
        SOURCE_ROOT / "train_data/model_input_fixed.jsonl",
    )
)
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", ROOT / "outputs"))
PROVENANCE_DIR = OUTPUT_ROOT / "provenance"

SFT_ADAPTER = OUTPUT_ROOT / "sft_adapter"
SFT_MERGED = OUTPUT_ROOT / "sft_merged"
GRPO_ADAPTER = OUTPUT_ROOT / "grpo_adapter"
GRPO_MERGED = OUTPUT_ROOT / "grpo_merged"

MAX_LENGTH = int(os.getenv("MAX_LENGTH", "4096"))
MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "2048"))
LORA_RANK = int(os.getenv("LORA_RANK", "16"))


def fsdp_config() -> dict:
    """FSDP settings required for mixed frozen/trainable PEFT parameters."""
    return {
        "version": 2,
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "transformer_layer_cls_to_wrap": ["Gemma3DecoderLayer"],
        # FSDP2: True == reshard after forward == FULL_SHARD behavior.
        "reshard_after_forward": True,
        "activation_checkpointing": True,
        "cpu_ram_efficient_loading": True,
        "state_dict_type": "FULL_STATE_DICT",
    }


def _rank() -> int:
    return int(os.getenv("RANK", "0"))


def _local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", os.getenv("RANK", "0")))


def _world_size() -> int:
    return int(os.getenv("WORLD_SIZE", "1"))


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def snapshot_nvidia_smi(tag: str, stage: str) -> list[dict]:
    """Capture per-GPU memory/util and persist CSV + JSONL provenance."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    query = (
        "index,uuid,name,utilization.gpu,utilization.memory,"
        "memory.used,memory.total,memory.free"
    )
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        row = {
            "event": "nvidia_smi_failed",
            "stage": stage,
            "tag": tag,
            "rank": _rank(),
            "error": str(exc),
        }
        if is_main_process():
            _append_jsonl(PROVENANCE_DIR / f"{stage}_events.jsonl", row)
            print(f"[provenance] nvidia-smi failed: {exc}", flush=True)
        return []

    rows = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "utilization_gpu": float(parts[3]),
                "utilization_memory": float(parts[4]),
                "memory_used_mib": float(parts[5]),
                "memory_total_mib": float(parts[6]),
                "memory_free_mib": float(parts[7]),
            }
        )

    if is_main_process():
        csv_path = PROVENANCE_DIR / f"{stage}_nvidia_smi_{tag}_{stamp}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(
            "index,uuid,name,utilization.gpu,utilization.memory,"
            "memory.used.MiB,memory.total.MiB,memory.free.MiB\n"
            + raw,
            encoding="utf-8",
        )
        _append_jsonl(
            PROVENANCE_DIR / f"{stage}_events.jsonl",
            {
                "event": "nvidia_smi",
                "stage": stage,
                "tag": tag,
                "timestamp": stamp,
                "csv_path": str(csv_path),
                "gpus": rows,
            },
        )
        print(f"[provenance] nvidia-smi[{tag}] -> {csv_path}", flush=True)
        for gpu in rows:
            print(
                f"[provenance] GPU{gpu['index']} "
                f"{gpu['memory_used_mib']:.0f}/{gpu['memory_total_mib']:.0f} MiB "
                f"util={gpu['utilization_gpu']:.0f}%",
                flush=True,
            )
    return rows


def log_distributed_env(stage: str) -> dict:
    """Log world_size / rank / device placement for every process."""
    info = {
        "event": "distributed_env",
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "world_size": _world_size(),
        "rank": _rank(),
        "local_rank": _local_rank(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "torch_distributed_is_initialized": torch.distributed.is_initialized(),
        "nccl_p2p_disable": os.getenv("NCCL_P2P_DISABLE"),
        "nccl_ib_disable": os.getenv("NCCL_IB_DISABLE"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        device_index = _local_rank() if torch.cuda.device_count() > 1 else 0
        device_index = min(device_index, torch.cuda.device_count() - 1)
        props = torch.cuda.get_device_properties(device_index)
        info.update(
            {
                "device": f"cuda:{device_index}",
                "device_name": props.name,
                "device_total_memory_mib": round(props.total_memory / 1024**2, 1),
            }
        )
    _append_jsonl(PROVENANCE_DIR / f"{stage}_rank{_rank()}_env.jsonl", info)
    print(
        f"[provenance] world_size={info['world_size']} "
        f"rank={info['rank']} local_rank={info['local_rank']} "
        f"device={info.get('device')}",
        flush=True,
    )
    return info


def log_fsdp_config_provenance(stage: str, training_args) -> dict:
    """Persist intended FSDP strategy (FULL_SHARD semantics under FSDP2)."""
    cfg = fsdp_config()
    info = {
        "event": "fsdp_config",
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rank": _rank(),
        "world_size": _world_size(),
        "fsdp_enabled": bool(getattr(training_args, "fsdp", False)),
        "fsdp_version": cfg.get("version"),
        "sharding_strategy": (
            "FULL_SHARD"
            if cfg.get("reshard_after_forward") in (True, "full_shard", "FULL_SHARD")
            else str(cfg.get("reshard_after_forward"))
        ),
        "reshard_after_forward": cfg.get("reshard_after_forward"),
        "auto_wrap_policy": cfg.get("auto_wrap_policy"),
        "transformer_layer_cls_to_wrap": cfg.get("transformer_layer_cls_to_wrap"),
        "activation_checkpointing": cfg.get("activation_checkpointing"),
        "state_dict_type": cfg.get("state_dict_type"),
        "cpu_ram_efficient_loading": cfg.get("cpu_ram_efficient_loading"),
        "fsdp_config": cfg,
        "fsdp_plugin_args": getattr(training_args, "fsdp_plugin_args", None),
    }
    _append_jsonl(PROVENANCE_DIR / f"{stage}_rank{_rank()}_fsdp_config.jsonl", info)
    print(
        f"[provenance] FSDP strategy={info['sharding_strategy']} "
        f"version={info['fsdp_version']} "
        f"wrap={info['transformer_layer_cls_to_wrap']}",
        flush=True,
    )
    return info


def inspect_fsdp_model(model, stage: str, tag: str = "post_init") -> dict:
    """Inspect wrapped modules / trainable params after Trainer FSDP setup."""
    wrap_cls = tuple(fsdp_config().get("transformer_layer_cls_to_wrap") or [])
    wrapped = []
    total = trainable = 0
    fsdp_module_count = 0
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in wrap_cls:
            wrapped.append(name)
        if "FSDP" in cls_name or hasattr(module, "_fsdp_param_group"):
            fsdp_module_count += 1
    for parameter in model.parameters():
        n = parameter.numel()
        total += n
        if parameter.requires_grad:
            trainable += n
    info = {
        "event": "fsdp_model_inspect",
        "stage": stage,
        "tag": tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rank": _rank(),
        "model_class": model.__class__.__name__,
        "wrap_cls": list(wrap_cls),
        "num_wrapped_layers": len(wrapped),
        "wrapped_layer_examples": wrapped[:8],
        "fsdp_marked_modules": fsdp_module_count,
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": round(100.0 * trainable / total, 4) if total else 0.0,
    }
    _append_jsonl(PROVENANCE_DIR / f"{stage}_rank{_rank()}_model_inspect.jsonl", info)
    print(
        f"[provenance][{tag}] model={info['model_class']} "
        f"wrapped_layers={info['num_wrapped_layers']} "
        f"trainable={info['trainable_params']}/{info['total_params']} "
        f"({info['trainable_pct']}%)",
        flush=True,
    )
    return info


class FSDPProvenanceCallback(TrainerCallback):
    """Emit FSDP/GPU provenance at train begin, mid logging, and train end."""

    def __init__(self, stage: str):
        self.stage = stage
        self._logged_begin = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self._logged_begin:
            return control
        self._logged_begin = True
        print(
            f"[provenance] on_train_begin rank={_rank()} "
            f"world_size={_world_size()} global_step={state.global_step}",
            flush=True,
        )
        if model is not None:
            inspect_fsdp_model(model, self.stage, tag="on_train_begin")
        snapshot_nvidia_smi("on_train_begin", self.stage)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Capture a mid-run GPU snapshot around the first logged step and every 50 steps.
        if state.global_step in {1, 10} or (
            state.global_step > 0 and state.global_step % 50 == 0
        ):
            snapshot_nvidia_smi(f"step_{state.global_step}", self.stage)
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        print(
            f"[provenance] on_train_end rank={_rank()} "
            f"global_step={state.global_step}",
            flush=True,
        )
        if model is not None:
            inspect_fsdp_model(model, self.stage, tag="on_train_end")
        snapshot_nvidia_smi("on_train_end", self.stage)
        return control


def lora_config() -> LoraConfig:
    """Apply LoRA only to the language tower, never the vision tower."""
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        lora_dropout=0.0,
        bias="none",
        target_modules=(
            r".*language_model.*\."
            r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        ),
    )


def load_tokenizer(model_path: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return tokenizer


def load_model(model_path: str | Path):
    """Load BF16 weights without device_map; FSDP owns device placement."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="bfloat16",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    # Gemma ties token embeddings and lm_head across different FSDP2 groups.
    # FSDP2 rejects such cross-group aliases, so keep identical values but
    # give the output head independent storage (as the original merge did).
    input_weight = model.get_input_embeddings().weight
    output_layer = model.get_output_embeddings()
    if output_layer is not None and output_layer.weight is input_weight:
        output_layer.weight = torch.nn.Parameter(input_weight.detach().clone())
        model.config.tie_word_embeddings = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.tie_word_embeddings = False
    model.config.use_cache = False
    return model


def load_rows(path: str | Path = DATA_PATH) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_messages(row: dict, include_answer: bool = True) -> list[dict]:
    conversations = row.get("conversations", [])
    if not include_answer:
        conversations = conversations[:-1]
    return [
        {
            "role": "assistant" if turn["role"] == "model" else turn["role"],
            "content": turn["content"],
        }
        for turn in conversations
    ]


def sft_dataset(path: str | Path = DATA_PATH) -> Dataset:
    rows = load_rows(path)
    return Dataset.from_list(
        [{"messages": to_messages(row, include_answer=True)} for row in rows]
    )


def grpo_dataset(path: str | Path = DATA_PATH) -> Dataset:
    rows = load_rows(path)
    return Dataset.from_list(
        [{"prompt": to_messages(row, include_answer=False)} for row in rows]
    )


def is_main_process() -> bool:
    return int(os.getenv("RANK", "0")) == 0


def save_fsdp_adapter(trainer, output_path: str | Path, tokenizer) -> None:
    """Gather only LoRA DTensors instead of materializing the 12B full state."""
    output_path = Path(output_path)
    model = trainer.accelerator.unwrap_model(trainer.model)
    gathered = {}

    # `full_tensor()` is collective, so every rank must visit parameters in
    # exactly the same order even though only rank 0 retains the CPU copies.
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            continue
        tensor = parameter.detach()
        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()
        if is_main_process():
            # Activation-checkpoint wrappers and missing adapter name break PEFT load.
            clean = name.replace("._checkpoint_wrapped_module", "")
            if ".lora_A.weight" in clean and ".lora_A.default.weight" not in clean:
                clean = clean.replace(".lora_A.weight", ".lora_A.default.weight")
            if ".lora_B.weight" in clean and ".lora_B.default.weight" not in clean:
                clean = clean.replace(".lora_B.weight", ".lora_B.default.weight")
            gathered[clean] = tensor.detach().to("cpu").contiguous()

    trainer.accelerator.wait_for_everyone()
    if is_main_process():
        adapter_state = get_peft_model_state_dict(model, state_dict=gathered)
        # Prefer the cleaned gathered tensors if PEFT remapping drops/renames oddly.
        if not adapter_state:
            adapter_state = gathered
        else:
            # Ensure checkpoint-wrapper prefixes are gone in the final dict too.
            adapter_state = {
                k.replace("._checkpoint_wrapped_module", "")
                .replace(".lora_A.weight", ".lora_A.default.weight")
                .replace(".lora_B.weight", ".lora_B.default.weight"): v
                for k, v in adapter_state.items()
            }
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(
            output_path,
            state_dict=adapter_state,
            safe_serialization=True,
            is_main_process=True,
        )
        tokenizer.save_pretrained(output_path)
        print(
            f"Adapter saved to {output_path} "
            f"({len(adapter_state)} tensors)",
            flush=True,
        )
    trainer.accelerator.wait_for_everyone()
