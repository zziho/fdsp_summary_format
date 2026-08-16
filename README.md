# Gemma 3 12B FSDP training pipeline

This reproduces the original `summary_format` flow on two GPUs without Unsloth:

1. BF16 LoRA SFT from Gemma 3 12B IT
2. Merge the SFT adapter
3. BF16 LoRA GRPO from the merged SFT model
4. Merge the GRPO adapter

The source dataset is
`../summary_format/train_data/model_input_fixed.jsonl`. SFT consumes all
`system + user + assistant` turns. GRPO reuses only `system + user`, generates
four completions, and applies the original format-oriented rewards.

## Run

```bash
uv pip install --python /venv/main/bin/python -r requirements.txt
bash run_pipeline.sh
```

Outputs:

- `outputs/sft_adapter`
- `outputs/sft_merged`
- `outputs/grpo_adapter`
- `outputs/grpo_merged`

Both training stages use `torchrun` with two processes and Transformers
FSDP `full_shard auto_wrap`. `use_orig_params=true` is required because LoRA
mixes frozen base parameters with trainable adapter parameters.

For a short integration test:

```bash
/venv/main/bin/torchrun --standalone --nproc_per_node=2 train_sft.py \
  --limit 32 --max-steps 1
```
