# Overnight pipeline status

Updated: 2026-08-16T19:05Z

## What's running (agent disconnect-safe)

1. **SFT** (already in progress): pid `14063` / `torchrun train_sft.py`  
   - Base: `/workspace/models/google-gemma-3-12b-it`  
   - NVLink P2P: `NCCL_P2P_DISABLE=0`  
   - IB: enabled in env, no device on host  

2. **Continuer** (nohup): `bash ./continue_after_sft.sh` (pid ~19008)  
   - Waits for SFT to finish  
   - Then: extract metrics → SFT report → **merge SFT** → **GRPO** → GRPO report → **merge GRPO**

## Where to look when you wake up

| Item | Path |
|------|------|
| Continuer log | `outputs/logs/continue_after_sft.log` |
| SFT train log | `outputs/logs/sft_train.log` |
| GRPO train log | `outputs/logs/grpo_train.log` |
| SFT report | `reports/SFT_REPORT.md` |
| GRPO report | `reports/GRPO_REPORT.md` |
| Artifacts | `outputs/sft_adapter`, `sft_merged`, `grpo_adapter`, `grpo_merged` |

## Check commands

```bash
tail -50 /workspace/fdsp_summary_format/outputs/logs/continue_after_sft.log
pgrep -af 'train_sft|train_grpo|continue_after_sft|merge_adapter'
ls -la /workspace/fdsp_summary_format/outputs/{sft_adapter,sft_merged,grpo_adapter,grpo_merged} 2>/dev/null
```

Note: full GRPO can take a very long time (generation-heavy). Continuer will keep going on the Vast instance even if Cursor/agent disconnects.
