#!/usr/bin/env python3
"""Print the newest tqdm progress bars from the GRPO log.

The log interleaves large completion tables with carriage-return progress
bars, so a plain `tail | grep` usually lands inside a table and shows nothing.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

LOG = Path("/workspace/fdsp_summary_format/outputs/logs/grpo_train.log")
BAR = re.compile(r"\s*\d+%\|[^|]*\|\s*\d+/\d+ \[[^\]]+\]")


def main() -> None:
    if not LOG.exists():
        print("log not found:", LOG)
        return

    text = LOG.read_bytes()[-400_000:].decode("utf-8", "replace").replace("\r", "\n")
    bars = BAR.findall(text)

    seen: dict[str, str] = {}
    for bar in bars:
        step = re.search(r"(\d+)/(\d+)", bar)
        if step:
            seen[step.group(1)] = bar.strip()

    for bar in list(seen.values())[-6:]:
        print(bar)

    if seen:
        last = list(seen.values())[-1]
        cur = int(re.search(r"(\d+)/(\d+)", last).group(1))
        total = int(re.search(r"(\d+)/(\d+)", last).group(2))
        rate = re.search(r"([\d.]+)s/it", last)
        if rate:
            left = (total - cur) * float(rate.group(1)) / 3600
            print(f"\n{cur}/{total}  ~{left:.1f}h left  @ {float(rate.group(1)):.0f}s/step")

    lengths = re.findall(r"completions/mean_length.: .([\d.]+)", text)
    clipped = re.findall(r"completions/clipped_ratio.: .([\d.]+)", text)
    rewards = re.findall(r"'reward': '([\d.\-e+]+)'", text)
    if lengths:
        parts = [f"mean_len={lengths[-1]}"]
        if clipped:
            parts.append(f"clipped={clipped[-1]}")
        if rewards:
            parts.append(f"reward={rewards[-1]}")
        print("  ".join(parts))

    ckpts = re.findall(r"\[checkpoint\] step (\d+)", text)
    if ckpts:
        print(f"last checkpoint: step {ckpts[-1]}")

    print(datetime.now(timezone.utc).strftime("updated %H:%M:%SZ"))


if __name__ == "__main__":
    main()
