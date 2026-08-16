"""Deterministic format rewards carried over from the original GRPO flow."""

from __future__ import annotations

import json
import re
from typing import Any


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _content(value.get("content", ""))
    if isinstance(value, list):
        return "\n".join(_content(item) for item in value)
    return str(value)


def _requested_items(prompt: str) -> list[str]:
    match = re.search(r"대항목명은\s*(.+?)\s*(?:입니다|\.|$)", prompt)
    if not match:
        match = re.search(r"항목명은\s*(.+?)\s*(?:입니다|\.|$)", prompt)
    return [item.strip() for item in match.group(1).split(",")] if match else []


def _sub_items(prompt: str, major: str) -> list[str]:
    match = re.search(
        rf"{re.escape(major)}\s*의\s*항목명은\s*(.+?)\s*(?:입니다|\.|$)",
        prompt,
    )
    return [item.strip() for item in match.group(1).split(",")] if match else []


def _strip_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:jsonl|json|html|markdown|md)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _json_reward(response: str, prompt: str) -> float:
    try:
        parsed = json.loads(_strip_fence(response))
    except (TypeError, json.JSONDecodeError):
        return 0.0
    if not isinstance(parsed, dict):
        return 0.0

    majors = _requested_items(prompt)
    if not majors:
        return 3.0
    major_hits = sum(major in parsed for major in majors)
    expected, sub_hits = 0, 0
    for major in majors:
        subs = _sub_items(prompt, major)
        value = parsed.get(major)
        if subs:
            expected += len(subs)
            if isinstance(value, dict):
                sub_hits += sum(sub in value and str(value[sub]).strip() for sub in subs)
        else:
            expected += 1
            sub_hits += int(value is not None and bool(str(value).strip()))
    score = 3.5 * major_hits / len(majors)
    score += 3.5 * sub_hits / expected if expected else 0.0
    return score + (2.0 if score >= 5.0 else 0.0)


def _markdown_table_reward(response: str, prompt: str) -> float:
    text = _strip_fence(response)
    valid_table = bool(
        re.search(r"^\s*\|.*\|\s*$\n^\s*\|[-:|\s]+\|\s*$", text, re.M)
    )
    score = 3.0 if valid_table else 0.0
    items = _requested_items(prompt)
    if items:
        score += 2.0 * sum(item in text for item in items) / len(items)
    return score + (2.0 if score >= 5.0 else 0.0)


def _markdown_reward(response: str, prompt: str) -> float:
    has_structure = bool(
        re.search(r"^\s*(?:[-*]|\d+\.)\s+.+", response, re.M)
        or re.search(r"^\s*\*\*[^*\n]+\*\*", response, re.M)
    )
    score = 3.0 if has_structure else 0.0
    items = _requested_items(prompt)
    if items:
        score += 2.0 * sum(
            bool(re.search(rf"\*\*{re.escape(item)}\*\*", response))
            for item in items
        ) / len(items)
    return score + (2.0 if score >= 5.0 else 0.0)


def _html_table_reward(response: str, prompt: str) -> float:
    has_table = bool(re.search(r"<table\b.*?</table>", response, re.I | re.S))
    balanced = all(
        len(re.findall(rf"<{tag}\b", response, re.I))
        == len(re.findall(rf"</{tag}>", response, re.I))
        for tag in ("table", "tr", "th", "td")
    )
    score = 3.0 if has_table and balanced else 0.0
    items = _requested_items(prompt)
    if items:
        headers = re.findall(r"<th\b[^>]*>(.*?)</th>", response, re.I | re.S)
        header_text = " ".join(re.sub(r"<[^>]+>", "", h).strip() for h in headers)
        score += 2.0 * sum(item in header_text for item in items) / len(items)
    return score + (2.0 if score >= 5.0 else 0.0)


def _html_reward(response: str, prompt: str) -> float:
    has_lists = bool(re.search(r"<ul\b.*?</ul>", response, re.I | re.S))
    balanced = all(
        len(re.findall(rf"<{tag}\b", response, re.I))
        == len(re.findall(rf"</{tag}>", response, re.I))
        for tag in ("ul", "li", "b")
    )
    score = 3.0 if has_lists and balanced else 0.0
    items = _requested_items(prompt)
    if items:
        score += 2.0 * sum(item in response for item in items) / len(items)
    return score + (2.0 if score >= 5.0 else 0.0)


def format_reward(completions, prompts, **_) -> list[float]:
    """Select the same format-specific reward from each sample's prompt."""
    rewards = []
    for completion, prompt in zip(completions, prompts, strict=True):
        response_text = _content(completion)
        prompt_text = _content(prompt)
        lower = prompt_text.lower()
        if "html table" in lower:
            reward = _html_table_reward(response_text, prompt_text)
        elif "html" in lower:
            reward = _html_reward(response_text, prompt_text)
        elif "markdown table" in lower:
            reward = _markdown_table_reward(response_text, prompt_text)
        elif "마크다운" in lower or "markdown" in lower:
            reward = _markdown_reward(response_text, prompt_text)
        elif "jsonl" in lower or "json" in lower:
            reward = _json_reward(response_text, prompt_text)
        else:
            reward = 0.0
        rewards.append(float(reward))
    return rewards
