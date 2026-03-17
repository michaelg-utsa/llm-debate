from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def majority_vote(items: list[str]) -> str:
    counts = Counter(items)
    return counts.most_common(1)[0][0]


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON in model output:\n{text}")
    return json.loads(match.group(0))


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
