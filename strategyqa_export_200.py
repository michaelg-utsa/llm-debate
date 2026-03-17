from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

RAW_JSON_CANDIDATES = [
    "https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/train.json",
    "https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/dev.json",
]

HF_DATASET_NAME = "wics/strategy-qa"


def count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def normalize_answer(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"

    text = str(value).strip().lower()
    if text in {"yes", "true", "1"}:
        return "yes"
    if text in {"no", "false", "0"}:
        return "no"
    raise ValueError(f"Unrecognized answer value: {value!r}")


def extract_question(item: dict) -> str | None:
    for key in ("question", "input", "query"):
        value = item.get(key)
        if value:
            return str(value).strip()
    return None


def extract_answer(item: dict) -> str | None:
    for key in ("answer", "label", "target"):
        if key in item and item[key] is not None:
            try:
                return normalize_answer(item[key])
            except ValueError:
                pass
    return None


def iter_rows_from_json_items(items: Iterable[dict]):
    for item in items:
        question = extract_question(item)
        answer = extract_answer(item)
        if question and answer:
            yield {"question": question, "answer": answer}


def fetch_json_from_url(url: str):
    with urlopen(url) as response:
        return json.load(response)


def load_source_rows(split: str):
    last_error = None

    for url in RAW_JSON_CANDIDATES:
        try:
            data = fetch_json_from_url(url)
            if isinstance(data, list):
                rows = list(iter_rows_from_json_items(data))
                if rows:
                    return rows
        except Exception as exc:
            last_error = exc

    try:
        from datasets import load_dataset  # type: ignore

        dataset = load_dataset(HF_DATASET_NAME, split=split)
        rows = []
        for item in dataset:
            question = extract_question(item)
            answer = extract_answer(item)
            if question and answer:
                rows.append({"question": question, "answer": answer})
        if rows:
            return rows
    except Exception as exc:
        last_error = exc

    raise RuntimeError(
        "Could not load StrategyQA data from raw JSON URLs or Hugging Face."
    ) from last_error


def export_strategyqa_random_sample(
    output_path: str,
    limit: int = 200,
    force: bool = False,
    split: str = "train",
    seed: int = 42,
) -> str:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = count_nonempty_lines(output_file)
    if existing_rows >= limit and not force:
        print(f"[skip] {output_file} already has {existing_rows} rows.")
        return str(output_file)

    rows = load_source_rows(split)

    import random
    rng = random.Random(seed)
    rng.shuffle(rows)

    selected = rows[:limit]

    with output_file.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[done] Wrote {len(selected)} randomly sampled rows to {output_file} using seed={seed}")
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(description="Export a random sample of StrategyQA questions to JSONL.")
    parser.add_argument("--output", default="data/strategyqa_200.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    export_strategyqa_random_sample(
        output_path=args.output,
        limit=args.limit,
        force=args.force,
        split=args.split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
