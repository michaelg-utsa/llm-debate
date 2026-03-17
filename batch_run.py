from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from agents import build_default_agents
from llm_client import OpenAIJSONClient
from orchestrator import DebateOrchestrator
from utils import load_config


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_nonempty_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def looks_like_strategyqa_target(path: Path) -> bool:
    name = path.name.lower()
    return "strategyqa" in name


def find_exporter_script(base_dir: Path) -> Path | None:
    candidates = [
        base_dir / "strategyqa_export_200_fixed.py",
        base_dir / "strategyqa_export_200.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def maybe_prepare_strategyqa_file(data_path: Path, needed_rows: int, force_prepare: bool = False) -> None:
    existing_rows = count_nonempty_jsonl_rows(data_path)
    if existing_rows >= needed_rows:
        print(f"Dataset ready: {data_path} already has {existing_rows} rows.")
        return

    if not force_prepare and not looks_like_strategyqa_target(data_path):
        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {data_path}. If this is intended to be StrategyQA, "
                f"rename it to include 'strategyqa' or use --auto-prepare-strategyqa."
            )
        print(
            f"Dataset has only {existing_rows} rows, but auto-prepare was skipped because "
            f"{data_path.name} does not look like a StrategyQA file."
        )
        return

    exporter = find_exporter_script(Path(__file__).resolve().parent)
    if exporter is None:
        raise FileNotFoundError(
            "Could not find strategyqa_export_200_fixed.py or strategyqa_export_200.py "
            "next to batch_run.py."
        )

    print(
        f"Preparing StrategyQA file because {data_path} has {existing_rows} rows "
        f"and {needed_rows} are needed..."
    )
    cmd = [
        sys.executable,
        str(exporter),
        "--output",
        str(data_path),
        "--limit",
        str(needed_rows),
    ]
    subprocess.run(cmd, check=True)

    final_rows = count_nonempty_jsonl_rows(data_path)
    if final_rows < needed_rows:
        raise RuntimeError(
            f"Exporter ran, but {data_path} still has only {final_rows} rows (needed {needed_rows})."
        )

    print(f"Prepared {data_path} with {final_rows} rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_strategyqa.jsonl")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--auto-prepare-strategyqa",
        action="store_true",
        help=(
            "Force automatic StrategyQA dataset preparation before running, even if the "
            "data filename does not include 'strategyqa'."
        ),
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    maybe_prepare_strategyqa_file(
        data_path=data_path,
        needed_rows=args.limit,
        force_prepare=args.auto_prepare_strategyqa,
    )

    config = load_config(args.config)
    client = OpenAIJSONClient()
    debater_a, debater_b, judge, baselines = build_default_agents(config, client)
    orchestrator = DebateOrchestrator(debater_a, debater_b, judge, baselines, config)

    rows = load_jsonl(str(data_path))[: args.limit]
    for i, row in enumerate(rows, start=1):
        print(f"Running example {i}/{len(rows)}: {row['question']}")
        result = orchestrator.run_single(
            question=row["question"],
            ground_truth=row.get("answer"),
            metadata={"row_index": i, "source_file": data_path.name},
        )
        print("  debate:", result["judge"].get("final_answer"))
        print("  direct:", result["baselines"]["direct_qa"].get("answer"))
        print("  sc:", result["baselines"]["self_consistency"].get("majority_answer"))
