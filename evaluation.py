from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


EXCLUDED_FILES = {
    "debate_20260313_165029_712665.json",
}


def summarize_logs(log_dir: str = "logs") -> pd.DataFrame:
    rows: list[dict] = []
    for path in Path(log_dir).glob("*.json"):
        if path.name in EXCLUDED_FILES:
            continue

        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)

        rows.append(
            {
                "file": path.name,
                "question": item["question"],
                "ground_truth": item.get("ground_truth"),
                "debate_answer": item["judge"].get("final_answer"),
                "direct_answer": item["baselines"]["direct_qa"].get("answer"),
                "sc_answer": item["baselines"]["self_consistency"].get("majority_answer"),
                "debate_correct": item["metrics"]["debate_correct"].get("correct"),
                "direct_correct": item["metrics"]["direct_qa_correct"].get("correct"),
                "sc_correct": item["metrics"]["self_consistency_correct"].get("correct"),
                "rounds": len(item["rounds"]),
                "total_llm_calls": item["metrics"]["total_llm_calls"],
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("file").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = summarize_logs()
    if df.empty:
        print("No logs found.")
    else:
        print(df)
        print(f"\nRows included: {len(df)}")

        print("\nAccuracy summary:")
        print(df[["debate_correct", "direct_correct", "sc_correct"]].mean(numeric_only=True))

        print(f"\nAverage rounds: {df['rounds'].mean():.3f}")

        print("\nRound distribution:")
        print(df["rounds"].value_counts().sort_index())

        print(f"\nAverage total_llm_calls: {df['total_llm_calls'].mean():.3f}")

        round_summary = (
            df["rounds"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        round_summary.columns = ["rounds", "count"]
        round_summary.to_csv("round_summary.csv", index=False)
        print("\nSaved round_summary.csv")
