from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from agents import BaselineRunner, Debater, Judge
from utils import ensure_dir, majority_vote, normalize_answer, save_json


class DebateOrchestrator:
    def __init__(
        self,
        debater_a: Debater,
        debater_b: Debater,
        judge: Judge,
        baselines: BaselineRunner,
        config: dict[str, Any],
    ) -> None:
        self.debater_a = debater_a
        self.debater_b = debater_b
        self.judge = judge
        self.baselines = baselines
        self.config = config
        self.log_dir = ensure_dir(config["logging"]["log_dir"])

    def _format_transcript(self, rounds: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for round_item in rounds:
            round_number = round_item["round"]
            a = round_item["debater_a"]
            b = round_item["debater_b"]
            blocks.append(
                f"Round {round_number}\n"
                f"Debater A answer: {a['answer']}\n"
                f"Debater A argument: {a['argument']}\n"
                f"Debater A rebuttal: {a['rebuttal']}\n"
                f"Debater B answer: {b['answer']}\n"
                f"Debater B argument: {b['argument']}\n"
                f"Debater B rebuttal: {b['rebuttal']}"
            )
        return "\n\n".join(blocks)

    def _format_initial_positions(self, initial_a: dict[str, Any], initial_b: dict[str, Any]) -> str:
        return (
            f"Debater A initial answer: {initial_a['answer']}\n"
            f"Debater A reasoning: {initial_a['brief_reasoning']}\n\n"
            f"Debater B initial answer: {initial_b['answer']}\n"
            f"Debater B reasoning: {initial_b['brief_reasoning']}"
        )

    def _evaluate(self, predicted: str | None, ground_truth: str | None) -> dict[str, Any]:
        if ground_truth is None or predicted is None:
            return {"correct": None}
        return {"correct": normalize_answer(predicted) == normalize_answer(ground_truth)}

    def run_single(self, question: str, ground_truth: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        max_rounds = self.config["generation"]["debate_rounds"]
        min_rounds_before_early_stop = self.config["generation"]["min_rounds_before_early_stop"]
        early_stop_consecutive = self.config["generation"]["early_stop_consecutive"]

        initial_a = self.debater_a.initial_position(question)
        initial_b = self.debater_b.initial_position(question)
        rounds: list[dict[str, Any]] = []
        llm_calls = 2
        consensus_streak = 0
        current_a_answer = initial_a["answer"]
        current_b_answer = initial_b["answer"]
        consensus_after_init = normalize_answer(current_a_answer) == normalize_answer(current_b_answer)

        if not consensus_after_init:
            for round_number in range(1, max_rounds + 1):
                transcript_so_far = self._format_transcript(rounds)
                round_a = self.debater_a.debate_round(question, transcript_so_far, round_number, current_a_answer)
                transcript_for_b = self._format_transcript(
                    rounds + [{"round": round_number, "debater_a": round_a, "debater_b": {"answer": "pending", "argument": "pending", "rebuttal": "pending"}}]
                )
                round_b = self.debater_b.debate_round(question, transcript_for_b, round_number, current_b_answer)
                llm_calls += 2
                current_a_answer = round_a["answer"]
                current_b_answer = round_b["answer"]

                rounds.append(
                    {
                        "round": round_number,
                        "debater_a": round_a,
                        "debater_b": round_b,
                    }
                )

                if normalize_answer(current_a_answer) == normalize_answer(current_b_answer):
                    consensus_streak += 1
                else:
                    consensus_streak = 0

                if round_number >= min_rounds_before_early_stop and consensus_streak >= early_stop_consecutive:
                    break

        transcript = self._format_transcript(rounds)
        initial_positions = self._format_initial_positions(initial_a, initial_b)
        judge_result = self.judge.decide(question, initial_positions, transcript)
        llm_calls += 1
        debate_llm_calls = llm_calls

        direct_qa = self.baselines.direct_qa(question)
        llm_calls += 1

        sc_sample_count = debate_llm_calls
        sc_samples = [self.baselines.self_consistency_sample(question) for _ in range(sc_sample_count)]
        llm_calls += sc_sample_count
        sc_answers = [sample["answer"] for sample in sc_samples]
        sc_majority = majority_vote(sc_answers)
        sc_vote_counts = dict(Counter(sc_answers))

        result = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "question": question,
            "ground_truth": ground_truth,
            "metadata": metadata or {},
            "config": self.config,
            "initial_positions": {
                "debater_a": initial_a,
                "debater_b": initial_b,
            },
            "rounds": rounds,
            "judge": judge_result,
            "baselines": {
                "direct_qa": direct_qa,
                "self_consistency": {
                    "num_samples": sc_sample_count,
                    "samples": sc_samples,
                    "majority_answer": sc_majority,
                    "vote_counts": sc_vote_counts,
                },
            },
            "metrics": {
                "debate_llm_calls": debate_llm_calls,
                "llm_calls_for_debate_plus_direct": llm_calls - sc_sample_count,
                "total_llm_calls": llm_calls,
                "debate_correct": self._evaluate(judge_result.get("final_answer"), ground_truth),
                "direct_qa_correct": self._evaluate(direct_qa.get("answer"), ground_truth),
                "self_consistency_correct": self._evaluate(sc_majority, ground_truth),
            },
        }

        filename = datetime.utcnow().strftime("debate_%Y%m%d_%H%M%S_%f.json")
        save_json(result, Path(self.log_dir) / filename)
        return result
