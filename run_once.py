from __future__ import annotations

import argparse

from agents import build_default_agents
from llm_client import OpenAIJSONClient
from orchestrator import DebateOrchestrator
from utils import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    client = OpenAIJSONClient()
    debater_a, debater_b, judge, baselines = build_default_agents(config, client)
    orchestrator = DebateOrchestrator(debater_a, debater_b, judge, baselines, config)
    result = orchestrator.run_single(args.question, args.ground_truth)

    print("Judge final answer:", result["judge"]["final_answer"])
    print("Judge analysis:", result["judge"]["analysis"])
    print("Direct QA answer:", result["baselines"]["direct_qa"]["answer"])
    print("Self-consistency majority:", result["baselines"]["self_consistency"]["majority_answer"])
