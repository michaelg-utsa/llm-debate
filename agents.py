from __future__ import annotations

from dataclasses import dataclass

from llm_client import OpenAIJSONClient
from prompts import (
    DIRECT_QA_TEMPLATE,
    INITIAL_TEMPLATE,
    JUDGE_TEMPLATE,
    ROUND_TEMPLATE,
    SELF_CONSISTENCY_TEMPLATE,
    SYSTEM_DEBATER_A,
    SYSTEM_DEBATER_B,
    SYSTEM_JUDGE,
)


@dataclass
class Debater:
    name: str
    system_prompt: str
    model: str
    client: OpenAIJSONClient
    domain: str
    allowed_answers: list[str]
    temperature: float
    max_output_tokens: int

    def initial_position(self, question: str) -> dict:
        prompt = INITIAL_TEMPLATE.format(
            domain=self.domain,
            question=question,
            allowed_answers=", ".join(self.allowed_answers),
        )
        return self.client.generate_json(
            model=self.model,
            instructions=self.system_prompt,
            user_input=prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )

    def debate_round(self, question: str, transcript: str, round_number: int, current_answer: str) -> dict:
        prompt = ROUND_TEMPLATE.format(
            domain=self.domain,
            question=question,
            allowed_answers=", ".join(self.allowed_answers),
            role_name=self.name,
            current_answer=current_answer,
            transcript=transcript or "No prior rounds.",
            round_number=round_number,
        )
        return self.client.generate_json(
            model=self.model,
            instructions=self.system_prompt,
            user_input=prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )


@dataclass
class Judge:
    model: str
    client: OpenAIJSONClient
    domain: str
    allowed_answers: list[str]
    temperature: float
    max_output_tokens: int

    def decide(self, question: str, initial_positions: str, transcript: str) -> dict:
        prompt = JUDGE_TEMPLATE.format(
            domain=self.domain,
            question=question,
            allowed_answers=", ".join(self.allowed_answers),
            initial_positions=initial_positions,
            transcript=transcript or "No debate rounds occurred.",
        )
        return self.client.generate_json(
            model=self.model,
            instructions=SYSTEM_JUDGE,
            user_input=prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )


class BaselineRunner:
    def __init__(
        self,
        client: OpenAIJSONClient,
        domain: str,
        allowed_answers: list[str],
        direct_model: str,
        sc_model: str,
        temperature: float,
        max_output_tokens: int,
    ) -> None:
        self.client = client
        self.domain = domain
        self.allowed_answers = allowed_answers
        self.direct_model = direct_model
        self.sc_model = sc_model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def direct_qa(self, question: str) -> dict:
        prompt = DIRECT_QA_TEMPLATE.format(
            domain=self.domain,
            question=question,
            allowed_answers=", ".join(self.allowed_answers),
        )
        return self.client.generate_json(
            model=self.direct_model,
            instructions="You are answering directly, without debate. Return JSON only.",
            user_input=prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )

    def self_consistency_sample(self, question: str) -> dict:
        prompt = SELF_CONSISTENCY_TEMPLATE.format(
            domain=self.domain,
            question=question,
            allowed_answers=", ".join(self.allowed_answers),
        )
        return self.client.generate_json(
            model=self.sc_model,
            instructions="You are producing one independent sample for self-consistency. Return JSON only.",
            user_input=prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )


def build_default_agents(config: dict, client: OpenAIJSONClient) -> tuple[Debater, Debater, Judge, BaselineRunner]:
    domain = config["task"]["domain"]
    allowed_answers = config["task"]["answer_choices"]
    temperature = config["generation"]["temperature"]
    max_output_tokens = config["generation"]["max_output_tokens"]

    debater_a = Debater(
        name="Debater A",
        system_prompt=SYSTEM_DEBATER_A,
        model=config["models"]["debater_a"],
        client=client,
        domain=domain,
        allowed_answers=allowed_answers,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    debater_b = Debater(
        name="Debater B",
        system_prompt=SYSTEM_DEBATER_B,
        model=config["models"]["debater_b"],
        client=client,
        domain=domain,
        allowed_answers=allowed_answers,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    judge = Judge(
        model=config["models"]["judge"],
        client=client,
        domain=domain,
        allowed_answers=allowed_answers,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    baselines = BaselineRunner(
        client=client,
        domain=domain,
        allowed_answers=allowed_answers,
        direct_model=config["models"]["direct_qa"],
        sc_model=config["models"]["self_consistency"],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return debater_a, debater_b, judge, baselines
