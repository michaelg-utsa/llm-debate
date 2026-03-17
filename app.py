from __future__ import annotations

import streamlit as st

from agents import build_default_agents
from llm_client import OpenAIJSONClient
from orchestrator import DebateOrchestrator
from utils import load_config


st.set_page_config(page_title="LLM Debate UI", layout="wide")


def render_agent_card(title: str, payload: dict, initial: bool = False) -> None:
    st.markdown(f"### {title}")
    answer = payload.get("answer", "—")
    st.markdown(f"**Answer:** {answer}")

    if initial:
        reasoning = payload.get("brief_reasoning", "")
        if reasoning:
            st.markdown("**Initial reasoning**")
            st.write(reasoning)
    else:
        argument = payload.get("argument", "")
        rebuttal = payload.get("rebuttal", "")
        if argument:
            st.markdown("**Argument**")
            st.write(argument)
        if rebuttal:
            st.markdown("**Rebuttal**")
            st.write(rebuttal)


def render_judge_panel(judge: dict) -> None:
    st.subheader("Judge Verdict")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Final answer", judge.get("final_answer", "—"))
    with c2:
        st.metric("Confidence", judge.get("confidence", "—"))

    analysis = judge.get("analysis")
    if analysis:
        st.markdown("**Judge reasoning**")
        st.write(analysis)

    strongest = judge.get("strongest_argument")
    weakest = judge.get("weakest_argument")
    if strongest or weakest:
        left, right = st.columns(2)
        with left:
            if strongest:
                st.markdown("**Strongest argument noticed by judge**")
                st.write(strongest)
        with right:
            if weakest:
                st.markdown("**Weakest argument noticed by judge**")
                st.write(weakest)


config = load_config("config.yaml")

st.title("LLM Debate + Judge Web UI")
st.caption(
    "Enter one yes/no question, choose the ground-truth label, then view initial positions, "
    "round-by-round debate, the judge verdict, and baseline outputs."
)

with st.sidebar:
    st.header("Current settings")
    st.write(
        {
            "domain": config["task"].get("domain"),
            "allowed_answers": config["task"].get("answer_choices"),
            "debate_rounds": config["generation"].get("debate_rounds"),
            "min_rounds_before_early_stop": config["generation"].get("min_rounds_before_early_stop"),
            "early_stop_consecutive": config["generation"].get("early_stop_consecutive"),
        }
    )
    st.info("Set OPENAI_API_KEY in your environment before running the app.")

question = st.text_area(
    "Question",
    value="Did the Roman Empire exist at the same time as the Mayan civilization?",
    height=110,
    help="Enter a StrategyQA-style yes/no question.",
)

ground_truth = st.radio(
    "Ground truth label",
    options=["yes", "no"],
    horizontal=True,
    help="Select the expected answer for this test question.",
)

run_clicked = st.button("Run debate", type="primary", use_container_width=True)

if run_clicked:
    if not question.strip():
        st.error("Please enter a question before running the debate.")
    else:
        with st.spinner("Running debaters, judge, and baselines..."):
            client = OpenAIJSONClient()
            debater_a, debater_b, judge, baselines = build_default_agents(config, client)
            orchestrator = DebateOrchestrator(debater_a, debater_b, judge, baselines, config)
            result = orchestrator.run_single(question.strip(), ground_truth)

        st.success("Run complete. A JSON log was also saved to the logs folder.")

        st.subheader("Question and Ground Truth")
        q1, q2 = st.columns([4, 1])
        with q1:
            st.write(question.strip())
        with q2:
            st.metric("Ground truth", ground_truth)

        st.subheader("Initial Positions")
        left, right = st.columns(2)
        with left:
            render_agent_card("Debater A Initial Position", result["initial_positions"]["debater_a"], initial=True)
        with right:
            render_agent_card("Debater B Initial Position", result["initial_positions"]["debater_b"], initial=True)

        st.subheader("Debate Rounds")
        if result["rounds"]:
            for round_item in result["rounds"]:
                with st.container(border=True):
                    st.markdown(f"## Round {round_item['round']}")
                    round_left, round_right = st.columns(2)
                    with round_left:
                        render_agent_card("Debater A", round_item["debater_a"], initial=False)
                    with round_right:
                        render_agent_card("Debater B", round_item["debater_b"], initial=False)
        else:
            st.info("No debate rounds were needed because the debaters agreed during initialization.")

        render_judge_panel(result["judge"])

        st.subheader("Baselines")
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("### Direct QA")
            st.markdown(f"**Answer:** {result['baselines']['direct_qa'].get('answer', '—')}")
            if result['baselines']['direct_qa'].get('reasoning'):
                st.markdown("**Reasoning**")
                st.write(result['baselines']['direct_qa']['reasoning'])
        with b2:
            sc = result['baselines']['self_consistency']
            st.markdown("### Self-Consistency")
            st.markdown(f"**Majority answer:** {sc.get('majority_answer', '—')}")
            st.markdown(f"**Samples used:** {sc.get('num_samples', '—')}")
            st.write("Vote counts:", sc.get('vote_counts', {}))

        st.subheader("Run Metrics")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Debate rounds played", len(result["rounds"]))
        with m2:
            st.metric("Debate LLM calls", result["metrics"].get("debate_llm_calls", "—"))
        with m3:
            st.metric("Total LLM calls", result["metrics"].get("total_llm_calls", "—"))

        with st.expander("Show raw JSON result"):
            st.json(result)
