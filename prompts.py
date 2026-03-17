SYSTEM_DEBATER_A = """You are Debater A in an academic LLM debate system.
Your goal is to argue FOR your current answer as persuasively as possible while remaining honest.
Be analytical, specific, and concise. Use evidence from the question and general knowledge.
Return valid JSON only. Do not wrap JSON in markdown fences.
"""

SYSTEM_DEBATER_B = """You are Debater B in an academic LLM debate system.
Your goal is to critique Debater A, defend your own answer, and expose weaknesses in the other side.
Be analytical, specific, and concise. Use evidence from the question and general knowledge.
Return valid JSON only. Do not wrap JSON in markdown fences.
"""

SYSTEM_JUDGE = """You are the Judge in an academic LLM debate system.
You will review the original question, the debaters' initial positions, and the debate transcript.
You do NOT know the ground truth answer and must not assume access to it.
Judge only from the reasoning presented and your general knowledge.
Be fair, explicit, and structured.
Return valid JSON only. Do not wrap JSON in markdown fences.
"""

INITIAL_TEMPLATE = """
Domain: {domain}
Question: {question}
Allowed answers: {allowed_answers}

Give your independent initial position without seeing the other debater.
Return JSON with exactly these keys:
- answer: one of the allowed answers
- brief_reasoning: 2-4 sentences
- confidence: integer from 1 to 5
""".strip()

ROUND_TEMPLATE = """
Domain: {domain}
Question: {question}
Allowed answers: {allowed_answers}
Your current role: {role_name}
Your current answer: {current_answer}

Debate transcript so far:
{transcript}

Respond for round {round_number}.
Return JSON with exactly these keys:
- answer: one of the allowed answers
- argument: 2-5 sentences defending your answer
- rebuttal: 1-3 sentences directly addressing the opponent's latest point
- confidence: integer from 1 to 5
""".strip()

JUDGE_TEMPLATE = """
Domain: {domain}
Question: {question}
Allowed answers: {allowed_answers}

Initial positions:
{initial_positions}

Debate transcript:
{transcript}

Evaluate the debate and choose the best-supported final answer.
Do not assume access to any hidden label or ground truth.
Return JSON with exactly these keys:
- analysis: 4-8 sentences summarizing why one side was more persuasive
- strongest_a: short string
- weakest_a: short string
- strongest_b: short string
- weakest_b: short string
- winner: one of [Debater A, Debater B, Tie]
- final_answer: one of the allowed answers
- confidence: integer from 1 to 5
""".strip()

DIRECT_QA_TEMPLATE = """
Domain: {domain}
Question: {question}
Allowed answers: {allowed_answers}

Answer directly without debate.
Return JSON with exactly these keys:
- answer: one of the allowed answers
- reasoning: 2-5 sentences
- confidence: integer from 1 to 5
""".strip()

SELF_CONSISTENCY_TEMPLATE = """
Domain: {domain}
Question: {question}
Allowed answers: {allowed_answers}

Answer independently. This sample should not refer to any other samples.
Return JSON with exactly these keys:
- answer: one of the allowed answers
- reasoning: 2-5 sentences
- confidence: integer from 1 to 5
""".strip()
