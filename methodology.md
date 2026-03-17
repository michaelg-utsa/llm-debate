# Methodology

In this project, we implemented a multi-agent debate framework for answering commonsense questions (sampled from the StrategyQA dataset). Three approaches have been compared on the same questions: Multi-round debates, direct answering, and self-consistency sampling. The question we want to answer:

> **Can a structured adversarial debate between two LLM agents, supervised by an LLM judge, produce more accurate and well-reasoned answers than a single LLM answering directly?**

## Overview

- **Domain:** Commonsense QA using 200 yes/no StrategyQA questions
- **Primary Method:** Two argue (or agree), then the judge selects a final answer.
- **Baselines:** Debate, direct QA & self-consistency will all be run on the same questions.
- **Outputs:** Each run produces a full JSON transcript including arguments, verdicts, and metrics

## System Architecture

The debate pipeline is organized into four stages: initialization, iterative debate, judgment, and evaluation. Each stage is handled by a modular Python component so that prompts, stopping criteria, and model configuration can be changed without redesigning the rest of the system.

ChatGPT was used to assist in code generation & formatting, in particular with the UI section, as well as the prompt templates and evaluation metrics.

- **data/** stores the input dataset files used for batch experiments.
- **logs/** stores the JSON outputs for each completed run, including debate transcripts, baseline results, and metrics.
- **agents.py** defines the debater, judge, and baseline agent behaviors that interact with the language model.
- **app.py** provides the Streamlit web UI for entering questions and viewing debate results.
- **batch_run.py** runs the full pipeline over many dataset questions and supports resume/auto-preparation behavior.
- **config.yaml** stores the experiment settings such as model choices, round limits, and answer options.
- **evaluation.py** reads saved logs and computes summary results like accuracy and number of rounds.
- **llm_client.py** handles the actual API calls to the language model service.
- **methodology.md** contains the writeup draft for the Methodology section of the report/site.
- **orchestrator.py** coordinates the end-to-end debate workflow, including initialization, debate rounds, judgment, baselines, and logging.
- **prompts.py** stores the editable prompt templates used by the debaters, judge, and baselines.
- **README.md** explains the project, setup steps, and how to run the system.
- **requirements.txt** lists the Python dependencies needed to install and run the project.
- **run_once.py** runs the full pipeline for a single question from the command line.
- **strategyqa_export_200.py** prepares a 200-question StrategyQA dataset file for experiments if needed.
- **utils.py** provides shared helper functions used across the project, such as normalization and file utilities.

### Model Justification

GPT-5.2 was chosen as the model for all parties in the Multi-round debates, direct answering, and self-consistency sampling. It was chosen because it is a strong general-purpose reasoning model. It is able to reliably follow structured prompts, and decent at "role-playing" as a judge or debater for example. By using the same model for each agent, we were able to control the experiment, as the model capability was consistent. Since the OpenAI API could be used, it was ideal for running batches of questions. Additionally, the API cost was not very expensive.

## Debate Protocol

Each question begins with an initialization phase in which both debaters independently state an answer and brief reasoning. If both agents agree immediately, the system skips directly to the judge. Otherwise, the debate will go on for up to 4 rounds.

In every round, both debaters respond to the other side's reasoning, revising or  adapting their arguments as needed. The adaptive early-stopping rule allows the debate to end sooner if both agents agree for 2 consecutive rounds, making sure at least 3 rounds occur.

### Step-by-step flow

1. **Initialization:** each debater answers independently.
2. **Round debate:** each side provides an updated answer, argument, and rebuttal.
3. **Stopping rule:** debate ends early if convergence is sustained across consecutive rounds.
4. **Judgment:** the judge selects the final answer and provides supporting reasoning.

## Configuration

- **Models:** All parties: Debaters, Judge, Direct QA, and Self-Consistency, used GPT-5.2.
- **Temperature (0.4):** A low temperature was used to keep outputs stable and consistent..
- **Max output tokens (900):** Each model response was capped at 900 tokens.
- **Debate rounds (4):** The debate was allowed to run for up to four rounds.
- **Minimum rounds before early stop (3):** The system required at least three rounds before considering early stopping.
- **Early stop consecutive (2):** Debate ended early if both agents agreed on the same answer for two consecutive rounds.
- **Self-consistency samples (8):** The self-consistency baseline generated eight independent responses, taking the majority answer.
- **Logging )logs)** All outputs are saved in the logs directory.
- **Task domain (strategyqa):** The experiment used the StrategyQA commonsense question answering domain.
- **Answer choices (yes, no):** Self-explanatory.

# Experiments

## Experimental Setup

As per the assignment requirements, I evaluated the pipeline on 200 questions from the StrategyQA dataset. All roles in the system used GPT-5.2.

### Debater A

Debater A produces an initial answer and short justification, then responds in each round with an updated position, argument, and rebuttal against Debater B.

### Debater B

Debater B follows the same structure as Debater A, but is prompted as an independent agent. This creates the adversarial exchange needed for the debate protocol.

### Judge

The judge reviews the transcript after initialization or after the debate rounds complete. It returns a final yes/no answer, confidence estimate, and explanation of which side presented the stronger case.

After the 4-phase debate pipeline described in the Methodology section, the final evaluation output contained 200 rows.
- Each row included per-question predictions for all three methods, along with summary statistics for accuracy, round counts, and average LLM usage.
- The aggregate results were:
  - Debate accuracy: 0.760
  - Direct QA accuracy: 0.775
  - Self-consistency accuracy: 0.745
  - Average rounds: 0.250
  - Average total LLM calls: 8.000
- The round distribution was:
  - 187 debates with 0 rounds
  - 2 debates with 3 rounds
  - 11 debates with 4 rounds

## Main Results

Table 1 shows the core accuracy comparison across all three methods.

| Method | Accuracy | Correct / 200 |
|---|---:|---:|
| Debate + Judge | 76.0% | 152 |
| Direct QA | 77.5% | 155 |
| Self-Consistency | 74.5% | 149 |

All three methods performed similarly, with Direct QA achieving the highest accuracy at 77.5%, followed by Debate + Judge at 76.0%, and Self-Consistency at 74.5%. In this run, the debate pipeline performed well, but it did not outperform the simpler direct-answer baseline. This suggests that structured adversarial interaction between two model instances was not noticeably more effective than a single direct response on this particular 200-question sample, although it still performed slightly better than repeated independent sampling with majority voting.

| Rounds Used | Count | Percentage |
|---|---:|---:|
| 0 | 187 | 93.5% |
| 3 | 2 | 1.0% |
| 4 | 11 | 5.5% |

Because the debate pipeline included adaptive stopping, the number of rounds varied across examples. However, there are no 1 or 2 round examples. This is because the pipeline is designed to go until 4 rounds, or until the debaters agree for 2 rounds. So if they don't agree immediately, they naturally will go for at least 3 rounds. The vast majority of questions were resolved without any debate. This I completely expected; most of these questions are common sense, so of course any decently-trained model is going to come to a common sense conclusion, and most of the time those conclusions will be the same between debaters.

Only 6.5% of the samples involved actual debate rounds, with just 2 ending after 3 rounds and 11 going all the way to 4 rounds. This indicates that only a small subset of questions required extended argumentation, while most were settled immediately.

The average number of rounds per question was only 0.250, which indicates that for most questions in this run, the two GPT-5.2 agents converged in opinions immediately. Since debate performed worse than Direct QA, this suggests that immediate agreement between debaters did not necessarily produce better final outcomes than simply asking a single model directly.

The logs also recorded total LLM calls per question. Since debates with 0 rounds required fewer model calls than debates with 3 or 4 rounds, the actual compute cost varied by example.

| Metric | Value |
|---|---:|
| Average rounds per question | 0.250 |
| Average total LLM calls per question | 8.000 |
| Estimated total LLM calls across 200 questions | 1600 |

This makes the debate pipeline more efficient than a worst-case fixed-round implementation, since most questions ended before entering the more expensive multi-round phase. The more difficult cases could still use additional rounds when needed, though the extra rounds did not translate into a clear overall performance advantage on this sample.

## Statistical Significance

Because all three methods were evaluated on the same 200 questions, I used paired comparisons based on per-question win/loss counts. For each pair of methods, I counted how often one method was correct while the other was wrong, and then applied an exact McNemar-style binomial test to the disagreement counts.

| Comparison | Method A Correct / Method B Wrong | Method A Wrong / Method B Correct | Exact p-value |
|---|---:|---:|---:|
| Debate vs Direct QA | 7 | 10 | 0.6291 |
| Debate vs Self-Consistency | 10 | 7 | 0.6291 |
| Direct QA vs Self-Consistency | 11 | 5 | 0.2101 |

These results were not statistically significant at conventional thresholds (for example, p < 0.05). In other words, although Direct QA had the highest raw accuracy in this run (77.5%), the head-to-head differences between methods were small enough that they could plausibly be due to chance on this 200-question sample.

## Overall Conclusions

Some patterns stand out from these results. 
1. Direct QA achieved the strongest overall performance, finishing slightly ahead of the other two methods.
2. The round distribution shows that most questions did not require extended debate at all; the two debaters often agreed immediately, and the judge then confirmed that consensus. 
3. Despite debate remaining competitive and slightly outperforming Self-Consistency, it did not surpass the simpler Direct QA baseline on this 200-question samples.

The low average round count highlights an important limitation of this experimental setting: because most items terminated at initialization, the benefits of multi-round argumentation were concentrated in a relatively small subset of difficult examples. This means the qualitative transcript analysis in the next section is especially important, since it can show what actually happened in the minority of cases where longer debate was triggered. The experimental results therefore support two conclusions at once: the overall debate pipeline was highly effective, but the strongest evidence for multi-round adversarial reasoning specifically is most visible in the subset of nontrivial debates rather than in the aggregate averages alone. :contentReference

However, I do think that the primary limitation of the experiment should be acknowledged: Because most debates terminated at initialization, the benefits of multi-round arguments were only seen in a relatively small subset of difficult examples. This means the qualitative transcript analysis in the next section is especially important, since it can show what actually happened in the minority of cases where longer debate was triggered. The experiment therefore suggests that while the debate pipeline can produce well-structured answers, the strongest evidence for multi-round adversarial reasoning is the most apparent in cases of certain specific more difficult questions/nontrivial answers than in the aggregate averages.

That being said, the overall answer to our question: "Can a structured adversarial debate between two LLM agents, supervised by an LLM judge, produce more accurate and well-reasoned answers than a single LLM answering directly?" is: **Not conclusively in this experiment.**

# Analysis

To go with the aggregate results, I examined four debate transcripts that represent a useful mix of success cases, failure cases, and ambiguity-driven errors. Together, they show that the debate framework was most helpful when it forced the debaters to consider unspoken assumptions about the question, and less helpful when all components misinterpreted something together. or when the question wording itself was very ambiguous. This pattern fits the theoretical predictions of Irving et al.: **debate is most valuable when one side can expose a flaw in the other side’s reasoning in a way that is legible to the judge, but it is less effective when both sides inherit the same misconception.**

## Case 1: “Did number of Imams Reza Shah believed in exceed number of Jesus's disciples?” — debate successfully resolvs a disagreement due to missing context

This was a strong example of the debate system adding value, as the debate pipeline got the correct answer, whiel both Direct QA and SC were inmcorrect. Both debaters seemed to zero in on important context about Imam Reza, and a hidden assumption necessary to answer the question. The Judge ultimately focused on that assumption, leading it to the correct answer. This shows that debate can be especially helpful when the question needs to be framed in a different way. Connection to Irving et al.’s theory: The debate was useful because it exposed a flaw in earlier reasoning, and made the relevant context known to the judge.

## Case 2: “Will Gremlins sequels tie number of Matrix sequels?” — direct answering succeeds where debate overcommits to the wrong framing

This case was valuable because it shows a clear failure of debate relative to Direct QA. Both debaters immediately agreed on no, and the judge confirmed that consensus without any debate. Their reasoning was straightforward: *Gremlins* has one sequel, while *The Matrix* has three, so the numbers do not tie. Self-Consistency also unanimously selected no. Direct QA returned yes, making it the only correct method on this question.

This is a good reminder that debate is not automatically better. In this transcript, the debate system did not expose a weakness in the initial assumption because both debaters started from the same mistaken frame, and the judge simply chose between the two versions. From the perspective of Irving et al., this is exactly where debate has limited power: if neither side raises the right counterpoint, the judge has no adversarial signal to work with. Debate systems can still fail confidently when all participants inherit the same error.

## Case 3: “Would Atlantic Salmon be within David Duchovny's dietary guidelines?” — longer debate does not guarantee a better answer

This was an interesting failure case as it went all 4 rounds, yet the debate still got the answer wrong, unlike both baselines. A claimed that Duchovny follows is a pescatarian, which would allow salmon, while Debater B argued that Duchovny is vegan, which would exclude fish. Over the 4 rounds, A backpedaled on its claim of pescatarian, focusing more on a general health argument. Debater B maintained the stricter vegan interpretation. The judge sided with Debater B and incorrectly returned no. Both Direct QA and Self-Consistency, by contrast, converged on the pescatarian reading and were correct.

This transcript shows that more rounds do not necessarily improve performance. The debate lasted longer, but the 2 debaters only hardened in their positions. Instead, the judge preferred the more internally consistent framing, even though that framing turned out to be wrong. In other words, debate can fail when rhetorical neatness beats factual correctness.

## Case 4: “Would costumes with robes and pointy hats be helpful for Macbeth?” — all methods fail due to a narrow interpretation of “helpful”

All methods failed in this set. Both debaters agreed no, as those clothes are associated with wizards and witches, which MacBeth is not. The judge concurred, which was incorrect. It even noted a weakness in A's reasoning concerning the meaning of "helpful", and did not consider that those pieces of clothing might be helpful for an actor playing MacBeth.

This case shows a situation where the system failed not because the reasoning was incoherent, but because the interpreted definition of "helpful" was too narrow. All three methods interpreted “helpful for Macbeth” as helpful to the character rather than an actor playing him. It seems that this level of ambiguity can cause all methods to fail equally. If both sides share the same underlying interpretation, there is no adversarial correction for the judge to detect.

## Overall Takeaways

In all these examples, the framework works the best when there is a hidden assumption about the question that needs to be identified. The strongest success case occurred when the adversarial exchange caused the framing to be clarified for the judge.

Simultaneously, the failure cases indicated that debate is not automatically superior. In cases where both debaters begin with the same mistaken assumption, the judge has little useful contrast to evaluate.

Case 3 shows that more debate does not necessarily equate to better reasoning. Additional rounds could potentially just reinforce the debater's positions rather than have them reconsider.

This is consistent with the experiment results earlier: "Can a structured adversarial debate between two LLM agents, supervised by an LLM judge, produce more accurate and well-reasoned answers than a single LLM answering directly?" **Not conclusively in this experiment.**

# Prompt Engineering

The prompt design process was relatively simple. The most important thing was making each agent's role as clear as possible. to do this, outputs were structured in order to reduce the chance that the models drift into unhelpful reasoning.

The explicit role framing given:
- Debater A was instructed to defend its current answer honestly and persuasively, 
- Debater B was instructed to critique the other side and defend its own answer
- The Judge was instructed to review the original question, the initial positions, and the transcript, then choose the best-supported answer.

The prompts were kept structured but not overly restrictive. For the debaters, I required JSON outputs with fields:

- answer
- argument
- rebuttal
- confidence

This helped keep each round consistent and made downstream logging and evaluation easier. For the judge, I required fields:
- analysis
- strongest_a
- weakest_a
- strongest_b
- weakest_b
- winner
- final_answer
- confidence 

This forced the judge to explicitly compare the two sides instead of simply giving a final label. The use of constrained JSON output was important both for reproducibility and for building the UI. 

Another key decision involved reasoning instructions. I wanted the agents to provide enough explanation to make the debates interpretable, but I did not want chain-of-thought style outputs that were unreasonably long. I asked for short but explicit reasonings: “2–5 sentences” for arguments and “1–3 sentences” for rebuttals. This gave the system enough room to express a real position while keeping the debates readable and affordable. In practice, this worked reasonably well: the transcripts were detailed enough for qualitative analysis, but still compact enough to run across 200 questions.

A mistake that caused my first iteration to fail was accidentally providing the judge agent with the true answer in its prompting. As a result, it had a suspicious 99% accuracy rate that caused me to become aware of the issue. The debates still worked correctly, however they did not tend to matter much, as the judge already knew the answer. These logs were discarded in favor of a new set of experiments where the judge is not given the answer beforehand, leading to much less biased results.

Overall, the prompt engineering process was iterative, and required some trial and error. My prompts aimed to be clear in role separation, and restricted in answering, but less restricted on reasoning.

# Appendix

## Full Prompts

Below are the complete final prompt templates used for the three agents. Variable placeholders are shown exactly as they appear in the code, using curly braces such as {question} and {allowed_answers}.

<details>
<summary><strong>Debater A</strong></summary>

### System Prompt

```text
You are Debater A in an academic LLM debate system.
Your goal is to argue FOR your current answer as persuasively as possible while remaining honest.
Be analytical, specific, and concise. Use evidence from the question and general knowledge.
Return valid JSON only. Do not wrap JSON in markdown fences.
```

### Initial Position Template

```text
Domain: {domain}
Question: {question}
Allowed answers: {allowed_answers}

Give your independent initial position without seeing the other debater.
Return JSON with exactly these keys:
- answer: one of the allowed answers
- brief_reasoning: 2-4 sentences
- confidence: integer from 1 to 5
```

### Debate Round Template

```text
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
```

</details>

<details>
<summary><strong>Debater B</strong></summary>

### System Prompt

```text
You are Debater B in an academic LLM debate system.
Your goal is to critique Debater A, defend your own answer, and expose weaknesses in the other side.
Be analytical, specific, and concise. Use evidence from the question and general knowledge.
Return valid JSON only. Do not wrap JSON in markdown fences.
```

### Initial Position Template

```text
Domain: {domain}
Question: {question}
Allowed answers: {allowed_answers}

Give your independent initial position without seeing the other debater.
Return JSON with exactly these keys:
- answer: one of the allowed answers
- brief_reasoning: 2-4 sentences
- confidence: integer from 1 to 5
```

### Debate Round Template

```text
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
```

</details>

<details>
<summary><strong>Judge</strong></summary>

### System Prompt

```text
You are the Judge in an academic LLM debate system.
You will review the original question, the debaters' initial positions, and the debate transcript.
You do NOT know the ground truth answer and must not assume access to it.
Judge only from the reasoning presented and your general knowledge.
Be fair, explicit, and structured.
Return valid JSON only. Do not wrap JSON in markdown fences.
```

### Judgment Template

```text
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
```

</details>