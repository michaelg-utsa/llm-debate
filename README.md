# LLM Debate System

This project implements a multi-agent LLM debate pipeline for **StrategyQA** yes/no commonsense questions.

## Project Files

- `app.py` — Streamlit web UI for entering a question and viewing the debate
- `run_once.py` — runs the full pipeline for one question
- `batch_run.py` — runs the pipeline over many dataset questions
- `agents.py` — defines Debater A, Debater B, Judge, and baseline agent behavior
- `orchestrator.py` — coordinates initialization, debate rounds, judgment, baselines, and logging
- `prompts.py` — editable prompt templates for all agents
- `llm_client.py` — OpenAI API wrapper
- `evaluation.py` — reads logs and summarizes results
- `config.yaml` — model and experiment settings
- `strategyqa_export_200.py` — creates a random 200-question StrategyQA JSONL file
- `utils.py` — helper utilities
- `data/` — dataset files
- `logs/` — one JSON log per completed run

## Requirements

- Python 3.10+
- OpenAI API key
- Internet access for API calls

## Setup (Windows Command Prompt)

Open **Command Prompt** and move into the project folder:

```cmd
cd C:\Users\micha\llm_debate
```

### 1. Create a virtual environment

```cmd
python -m venv .venv
```

### 2. Activate the virtual environment

```cmd
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```cmd
pip install -r requirements.txt
```

### 4. Set your OpenAI API key

```cmd
set OPENAI_API_KEY=your_key_here
```

## Configuration

The main experiment settings are stored in `config.yaml`.

Current settings include:

- **Model for all roles:** GPT-5.2
- **Temperature:** 0.4
- **Max output tokens:** 900
- **Max debate rounds:** 4
- **Minimum rounds before early stop:** 3
- **Early stop after consecutive agreement rounds:** 2
- **Self-consistency samples:** 8
- **Task domain:** StrategyQA
- **Allowed answers:** yes / no

## Generate a Random 200-Question StrategyQA File

To create a random reproducible sample of 200 questions:

```cmd
python strategyqa_export_200.py --seed 222 --force
```

This writes:

```text
data\strategyqa_200.jsonl
```

You can use a different seed if you want a different sample:

```cmd
python strategyqa_export_200.py --seed <number> --force
```

## Run One Question

Example:

```cmd
python run_once.py --question "Can penguins fly in the wild?" --ground-truth no
```

This will:

- run Debater A
- run Debater B
- run the Judge
- run Direct QA
- run Self-Consistency
- save a JSON log in `logs/`

## Run the Full Batch Experiment

If you want to rerun the full 200-question experiment, it is best to delete old logs first so they do not mix with the new run.

### 2. Generate a fresh dataset file if needed

```cmd
python strategyqa_export_200.py --seed <number> --force
```

### 3. Run the batch

```cmd
python batch_run.py --data data\strategyqa_200.jsonl --limit 200 --auto-prepare-strategyqa
```

This will:

- load the dataset file
- resume safely if partial runs already exist
- run Debate, Direct QA, and Self-Consistency for each question
- save one JSON log per completed item into `logs/`

## Run the Web UI

The Streamlit UI provides:

- question input
- yes/no ground-truth selector
- Debater A initial position
- Debater B initial position
- round-by-round debate display
- judge verdict panel
- baseline outputs

Run it with:

```cmd
streamlit run app.py
```

## Evaluate Results

Run:

```cmd
python evaluation.py
```

This prints:

- rows included
- accuracy summary
- average rounds
- round distribution
- average total LLM calls

To save the output to a text file:

```cmd
python evaluation.py > evaluation_output.txt
```

## Output Files

### Dataset file

```text
data\strategyqa_200.jsonl
```

Each line looks like:

```json
{"question": "Did the Roman Empire exist at the same time as the Mayan civilization?", "answer": "yes"}
```

### Log files

```text
logs\debate_YYYYMMDD_HHMMSS_microseconds.json
```

Each log contains:

- question
- ground truth
- initial positions
- round-by-round debate data
- judge output
- Direct QA output
- Self-Consistency outputs
- metrics