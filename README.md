# LLM Bias Analysis Tool

A Streamlit web application for testing whether large language models (LLMs) exhibit systematic explanatory bias when answering questions about international economic law and geopolitics. The tool sends a set of questions to multiple LLMs, uses a separate "judge" LLM to classify each response by the explanatory framework it uses, and then visualizes the results across models.

---

## Project Background

This project was built for a research proposal by Professors Gregory Shaffer and Sergio Puig titled *"What Type of International Order Is Promoted by AI? Investigating Biases in Large Language Models through International Economic Law."*

The core concern is that LLMs present contested interpretations as settled facts. When asked why President Trump expressed interest in Greenland, for example, ChatGPT framed geopolitical competition as the definitive explanation without acknowledging alternatives such as personalist or rent-seeking motivations. At scale, this systematic privileging of geopolitical frameworks over sociological ones could have real "world-constructing" effects.

This codebase is a tool built to empirically investigate that bias across domains of international economic law.

---

## How This Project Addresses the Research Requirements

The professors' proposal identified four specific research objectives. Here is how each is addressed in the code:

**1. Explanatory Bias — Whether LLMs systematically favor geopolitical explanations over sociological or elite-network-based ones**

The judge LLM is given a structured coding rubric (defined in `llm_judge.py`) that classifies every response into one of three primary frameworks: *Geopolitical* (state-level national security, great-power competition, geoeconomics), *Sociological* (elite networks, rent-seeking, personalist power), or *Economic Protectionism* (trade barriers, producer interests). A secondary framework field captures cases where a response blends two approaches. The Analysis Dashboard aggregates these classifications per model in bar charts, making systematic tendencies visible at a glance.

**2. Framing Effects — How LLMs present contested explanations (as hypotheses vs. settled facts)**

The judge assigns a *Certainty Score* from 1 to 5 using a detailed linguistic rubric: a score of 1 reflects heavy hedging ("might," "could," "one possible explanation"), while a score of 5 reflects unqualified causal claims ("causes," "is clearly due to," "demonstrates"). The rubric instructs the judge to weight hedges and boosters by their prominence and frequency rather than simply averaging them. The judge also extracts verbatim *certainty strings* — 2–5 short phrases from the response that exemplify the epistemic register. These strings are highlighted in yellow in the detailed response view, giving a reviewer direct textual evidence for each score.

**3. Presence or Absence of Alternative Hypotheses — Whether elite-network explanations are acknowledged**

A dedicated boolean field, *Elite Networks Mentioned*, is extracted from every judge evaluation. It is `True` only when a response explicitly references elite networks, rent-seeking, or personalist power dynamics. The dashboard displays this dimension in its own bar chart and filter, making it straightforward to identify which models are most likely to surface (or omit) the sociological alternative hypothesis.

**4. Comparative Empirical Analysis Across Multiple LLMs**

The prompt ingestion system (`prompt_ingestion.py`) queries multiple LLMs concurrently using standardized prompts, with per-model rate limiting to avoid throttling. Results are stored in a flat CSV keyed by (model, prompt), so responses from different providers can be compared on equal footing. The Model Comparison sub-tab in the dashboard allows side-by-side inspection of up to three models' answers to the same question, along with their judge-assigned framework and certainty scores.

---

## What It Does

The tool works in three stages, each corresponding to a tab in the app:

1. **Prompt Ingestion** — Upload a CSV of questions and run them against multiple LLMs simultaneously. Responses are saved to `model_responses.csv`.
2. **LLM Judge** — A judge LLM reads each response and classifies it: which explanatory framework does it use (Geopolitical, Sociological, or Economic Protectionism)? How certain is the language? Are elite networks mentioned? Results are saved to `final_judge_responses_parsed.csv`.
3. **Analysis Dashboard** — Charts and a filterable table let you explore how different models differ in their explanatory tendencies, certainty levels, and secondary frameworks. A side-by-side model comparison view is also available.

The fourth tab lets you browse available models from OpenRouter and configure which models are available as test cases for the prompt ingestion tool.

---

## Installation

### Prerequisites

- Python 3.11 or later
- A virtual environment (recommended)

### Setup

```bash
# Clone or copy the project files, then from the project directory:
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Option 1 — Hosted App (Recommended)

The easiest way to use this tool is through the hosted Streamlit deployment, which requires no installation and has all API keys pre-configured:

**https://final-computer-science-project-egvthvrmrwzvhzazihhappo.streamlit.app**

Open the link in any browser and proceed directly to the usage walkthrough below. No account, API key, or local setup is needed. 

> **Note:** These API keys are my personal keys. OpenRouter has been preloaded with $10 so it should be enough for the research scope of this project. Tinker is still using my student allowance. 

> **Note:** The hosted app syncs data to a GitHub backend between sessions, so results from a previous run (model responses, judge evaluations) will be available when you return.

---

## Option 2 — Run Locally with Streamlit

Use this option if you want to run the app on your own machine with your own API keys, or if you need to work with local data files directly.

### Prerequisites

- Python 3.11 or later
- A virtual environment (recommended)

### Setup

```bash
# From the project directory:
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### API Keys

Create a `.env` file in the project root with keys for whichever providers you intend to use:

```
OPENROUTER_API_KEY=your_openrouter_key_here
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here
TINKER_API_KEY=your_tinker_key_here
```

- **OpenRouter** — https://openrouter.ai/keys — broad model catalog via a single endpoint
- **Groq** — https://console.groq.com/keys — fast inference on open-source models
- **Gemini** — https://aistudio.google.com/app/apikey — Google's Gemini models
- **Tinker** — API key provided by Thinking Machines; required only if using the Fine-Tuned Judge, Base Llama, or Qwen3 Tinker judge options

You only need keys for the providers whose models you plan to use. The app will not error at startup for missing keys — it will only fail if you actually try to call a model from that provider. But, an OpenRouter key and a Tinker key are highly recommended as those are the main services for testing models and calling judges. 

### GitHub Sync (optional)

I have a GitHub Sync setup to push and pull files to a separate Streamlit Dashboard production branch. This can also be used if you want to push and pull files into your own github. To persist data files across sessions to a GitHub repository, add these to your `.env`:

```
GITHUB_TOKEN=your_personal_access_token
GITHUB_REPO=owner/repo-name
GITHUB_BRANCH=streamlit-deployment   # defaults to this value if omitted
```

If these are not set, the app works entirely with local files and sync is silently skipped.

### Launch

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## Usage Walkthrough

This walkthrough applies whether you are using the hosted app or a local instance.

### Prepare Your Questions CSV

Create a CSV file with exactly these two columns (case-sensitive):

| Prompt | Topic |
|---|---|
| Why did the United States impose tariffs on Chinese goods? | Trade Policy |
| What explains the rise of economic nationalism in Europe? | Geopolitics |

Any other columns are ignored. A sample file (`geopolitics_questions_final_1.csv`) is included in the project and can be used as a template.

### Step 1 — Model Config (Tab 4)

This tab is where you select models for availability in Tab 1 to run against the prompts held in the uploaded csv. 

Browse the full OpenRouter model catalog, filter by provider and release date, and save your selection to `models_config.json`. Non-OpenRouter entries (Groq, Google) already in the config are always preserved. 


### Step 2 — Prompt Ingestion (Tab 1)

1. Upload your CSV using the file uploader.
2. A preview of the first ten rows is shown. The app checks that the required columns are present.
3. A table of available models appears with checkboxes. Deselect any you don't want to use.
4. Click **Run Prompt Ingestion**. The app queries each selected model for each prompt concurrently, respecting per-model rate limits. This may take several minutes for large question sets.
5. Results are appended to `model_responses.csv`. Previously completed (model, prompt) pairs are skipped automatically, and any rows that previously returned an error are retried.
6. A second table below allows you to run additional models against the existing question set without re-uploading the CSV.

### Step 3 — LLM Judge (Tab 2)

1. The tab shows all responses in `model_responses.csv`. Responses that are empty, start with `ERROR`, or are fewer than 30 words are excluded from judging and listed in a collapsible section.
2. Select a judge model from the dropdown. Available options include:
   - **Fine-Tuned Judge (Tinker)** — a model fine-tuned specifically for this classification task (requires Tinker API access)
   - **Llama 3.1 8B Base / Qwen3 30B** via Tinker (also requires Tinker access) (These are not fine-tuned models)
   - **Groq models** — Llama 3.1 8B and Llama 3.3 70B
   - **OpenRouter models** — Qwen3 80B
3. Click **Run Judge**. The judge returns a JSON object with the primary framework, secondary framework, certainty score (1–5), whether elite networks were mentioned, and verbatim evidence and framework strings. The rubric used to instruct the judge is defined in `grading_prompt.txt` — edit that file to change the classification criteria without touching any code.
4. Already-judged (model, prompt, judge) triples are skipped on re-runs, so you can switch judges and accumulate results from multiple judges without overwriting previous work.
5. Results are parsed and saved to `final_judge_responses_parsed.csv`. A preview table and download button appear below.

### Step 4 — Analysis Dashboard (Tab 3)

The dashboard has two sub-tabs:

**Analysis** — Filter by judge model, response model, primary framework, elite networks flag, certainty score range, and a keyword search on evidence spans. Four bar charts update automatically:
- Average certainty score per model
- Primary framework count per model
- Secondary framework count per model
- Elite networks mentions per model

Click any row in the filtered results table to open a detailed view showing the original prompt, the model's answer, and the judge's analysis with color-highlighted evidence spans (yellow = certainty markers, blue = framework markers). The markers justify why the model gave a certain certainty score or picked a framework. 

**Model Comparison** — Select a question and up to three models to see their answers and judge classifications side by side.

---

## Option 3 — Command Line Interface

The two core scripts can be run directly from the terminal without Streamlit. This is useful for large batches where you want to leave a job running in the background, or to run ingestion and judging as separate steps.

The same `.env` setup from Option 2 is required. All output files are written to the project directory.

### Managing CLI Models

The CLI uses a separate config file (`cli_models_config.json`) so it stays independent from the Streamlit app's `models_config.json`. Use `models_config_cli.py` to manage it:

```bash
# Add a model by its OpenRouter ID
python models_config_cli.py add qwen/qwen3-30b:free

# List all models currently in the CLI config
python models_config_cli.py list

# Remove a model by its display name
python models_config_cli.py remove "Qwen 3 30B"
```

When adding a model, the script fetches its details from OpenRouter (name, context length, knowledge cutoff, pricing), displays them, and prompts you to confirm before writing to `cli_models_config.json`. You can accept the API's suggested display name by pressing Enter, or type a custom name.

### Prompt Ingestion

```bash
# Run all configured models against a CSV of questions
python prompt_ingestion.py questions.csv

# Run only specific models (names must match cli_models_config.json)
python prompt_ingestion.py questions.csv --models "Groq Llama 3.1 8B" "Gemini 2.5 Flash"

# See all available model names
python prompt_ingestion.py --list-models

# Use a custom config file
python prompt_ingestion.py questions.csv --config my_models.json
```

The script asks for confirmation before making any API calls, showing the total number of models, prompts, and responses it will generate. Results are appended to `model_responses.csv`.

### LLM Judge

```bash
# Run the judge using the default model (set in llm_judge.py)
python llm_judge.py

# Run with a specific judge model
python llm_judge.py --judge llama-3.3-70b
python llm_judge.py --judge llama-3.1-8b

# See all available judge model names
python llm_judge.py --list-judges
```

The script reads from `model_responses.csv` and writes raw judge output to `final_judge_responses.csv`. After it completes, run the parser to produce the structured output used by the dashboard:

```bash
python data_cleaning.py
```

This writes `final_judge_responses_parsed.csv`, which can be opened directly in the Streamlit dashboard.

---

## File Reference

| File | Purpose |
|---|---|
| `app.py` | Streamlit app entry point |
| `prompt_ingestion.py` | Queries LLMs with prompts; writes `model_responses.csv` |
| `llm_judge.py` | Runs a judge LLM on responses; writes `final_judge_responses.csv` |
| `data_cleaning.py` | Parses judge JSON output; writes `final_judge_responses_parsed.csv` |
| `data_viz.py` | Renders the analysis dashboard inside the app |
| `github_storage.py` | Optional GitHub sync for data files |
| `model_selection.py` | Fetches and filters OpenRouter model catalog |
| `models_config_cli.py` | CLI tool for managing `cli_models_config.json` |
| `models_config.json` | Model list used by the Streamlit app |
| `cli_models_config.json` | Model list used by the CLI scripts |
| `grading_prompt.txt` | Judge rubric and output format instructions — edit to change how responses are classified |
| `model_responses.csv` | Output: raw LLM responses (not version controlled) |
| `final_judge_responses.csv` | Output: raw judge evaluations (not version controlled) |
| `final_judge_responses_parsed.csv` | Output: parsed and structured judge results (not version controlled) |
| `current_prompts.csv` | Most recently uploaded prompts CSV, persisted via GitHub sync (not version controlled) |

---

## Common Issues

**Rate limit errors during ingestion** — Each model in `models_config.json` has a `requests_per_minute` field. If you are hitting limits frequently, lower these values. Free-tier OpenRouter models are especially constrained; the app uses exponential backoff but very aggressive concurrency can still exhaust limits.

**Judge returns unparseable JSON** — Some models occasionally return text outside the JSON object despite the prompt. The parser tries several fallback strategies (regex extraction, `ast.literal_eval`). Rows that cannot be parsed will have `null` values in the structured columns but will still appear in the results table.

**Tinker judge models** — The Fine-Tuned Judge, Base Llama, and Qwen3 Tinker options require access to Tinker inference service.

**GitHub sync fails silently** — If `GITHUB_TOKEN` or `GITHUB_REPO` are not set, sync is skipped without any error. If they are set but incorrect, you will see a warning in the Streamlit UI with the HTTP error details.

**CSV files are not version controlled** — All `.csv` and `.jsonl` files are listed in `.gitignore` and will not appear in `git status` or be committed. Data persistence between sessions is handled entirely by the GitHub sync feature in `github_storage.py`. If you are running locally without GitHub sync configured, output files are written to disk and persist until you delete them manually.

---

## Reflection on Broader Themes

*[Placeholder — to be completed per assignment requirement 2: reflection on whether programming is a valuable skill for lawyers, what types of projects lawyers might find useful, and advice about the future of computer programming and the law.]*

---

## Scope and Effort






*[Placeholder — to be completed per assignment requirement 3: description of the scope of the project, anything non-evident about the time and effort involved, and any design decisions worth highlighting.]*
