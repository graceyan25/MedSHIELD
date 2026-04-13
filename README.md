# Sentence Relevance Evaluation Toolkit

A toolkit for evaluating GPT models on multiple-choice QA datasets with sentence-level relevance labeling. Available as a **web app** or a **command-line tool**.

Given a CSV dataset, the toolkit will:
1. Query a GPT model to answer each question and label each sentence as High Relevance, Low Relevance, or Irrelevant
2. Save results incrementally with resume support
3. Compute accuracy, refusal rate, relevance distribution, and relevance-accuracy breakdown

---

## Input CSV Format

Your CSV needs at minimum a sentences column and a question+options column. All column names are configured at runtime — no renaming required.

| Column | Required | Description |
|---|---|---|
| Sentences | Yes | Pre-formatted sentence text, e.g. `1. ... 2. ...` or `S1: ... S2: ...` — sentence count is detected automatically |
| Question + options | Yes | The full question and answer choices sent to the model |
| Correct answer | No | Ground-truth answer letter (A, B, C, ...) — needed for accuracy metrics |
| ID | No | Unique identifier per row, used to resume interrupted runs |
| Group-by | No | A category column (e.g. `data_source`) used to break down stats per group |

---

## Option 1 — Web App

### Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`.

### Usage

The app has three tabs:

**1 · Configure**
- Upload your input CSV
- Assign each column role using the dropdowns
- Set your model name, temperature, and max sentences to label
- Enter your OpenAI API key (stored in memory only, never saved to disk)
- Optionally paste a custom prompt template
- Click **Save configuration**

**2 · Query**
- Review the active configuration
- Click **Run Query** to send each row to the model
- A progress bar tracks processing row by row
- If the run is interrupted, re-running will skip already-processed rows (resume support)
- Download the results CSV when complete

**3 · Analyze**
- Uses the output from the Query step automatically, or upload a previous results CSV
- Click **Run Analysis** to compute metrics
- Results are shown inline: accuracy, refusal rate, relevance distribution, and relevance-accuracy breakdown
- Download `summary.csv` and `analysis_report.txt`

### Sharing

To share the app with others, deploy it to [Streamlit Community Cloud](https://share.streamlit.io) (free):

1. Push `app.py`, `toolkit/`, `requirements.txt`, and `.gitignore` to a GitHub repo
2. Connect the repo at share.streamlit.io and set the main file to `app.py`
3. Share the generated URL — each user enters their own OpenAI API key

---

## Option 2 — Command Line

### Setup

Install dependencies:

```bash
pip install openai pandas numpy
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...
```

### Step 1 — Configure

Run the interactive setup to create a config file:

```bash
python -m toolkit setup --config my_experiment.json
```

This reads your CSV, shows available column names, and saves your choices to `my_experiment.json`. You only need to run this once per dataset.

**Example session:**

```
=== Toolkit Setup ===

Input CSV path: data/my_dataset.csv
Output CSV path: results/output.csv

-- Dataset columns --
  Columns: ['id', 'sentences', 'question_options', 'answer', 'source']

Unique ID column (for resume support) (Enter to skip): id
Pre-formatted sentences column: sentences
Question + answer options column: question_options
Group-by column for per-group analysis (Enter to skip): source

Correct answer column (for accuracy analysis) (Enter to skip): answer

-- Model --
Model name [gpt-4o]:
Temperature [0]:
Max sentences to label [30]:
Custom prompt template file path (optional) (Enter to skip):

Config saved to: my_experiment.json
```

### Step 2 — Query

```bash
python -m toolkit query --config my_experiment.json
```

If interrupted, re-running the same command resumes from where it left off.

To override input/output paths without editing the config:

```bash
python -m toolkit query --config my_experiment.json --input other_input.csv --output other_output.csv
```

**Output CSV columns added:**

| Column | Description |
|---|---|
| `Raw_Response` | Full model response text |
| `LLM_answer` | Extracted answer letter |
| `label_1` … `label_N` | Sentence relevance labels (`High Relevance`, `Low Relevance`, `Irrelevant`) |

### Step 3 — Analyze

```bash
python -m toolkit analyze --config my_experiment.json --output-dir results/analysis
```

Writes two files to the output directory:

- **`analysis_report.txt`** — human-readable stats report
- **`summary.csv`** — one row per input row with correctness flags, majority relevance label, and per-label percentages

**Metrics computed:**

| Metric | Description |
|---|---|
| Accuracy | % of rows where model answer matches correct answer |
| Refusal rate | % of rows where model did not return a valid answer letter |
| Relevance distribution | Mean % of sentences labeled High / Low / Irrelevant per row, with 95% CI |
| Relevance-accuracy | Accuracy broken down by each row's majority relevance label |

All metrics are reported overall and per group if a group-by column was configured.

### All-in-one

Run setup → query → analyze in a single command:

```bash
python -m toolkit run --config my_experiment.json --output-dir results/analysis
```

---

## Custom Prompts

To override the default prompt, create a `.txt` file using `{sentences}` and `{options}` as placeholders:

```
Given the following question and sentences, select the best answer and label each sentence.

Question and Options:
{options}

Sentences:
{sentences}

Return only a JSON object: {"Answer": "<letter>", "Sentence_Relevance": ["High Relevance", ...]}
```

In the web app, paste the template directly into the custom prompt field. On the CLI, provide the file path during setup or set `"custom_prompt_path"` in the config JSON.

---

## Config JSON Reference

The config file can be edited directly. All fields:

```json
{
  "input_path": "",
  "output_path": "",
  "id_col": null,
  "sentences_col": "",
  "options_col": "",
  "groupby_col": null,
  "correct_answer_col": null,
  "provider": "openai",
  "model_name": "gpt-4o",
  "temperature": 1.0,
  "api_key_env": "OPENAI_API_KEY",
  "custom_prompt_path": null,
  "max_sentences": 30
}
```
