# OllamaBench: Local LLM Benchmarking & ELO Ranking

OllamaBench is a Python-based tool designed to benchmark and rank local language models running via Ollama. It automates the process of generating model responses across various prompt categories, evaluating these responses using a designated "judge" LLM, and then calculating ELO ratings to rank model performance. Results are visualized in a user-friendly web dashboard.

## Features

* **Local LLM Benchmarking:** Focuses on models accessible through a local Ollama instance.
* **Categorized Prompts:** Supports different prompt categories (e.g., Writing, Coding, Math, Roleplay) for nuanced evaluation.
* **LLM-based Judging:** Uses a specified Ollama model as an objective judge to compare pairs of model responses.
* **Position Bias Mitigation:** The judge evaluates pairs in both orders (A vs B, B vs A) to reduce position bias.
* **ELO Rating System:** Implements an m-ELO (Maximum Likelihood Estimation ELO) based system to rank models within each category and overall.
* **Web Dashboard:** Provides a Flask-based web interface with Tailwind CSS and Chart.js for visualizing:
    * Overall model rankings.
    * ELO ratings per category (bar chart).
    * Comparative model performance across categories (radar chart).
    * Detailed ELO scores and win/loss/draw statistics.
* **Configurable:** Most aspects are configurable via a central `config/config.yaml` file.
* **Data Management:**
    * Loads prompts from local JSON files or Hugging Face datasets.
    * Saves generated model answers for review and re-use.
    * Persists ELO ratings and comparison statistics.
* **CLI & Web Operation:**
    * Run evaluations via a command-line interface (`main.py`).
    * View results and trigger evaluations via the web dashboard (`dashboard.py`).

## How it Works (Workflow)

1.  **Configuration:** Settings are loaded from `config/config.yaml` (models, categories, paths, etc.).
2.  **Prompt Loading:** Prompts for a selected category are loaded either from local JSON files (e.g., `data/categories/schreiben_prompts.json`) or downloaded from a specified Hugging Face dataset.
3.  **Answer Generation (Optional if `RUN_ONLY_JUDGEMENT` is true):**
    * The `generate_model_answers_for_category.py` script iterates through each `COMPARISON_MODEL` defined in the config.
    * For each model, it generates responses to all prompts in the current category.
    * Responses are saved to `model_responses/CATEGORY_NAME/PROMPT_ID.json`. Each file contains a list of responses from different models for that specific prompt.
4.  **Comparison & Judging:**
    * The `run_comparison.py` script (triggered by `main.py` or the dashboard) processes prompts for a category.
    * For each prompt, it loads the previously generated answers from `model_responses/`.
    * It creates pairs of model answers (e.g., Model A's answer vs. Model B's answer).
    * The `JudgeLLM` (defined in `models/judge_llm.py` and configured via `JUDGE_MODEL` in `config.yaml`) is tasked to compare these two answers.
    * To mitigate position bias, the judge evaluates twice: (Answer A, Answer B) and (Answer B, Answer A). A consistent verdict is sought.
    * The outcome (Model A wins, Model B wins, or Tie) is recorded.
5.  **ELO Calculation & Ranking:**
    * The `evaluation/ranking.py` script uses the collected match outcomes.
    * It employs an m-ELO (Maximum Likelihood Estimation ELO) algorithm to update the ELO ratings of the compared models within the specific category.
    * Updated ratings and match statistics (wins, losses, draws) are saved to `data/results.json`.
6.  **Dashboard Visualization:**
    * The Flask application (`dashboard.py`) serves `templates/index.html`.
    * The dashboard fetches data from `/api/results` (which reads `data/results.json`) and uses Server-Sent Events (`/stream`) to update live.
    * Charts (bar, radar) and tables display the model rankings and ELO scores.
    * The dashboard also provides a button to trigger the evaluation process.

## Prerequisites

* **Python 3.8+**
* **Ollama:** Installed and running. You can get it from [ollama.com](https://ollama.com/).
* **Ollama Models:** You need to have pulled the models you intend to benchmark and the judge model using Ollama. For example:
    ```bash
    ollama pull qwen2:7b
    ollama pull llama3:8b
    # etc. for all models in COMPARISON_MODELS and JUDGE_MODEL
    ```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd ollamabench
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    The key dependencies are:
    * `Flask`
    * `PyYAML`
    * `requests`
    * `datasets` (from Hugging Face)
    * `filelock`
    * `waitress` (for serving the Flask app)

## Configuration

The main configuration is done in `config/config.yaml`. Key settings to review and modify:

* **`JUDGE_MODEL`**: The Ollama model tag used for judging comparisons (e.g., `qwen2:7b-instruct-q4_0`). **Crucial for evaluation quality.**
* **`COMPARISON_MODELS`**: A list of Ollama model tags to be benchmarked (e.g., `['llama3:8b', 'mistral:7b']`).
* **`RUN_ALL_CATEGORIES_IS_ENABLED`**: If `true`, runs all categories in `ALL_CATEGORIES`. If `false`, runs only `CURRENT_CATEGORY`.
* **`RUN_ONLY_JUDGEMENT`**: If `true`, skips answer generation and only runs comparisons (assumes answers exist in `model_responses/`).
* **`ALL_CATEGORIES`**: List of categories to process if `RUN_ALL_CATEGORIES_IS_ENABLED` is true. Category names must match the base names of prompt files in `data/categories/` (e.g., `schreiben` for `schreiben_prompts.json`).
* **`PROMPT_DATASET`**: Default Hugging Face dataset to download if local category files are not found (e.g., `"VAGOsolutions/MT-Bench-TrueGerman"`).
* **`paths`**: Defines locations for data, results, lock files, and model responses.
* **`elo` & `mELO`**: Parameters for the ELO rating system (initial rating, K-factor, learning rate, epochs).
* **`LLM_runtime`**: Ollama API settings (base URL, timeout, retries).
* **`dashboard`**: Host, port, and debug settings for the web dashboard.

**Important:**
* Ensure all models listed in `JUDGE_MODEL` and `COMPARISON_MODELS` are downloaded via `ollama pull <model_tag>`.
* The quality of the `JUDGE_MODEL` significantly impacts the benchmark results. Choose a capable model.

## Usage

### 1. Running Evaluations via CLI

You can run the full evaluation pipeline (or parts of it based on config) using `main.py`:

```bash
python main.py
