# OllamaBench 🚀: Find the Champion Among Your Local LLMs\!

Juggling different local Ollama models and wondering which one is best for your tasks? **OllamaBench** takes the guesswork out of it\! This tool tests your LLMs, has them evaluated by an AI, and creates a clear ranking – all fully automated and with a sleek dashboard.

[](https://www.google.com/search?q=https://i.imgur.com/your-dashboard-image.png)
*(Placeholder for a dashboard screenshot)*

## ✨ Key Features

  * 🧠 **Smart Benchmarking**: Test your local Ollama models across various disciplines (Writing, Coding, Math, Reasoning, etc.).
  * 🤖 **AI Judge**: A designated, powerful LLM of your choice objectively evaluates your models' responses, minimizing human bias.
  * 🏆 **ELO Ranking**: Just like in chess\! Models earn ELO points for their performance, so you see clear winners. By default, the advanced **m-ELO algorithm** is used for more precise results.
  * 📊 **Sleek Dashboard**: Track rankings and stats live in your browser – with interactive charts and direct comparison capabilities.
  * 🛠️ **Flexibly Configurable**: Customize all important aspects in the `config/config.yaml` file to suit your needs, from the models and evaluation categories to the ELO parameters.
  * 🔍 **Transparent & Traceable**: All generated answers and evaluation results are saved for review in structured JSON files.
  * 💻 **CLI & Web Operation**: Launch tests via the command line for automation or conveniently through the web dashboard for visual control.

## ⚙️ How the Magic Happens

OllamaBench follows a simple yet effective workflow to rate your models:

1.  **Load Prompts**: The system loads test tasks (prompts) for a specific category, either from a local file or directly from Hugging Face.
2.  **Generate Answers** (Optional): Your `comparison_llms` defined in `config.yaml` generate and save answers to the prompts. This step can be skipped if answers already exist.
3.  **The Judge Decides** (Optional): The `judge_llm` compares the answers for each prompt pairwise. To avoid position bias, the answers are evaluated in two rounds in a swapped order. The judge returns a verdict (`[[A]]`, `[[B]]`, or `[[C]]` for a tie).
4.  **ELO Points Update**: The results are fed into the m-ELO calculation, and the `results.json` file is updated atomically (process-safe).
5.  **Showtime in the Dashboard**: The dashboard reads the `results.json` file and displays the latest rankings and charts via a live stream (SSE).

## 🚀 Quick Start: Get to Your LLM Battle in 5 Minutes\!

1.  **Clone the Repository**:

    ```bash
    git clone https://github.com/JulianWer/OllamaBench.git
    cd OllamaBench
    ```

2.  **Create a Virtual Environment (Recommended)**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    pip install waitress filelock # Required for the dashboard and safe file access
    ```

4.  **Ollama Up and Running?**: Ensure [Ollama](https://ollama.com/) is running and you've pulled the models you want to test (and use as a judge):

    ```bash
    ollama pull qwen2:7b        # Example for a comparison model
    ollama pull llama3:8b       # Example for another model
    ollama pull qwen2:32b-instruct # Example for a powerful judge model
    ```

5.  **Customize Your Configuration**: Open `config/config.yaml` and adjust the most important settings:

      * `judge_llm -> name`: Set the tag of your most powerful model here.
      * `comparison_llms -> names`: List all the models that should compete against each other.
      * Check `LLM_runtime -> api_base_url` to ensure the address matches your Ollama instance.

## 🕹️ Usage

You can use OllamaBench in two ways:

### 1\. Via the Web Dashboard (Recommended for getting started)

Start the web server:

```bash
python dashboard.py
```

Then open your browser to `http://127.0.0.1:5001`.

  * **Overview**: Shows the overall leaderboard, key metrics, and charts for ELO ratings by category.
  * **Detail-Analysis**: A table with the ELO scores of each model in every single category.
  * **Model Comparison**: Select two models to directly compare their performance in a bar chart.
  * **Start Evaluation**: Choose a category and start a new benchmark run directly from the browser.

### 2\. Via the Command-Line Interface (CLI)

The CLI is ideal for automation and scripting.

```bash
python main.py [ARGUMENTS]
```

**Key Arguments**:

  * `--all-categories`: Runs the benchmark for all categories defined in the configuration.
  * `--category <name>`: Runs the benchmark for a single specified category only.
  * `--generate-only`: Only generates model answers but does not run the evaluation.
  * `--judge-only`: Only runs the evaluation on existing answers without generating new ones.

**Example**:

```bash
# Only generate answers for the 'coding' category
python main.py --category coding --generate-only

# Only run the judgment for the 'coding' category
python main.py --category coding --judge-only

# Start a full run for all categories
python main.py --all-categories
```

## 🛠️ Configuration in Detail (`config/config.yaml`)

The `config/config.yaml` file is the heart of your OllamaBench setup.

```yaml
# Categories to be used for the benchmark
categories: ['schreiben','rollenspiel','begründung', 'extraktion','stamm','geisteswissenschaften']

# Number of pairwise comparisons per prompt
comparison:
  comparisons_per_prompt: 40

# --- Model Configuration ---
comparison_llms:
  names: ['gemma3:4b', 'qwen3:4b', 'llama3.1:8b']
  has_reasoning: true
  generation_options:
    temperature: 0.0

judge_llm:
  name: 'qwen3:30b-a3b' # A powerful model is highly recommended!
  has_reasoning: false
  # ...

# --- Dataset Configuration ---
dataset:
  name: "VAGOsolutions/MT-Bench-TrueGerman"
  columns:
    prompt: "turns"
    category: "category"
    id: "question_id"
    # ...

# --- Paths for Data Storage ---
paths:
  results_file: "./data/results.json"
  lock_file: "./data/results.lock"
  output_dir: "./data/model_responses"
  # ...

# --- Rating Method Configuration ---
used_rating_method: mElo # 'mElo' or 'elo'
mELO:
  initial_rating: 1000
  learning_rate: 100
  epochs: 300

# --- API and Dashboard Configuration ---
LLM_runtime:
  api_base_url: "http://localhost:11434"
dashboard:
  host: "127.0.0.1"
  port: 5001
```

## 📁 Project Structure

```
OllamaBench/
├── config/
│   └── config.yaml             # Main configuration file
├── data/
│   ├── categories/             # Locally stored, categorized prompts
│   │   └── *.json
│   ├── model_responses/        # Saved responses from the LLMs
│   │   └── [category]/
│   │       └── *.json
│   └── results.json            # ELO ranking results
├── evaluation/
│   └── ranking.py              # Logic for ELO and m-ELO calculations
├── models/
│   ├── llms.py                 # Class for the comparison models
│   └── judge_llm.py            # Class for the judge model
├── scripts/
│   ├── engine.py               # Main engine for the benchmark run
│   ├── generate_model_answers_for_category.py # Answer generation
│   └── run_comparison.py       # Execution of pairwise comparisons
├── templates/
│   └── index.html              # Frontend template for the dashboard
├── utils/
│   ├── config.py               # Loads and validates the configuration
│   ├── file_operations.py      # Ensures safe reading and writing of files
│   └── get_data.py             # Loads datasets from local or Hugging Face sources
├── dashboard.py                # Flask web server for the dashboard
├── main.py                     # Command-Line Interface (CLI)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```
