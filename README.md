# OllamaBench: Comprehensive Benchmarking Framework for Local LLMs

OllamaBench is a robust and extensible framework for performance evaluation of local language models running via Ollama. It provides an automated pipeline for rigorous testing, AI-based evaluation, and detailed analysis through an interactive dashboard and command-line interface (CLI).
The current Rakings are:
```
--- OllamaBench ELO Rankings ---
--------------------------------------------------------------------------------
Rank  Model                                            Avg ELO           W/L/D
--------------------------------------------------------------------------------
1     qwen3:8b                                          1248.3      181/10/103
2     gemma3n:e4b                                       1229.0      158/22/112
3     qwen3:4b                                          1209.5      153/14/124
4     gemma3:4b                                         1153.1      130/34/115
5     qwen3:1.7b                                        1110.0      117/40/130
6     Ministral-8B-Instruct-2410-Q4_K_M:latest          1008.6       79/74/143
7     gemma2:2b                                          997.1       59/71/134
8     gemma3:1b                                          992.8       72/89/124
9     llama3.1:8b                                        956.1       51/78/130
10    mistral:7b                                         929.8      51/107/129
11    qwen3:0.6b                                         909.6       27/69/185
12    llama3.2:3b                                        883.2      37/119/120
13    llama3.2:1b                                        803.3      14/146/124
14    qwen:7b-chat                                       788.9      14/137/101
15    gemma:2b-instruct                                  780.6       7/140/124
--------------------------------------------------------------------------------
```

## Core Features

  * **Modular Benchmarking**: Conduct evaluations across a wide range of disciplines such as `Writing`, `Coding`, `Reasoning`, and `Knowledge Extraction`.
  * **AI-Powered Adjudication**: Utilize a powerful "judge" LLM of your choice to provide objective, nuanced, and reproducible evaluations of model responses.
  * **Advanced ELO Ranking**: Model performance is rated using the m-ELO algorithm, a variant of the standard ELO system designed for faster convergence and higher accuracy in pairwise comparisons.
  * **Interactive Analysis Dashboard**: A web-based dashboard built with Flask and Chart.js offers live insights into results, including:
      * **Overall Leaderboards** and aggregate metrics.
      * **Detailed Categorical ELO Scores** for granular analysis.
      * **Direct Model-to-Model Comparison** across different task areas.
      * **Live Updates** of results via Server-Sent Events (SSE).
  * **Extensive Configurability**: Customize nearly every aspect of the benchmarking process via a central `config.yaml` file, including model parameters, datasets, evaluation categories, and ELO settings.
  * **Automation-Friendly CLI**: A robust command-line interface allows for scripting and automation of benchmarking workflows, with separate controls for the response generation and evaluation phases.
  * **Transparency and Traceability**: All generated responses and judge verdicts are archived in structured JSON files for detailed review and post-hoc analysis.

## System Architecture & Workflow

OllamaBench is designed to be process-safe and atomic to ensure data integrity during benchmark runs.

1.  **Configuration & Prompt Loading**: The system is initialized via the `config.yaml` file. Prompts for a specific category are loaded either from local JSON files or directly from a Hugging Face dataset.
2.  **Answer Generation (Optional)**: The models defined as `comparison_llms` generate answers to the loaded prompts. Existing answers are cached and not regenerated to improve efficiency.
3.  **Pairwise Adjudication (Optional)**: The `judge_llm` performs a pairwise evaluation of responses. To minimize positional bias, responses are evaluated in two rounds in a swapped order. The judge outputs a verdict (`[[A]]`, `[[B]]`, or `[[C]]` for a tie).
4.  **ELO Calculation**: The adjudication results are fed into the m-ELO algorithm to update the models' ELO ratings. The `results.json` file is updated atomically using file locking to prevent race conditions.
5.  **Visualization**: The Flask-based dashboard reads the `results.json` file and streams the latest rankings and charts to the client via SSE.

## Getting Started

### Prerequisites

  * Python 3.8+ and `pip`
  * A running [Ollama](https://ollama.com/) instance
  * The `git` command line

### Installation & Configuration

1.  **Clone the Repository**:

    ```bash
    git clone https://github.com/JulianWer/OllamaBench.git
    cd OllamaBench
    ```

2.  **Set up a Virtual Environment (Recommended)**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Pull Ollama Models**: Ensure Ollama is running and pull the models you wish to evaluate and use as a judge.

    ```bash
    ollama pull llama3:8b
    ollama pull qwen2:7b
    ollama pull qwen2:32b-instruct  # Recommended as a powerful judge
    ```

5.  **Customize Configuration**: Edit `config/config.yaml` to define your test run.

      * `judge_llm -> name`: Specify the model tag for your judge LLM.
      * `comparison_llms -> names`: List all models that will compete.
      * `LLM_runtime -> api_base_url`: Verify this matches your Ollama API URL.
      * `categories`: Select the categories you wish to test in.

## Usage

### Web Dashboard (Recommended Method)

For interactive analysis and control, start the server:

```bash
python dashboard.py
```

Navigate to `http://127.0.0.1:5001` in your browser.

### Command-Line Interface (For Automation)

Run benchmarks via the CLI:

```bash
python main.py [ARGUMENTS]
```

**Key CLI Arguments**:

  * `--show-rankings`: Displays the current ELO ranking table and exits.
  * `--all-categories`: Runs the benchmark for all configured categories.
  * `--category <name>`: Runs the benchmark for a single specified category only.
  * `--generate-only`: Only performs the answer generation step.
  * `--judge-only`: Only performs the adjudication step on existing answers.

**Example Workflow**:

```bash
# Generate answers for the 'coding' category only
python main.py --category coding --generate-only

# Subsequently, run the adjudication for the same category
python main.py --category coding --judge-only

# Run a full end-to-end benchmark for all categories
python main.py --all-categories
```
