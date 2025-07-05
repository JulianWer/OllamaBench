# OllamaBench: A Comprehensive Benchmarking Framework for Local LLMs

OllamaBench is a robust and extensible framework for the performance evaluation of local language models running via Ollama. This project was developed as part of a **bachelor thesis** and provides an automated pipeline for rigorous testing, AI-based evaluation, and detailed analysis through an interactive dashboard and command-line interface (CLI).

## Latest Rankings

The current rankings based on the latest benchmark run are:

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

*Ranking data last updated: 2025-06-30*

## License

See the [LICENSE](https://github.com/JulianWer/OllamaBench/blob/main/LICENSE.md) file for license rights and limitations (MIT).



## Benchmark Analysis & Validity

The credibility of the benchmark results is critical. The project includes an `analyse_results.ipynb` notebook for documentation and validity checks.

### Comparison with Public Leaderboards

The notebook includes a comparative analysis between this project's results (`Meine LLM Arena`) and the public `LMArena.ai` leaderboard. The plot from the notebook visually confirms two main points:

1.  **Strong Correlation**: The relative ranking of models is highly consistent between both arenas.
2.  **Systematic ELO Difference**: `LMArena.ai` provides significantly higher absolute ELO ratings, which is likely due to differences in voter pools, prompt sets, or the underlying ELO calculation methodology.

This indicates that while the absolute numbers differ, **OllamaBench produces a relative ranking of models that is in good agreement with larger, public benchmarks.**

### Internal Consistency

The leaderboard data demonstrates strong internal consistency, supported by two key observations from the analysis notebook:

1.  **Logical Performance Scaling**: Performance (ELO) scales logically with model size within the same family (e.g., `Qwen3: 8b > 4b > 1.7b > 0.6b`).
2.  **Correlation of ELO and W/L/D Record**: The ELO rankings are strongly supported by the win/loss/draw statistics. Models with higher ELO scores have a significantly better win-to-loss ratio.

These factors indicate that the evaluation methodology is applied consistently and that the rankings are a credible representation of the models' relative performance.

## System Architecture & Workflow

1.  **Configuration & Prompt Loading**: The system is initialized via `config.yaml`. Prompts for a specific category are loaded from local JSON files (e.g., `begründung_prompts.json`) or a Hugging Face dataset.
2.  **Answer Generation (Optional)**: Models defined in `comparison_llms` generate answers to the prompts.
3.  **Pairwise Adjudication (Optional)**: The `judge_llm` performs a pairwise evaluation of responses. Positional bias is minimized by evaluating in two rounds with swapped positions.
4.  **ELO Calculation**: The adjudication results are processed by the m-ELO algorithm to update model ratings. The `results.json` file is updated atomically using file locking.
5.  **Visualization**: The Flask-based dashboard reads the `results.json` file and streams the latest rankings to the client.

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
3.  **Install Dependencies**: The required packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
4.  **Pull Ollama Models**: Ensure Ollama is running and pull the models you wish to evaluate and use as a judge.
    ```bash
    ollama pull llama3.1:8b
    ollama pull gemma3:4b
    ollama pull qwen3:30b-a3b # Recommended as a powerful judge
    ```
5.  **Customize Configuration**: Edit `config/config.yaml` to define your test run.
      * `judge_llm -> name`: Specify the model tag for your judge LLM.
      * `comparison_llms -> names`: List all models that will compete.
      * `LLM_runtime -> api_base_url`: Verify this matches your Ollama API URL.
      * `categories`: Select the categories you wish to test.

## Usage

### Web Dashboard (Recommended)

For interactive analysis and control, start the server:

```bash
python dashboard.py
```

Navigate to `http://127.0.0.1:5001` in your browser.

### Command-Line Interface (For Automation)

Run benchmarks via the CLI using `main.py`.

**Key CLI Arguments**:

  * `--show-rankings`: Displays the current ELO ranking table and exits.
  * `--all-categories`: Runs the benchmark for all configured categories.
  * `--category <name>`: Runs the benchmark for a single specified category.
  * `--generate-only`: Only performs the answer generation step.
  * `--judge-only`: Only performs the adjudication step on existing answers.

**Example Workflow**:

```bash
# Creates a table for the current rankings
python main.py --show-rankings
# Generate answers for the 'coding' category only
python main.py --category schreiben --generate-only

# Subsequently, run the adjudication for the same category
python main.py --category schreiben --judge-only

# Run a full end-to-end benchmark for one specific category 
python main.py --category schreiben

# Run a full end-to-end benchmark for all categories
python main.py --all-categories

# Run a full end-to-end benchmark for all categories
python main.py
```


## Future Work

  * Add also remote Models
  * Reduce self-bias by using two Judge-Models

