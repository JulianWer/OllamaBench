# OllamaBench 🚀: Find the Champion Among Your Local LLMs!

Juggling different local Ollama models and wondering which one is best for your tasks? **OllamaBench** takes the guesswork out of it! This tool tests your LLMs, has them evaluated by an AI, and creates a clear ranking – all fully automated and with a sleek dashboard.

## Why OllamaBench Will Be Your New Best Friend

* 🧠 **Smart Benchmarking**: Test your local Ollama models across various disciplines (Writing, Coding, Math, etc.).
* 🤖 **AI Judge**: A designated LLM of your choice objectively evaluates your models' responses.
* 🏆 **ELO Ranking**: Just like in chess! Models earn ELO points for their performance, so you see clear winners (using the cool m-ELO algorithm!).
* 📊 **Sleek Dashboard**: Track rankings and stats live in your browser – with interactive charts!
* 🛠️ **Flexibly Configurable**: Customize all important aspects in the `config/config.yaml` file to suit your needs.
* 🔍 **Transparent & Traceable**: All generated answers and results are saved for review.
* 💻 **CLI & Web Operation**: Launch tests via the command line or conveniently through the web dashboard.

## Quick Start: Get to Your LLM Battle in 5 Minutes!

1.  **Clone the Repository**:
    ```bash
    git clone <your-repo-url>
    cd ollamabench
    ```
2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Ollama Up and Running?**: Ensure [Ollama](https://ollama.com/) is running and you've pulled the models you want to test (and use as a judge):
    ```bash
    ollama pull qwen2:7b # Example
    ollama pull llama3:8b # Example for another model
    ```
5.  **Customize Your Configuration**: Open `config/config.yaml` and set your `JUDGE_MODEL` and `COMPARISON_MODELS` to the Ollama tags of your downloaded models. Also, check paths and categories.

## How the Magic Happens (Quick Overview)

1.  **Prompts In!**: OllamaBench grabs test tasks (prompts) for a specific category.
2.  **Models Get to Work!** (Optional): Your `COMPARISON_MODELS` generate answers.
3.  **The Judge Decides**: Your `JUDGE_MODEL` compares the answers pair-wise (with tricks to avoid bias!).
4.  **ELO Points Update**: The results feed into the m-ELO calculation, and `results.json` is updated.
5.  **Showtime in the Dashboard**: The latest rankings and charts are served fresh for you!

## Configuration is Key

In `config/config.yaml`, you set up all the important bits:
* `JUDGE_MODEL`: Your main AI evaluator.
* `COMPARISON_MODELS`: The models that will compete.
* `ALL_CATEGORIES` & `CURRENT_CATEGORY`: Which areas to test.
* `paths`: Where your data is stored.
* `LLM_runtime`: Your Ollama API address.

## Let's Go!

**Option 1: Via the Command Line**

```bash
python main.py
