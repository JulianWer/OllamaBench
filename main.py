import argparse
import logging
import logging.handlers
import sys
import os
from typing import Dict, Any, List

import yaml

from scripts.engine import run_benchmark
from utils.config import load_config, DEFAULT_CONFIG_PATH
from utils.file_operations import load_json_file

CONFIG: Dict[str, Any] = {}

# --- Configuration Loading ---
CONFIG = None
try:
    CONFIG = load_config(DEFAULT_CONFIG_PATH)
    ALL_CATEGORIES = CONFIG.get("categories")

except (FileNotFoundError, KeyError, ValueError, Exception) as e:
    logging.basicConfig(level=logging.ERROR)
    logging.critical(f"FATAL: Failed to load or validate configuration from '{DEFAULT_CONFIG_PATH}': {e}", exc_info=True)
    exit(1)

def setup_logging(config: Dict[str, Any]):
    """Configures logging based on the loaded configuration."""
    logging_config = config.get("logging", {})
    
    log_level_str = logging_config.get("level", "INFO").upper()
    log_format_str = logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file_path = logging_config.get("log_file") 

    root_logger = logging.getLogger()
    log_level = getattr(logging, log_level_str, logging.INFO)
    root_logger.setLevel(log_level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(log_format_str)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file_path:
        try:
            log_dir = os.path.dirname(log_file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path, maxBytes=5*1024*1024, backupCount=2, mode='a', encoding='utf-8'
            )
            file_handler.setLevel(logging.WARNING) 
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            root_logger.info(f"Logging in die Konsole ist aktiviert. Warnings (und höher) werden in '{log_file_path}' gespeichert.")

        except Exception as e:
            root_logger.error(f"Fehler beim Einrichten des File-Loggers: {e}", exc_info=True)
    else:
        root_logger.info("Logging in die Konsole ist aktiviert. Es wird keine Log-Datei geschrieben.")



def display_rankings_cli(config: Dict[str, Any]):
    """
    Loads, processes, and displays the current ELO rankings in a formatted table in the console.
    """
    logger = logging.getLogger(__name__)
    results_file_path = config.get("paths", {}).get("results_file")

    if not results_file_path or not os.path.exists(results_file_path):
        logger.error(f"Results file not found at '{results_file_path}'. Please run a benchmark first.")
        return

    results_data = load_json_file(results_file_path)
    if not results_data or "models" not in results_data or not results_data["models"]:
        logger.info("No model data found in the results file. Run a benchmark to generate rankings.")
        return

    models_data = results_data["models"]
    model_stats = []
    for model_name, details in models_data.items():
        cats = details.get("elo_rating_by_category", {})
        rating_values = [r for r in cats.values() if isinstance(r, (int, float))]
        avg_rating = sum(rating_values) / len(rating_values) if rating_values else 0
        model_stats.append({
            "name": model_name,
            "avg_rating": avg_rating,
            "wins": details.get("wins", 0),
            "losses": details.get("losses", 0),
            "draws": details.get("draws", 0)
        })

    model_stats.sort(key=lambda x: x["avg_rating"], reverse=True)

    # Print table
    print("\n--- OllamaBench ELO Rankings ---")
    print("-" * 80)
    print(f"{'Rank':<5} {'Model':<40} {'Avg ELO':>15} {'W/L/D':>15}")
    print("-" * 80)

    if not model_stats:
        print("No models to rank.")
    else:
        for i, stats in enumerate(model_stats):
            rank = i + 1
            wld_str = f"{stats['wins']}/{stats['losses']}/{stats['draws']}"
            print(f"{rank:<5} {stats['name']:<40} {stats['avg_rating']:>15.1f} {wld_str:>15}")

    print("-" * 80)
    timestamp = results_data.get("timestamp", "N/A")
    print(f"Last updated: {timestamp}\n")


def run_evaluation_cli(args: argparse.Namespace ,config: Dict[str, Any]):
    """
    Runs the configured number of comparison cycles sequentially from the command line.
    Args:
        config: The loaded application configuration.
    """
    logger = logging.getLogger(__name__)
    categories_to_run: List[str]
    try:
        if args.category:
            categories_to_run = [args.category]
            logger.info(f"Preparing to run for a single specified category: {args.category}")
        else:
            if not args.all_categories:
                logger.info("No category specified, defaulting to --all-categories.")
            
            categories_to_run = ALL_CATEGORIES # Fallback

        complete_run = not args.generate_only and not args.judge_only
        generate_answers = (not args.judge_only and args.generate_only) or complete_run
        run_judgement =  (not args.generate_only and args.judge_only) or complete_run
        run_benchmark(config=config, categories_to_run=categories_to_run, generate_answers= generate_answers , run_judgement= run_judgement)

    except Exception as e:
        logger.exception(f"An unexpected error occurred during the main evaluation task: {e}")

# category
if __name__ == "__main__":
    print("--- OllamaBench CLI Execution Start ---")
    try:
        parser = argparse.ArgumentParser(description="OllamaBench: A benchmark tool for local LLMs.")
        
        parser.add_argument(
            '--show-rankings',
            action='store_true',
            help='Displays the current ELO ranking table from results.json and exits.'
        )

        category_group = parser.add_mutually_exclusive_group()
        category_group.add_argument(
            '--all-categories',
            action='store_true',
            help='Run benchmark for all categories defined in config.yaml.'
        )
        category_group.add_argument(
            '--category',
            type=str,
            help='Run benchmark for a single specified category.'
        )

        step_group = parser.add_mutually_exclusive_group()
        step_group.add_argument(
            '--generate-only',
            action='store_true',
            help='Only generate model answers, do not run judgement.'
        )
        step_group.add_argument(
            '--judge-only',
            action='store_true',
            help='Only run judgement on existing answers, do not generate new ones.'
        )

        args = parser.parse_args()
        
        # --- Load Configuration ---
        print(f"Loading configuration from: {DEFAULT_CONFIG_PATH}")
        CONFIG = load_config(DEFAULT_CONFIG_PATH)
        print("Configuration loaded.")

        # --- Setup Logging ---
        print("Setting up logging...")
        setup_logging(CONFIG)
        main_logger = logging.getLogger(__name__)

        # If --show-rankings is used, display the table and exit
        if args.show_rankings:
            display_rankings_cli(CONFIG)
            sys.exit(0)

        # --- Run Evaluation ---
        main_logger.info("Starting evaluation process...")
        run_evaluation_cli(args,CONFIG)
        main_logger.info("Evaluation process finished.")

    except FileNotFoundError as e:
        print(f"ERROR: Configuration file not found. {e}", file=sys.stderr)
        logging.critical(f"Configuration file not found. {e}")
        sys.exit(1)
    except (ValueError, yaml.YAMLError) as e:
        print(f"ERROR: Invalid configuration file. {e}", file=sys.stderr)
        logging.critical(f"Invalid configuration file. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR during initialization or execution: {e}", file=sys.stderr)
        try:
             logging.critical(f"FATAL ERROR during initialization or execution: {e}", exc_info=True)
        except:
             pass
        sys.exit(1)

    print("--- OllamaBench CLI Execution End ---")
    sys.exit(0)