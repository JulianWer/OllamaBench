import argparse
import logging
import sys
from typing import Dict, Any, List

import yaml

from scripts.engine import run_benchmark
from utils.config import load_config, DEFAULT_CONFIG_PATH

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
    log_level_str = config.get("logging", {}).get("level", "INFO").upper()
    log_format = config.get("logging", {}).get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    log_handlers = [logging.StreamHandler(sys.stdout)] # Log to stdout by default
    logging.basicConfig(level=getattr(logging, log_level_str, logging.INFO),
                        format=log_format,
                        handlers=log_handlers)


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
        generate_answers = not args.judge_only and args.generate_only or complete_run
        run_judgement =  not args.generate_only and args.judge_only or complete_run
        run_benchmark(config=config, categories_to_run=categories_to_run, generate_answers= generate_answers , run_judgement= run_judgement)

    except Exception as e:
        logger.exception(f"An unexpected error occurred during the main evaluation task: {e}")


if __name__ == "__main__":
    print("--- LocalBench CLI Execution Start ---")
    try:
        parser = argparse.ArgumentParser(description="OllamaBench: A benchmark tool for local LLMs.")

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

    print("--- LocalBench CLI Execution End ---")
    sys.exit(0) 
