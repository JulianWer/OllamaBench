import logging
import sys
from typing import Dict, Any

import yaml

from scripts.generate_model_answers_for_category import generate_and_save_model_answers_for_category
from utils.config import load_config, DEFAULT_CONFIG_PATH
from scripts.run_comparison import run_judge_comparison

CONFIG: Dict[str, Any] = {}

def setup_logging(config: Dict[str, Any]):
    """Configures logging based on the loaded configuration."""
    log_level_str = config.get("logging", {}).get("level", "INFO").upper()
    log_format = config.get("logging", {}).get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    log_handlers = [logging.StreamHandler(sys.stdout)] # Log to stdout by default
    logging.basicConfig(level=getattr(logging, log_level_str, logging.INFO),
                        format=log_format,
                        handlers=log_handlers)


def run_evaluation_cli(config: Dict[str, Any]):
    """
    Runs the configured number of comparison cycles sequentially from the command line.

    Args:
        config: The loaded application configuration.
    """
    logger = logging.getLogger(__name__) 
    try:
        default_category = config.get("CURRENT_CATEGORY")
        all_categories = config.get("ALL_CATEGORIES")
        run_all_categories_is_enabled = config.get("RUN_ALL_CATEGORIES_IS_ENABLED")
        run_only_judgement = config.get("RUN_ONLY_JUDGEMENT")

        
        if run_all_categories_is_enabled:
            for category in all_categories:
                try:
                    # firstly gerneate all answers from all models and then judge
                    if not run_only_judgement:
                        generate_and_save_model_answers_for_category(config=config, category=category)
                    else:
                        run_judge_comparison(config,category) 
                except Exception as e:
                    logger.exception(f"Critical error during comparison cycle: {e}")
                    break 
        else:
            try:
                if not run_only_judgement:
                    generate_and_save_model_answers_for_category(config=config, category=default_category)
                else:
                    run_judge_comparison(config,default_category)                      

            except Exception as e:
                logger.exception(f"Critical error: {e}")
                 

    except Exception as e:
        logger.exception(f"An unexpected error occurred during the main evaluation task: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- OllamaBench CLI Execution Start ---")
    try:
        # 1. Load Configuration
        print(f"Loading configuration from: {DEFAULT_CONFIG_PATH}")
        CONFIG = load_config(DEFAULT_CONFIG_PATH)
        print("Configuration loaded.")

        # 2. Setup Logging (using loaded config)
        print("Setting up logging...")
        setup_logging(CONFIG)
        main_logger = logging.getLogger(__name__) 

        # 3. Run Evaluation
        main_logger.info("Starting evaluation process...")
        run_evaluation_cli(CONFIG)
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
