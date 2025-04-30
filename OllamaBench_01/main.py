import logging
import time
import os
import sys
from typing import Dict, Any

import yaml

# Project specific imports
from utils.config import load_config, DEFAULT_CONFIG_PATH
from scripts.run_comparison import run_comparison_cycle

# --- Global Variables ---
CONFIG: Dict[str, Any] = {}

# --- Logging Setup Function ---
def setup_logging(config: Dict[str, Any]):
    """Configures logging based on the loaded configuration."""
    log_level_str = config.get("logging", {}).get("level", "INFO").upper()
    log_format = config.get("logging", {}).get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = config.get("paths", {}).get("log_file") # Optional log file from config

    log_handlers = [logging.StreamHandler(sys.stdout)] # Log to stdout by default
    if log_file:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir: # Create directory only if it's specified
                 os.makedirs(log_dir, exist_ok=True)
            log_handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
            print(f"Logging also to file: {log_file}") # Print confirmation
        except Exception as e:
            # Use basic config temporarily if full setup fails
            logging.basicConfig(level=logging.ERROR)
            logging.error(f"Failed to create file handler for log file '{log_file}': {e}. Logging to console only.")
            log_handlers = [logging.StreamHandler(sys.stdout)] # Fallback to console only

    # Set up root logger
    logging.basicConfig(level=getattr(logging, log_level_str, logging.INFO),
                        format=log_format,
                        handlers=log_handlers)

    # Configure log levels for specific libraries if needed
    # logging.getLogger("requests").setLevel(logging.WARNING)
    # logging.getLogger("urllib3").setLevel(logging.WARNING)
    # logging.getLogger("filelock").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__) # Get logger for this module
    logger.info("Logging configured successfully.")
    logger.info(f"Log level set to: {log_level_str}")


# --- Main Evaluation Function ---
def run_evaluation_cli(config: Dict[str, Any]):
    """
    Runs the configured number of comparison cycles sequentially from the command line.

    Args:
        config: The loaded application configuration.
    """
    logger = logging.getLogger(__name__) # Get logger for this function
    try:
        num_comparisons = config.get("comparison", {}).get("comparisons_per_run", 1)
        if not isinstance(num_comparisons, int) or num_comparisons <= 0:
             logger.warning(f"Invalid 'comparisons_per_run' value ({num_comparisons}). Defaulting to 1.")
             num_comparisons = 1

        logger.info(f"Starting command-line evaluation for {num_comparisons} comparison(s).")

        errors = []
        success_count = 0
        total_start_time = time.monotonic()

        for i in range(num_comparisons):
            cycle_start_time = time.monotonic()
            logger.info(f"--- Running Comparison Cycle {i + 1}/{num_comparisons} ---")
            try:
                # Execute a single comparison cycle using the shared function
                updated_ratings = run_comparison_cycle(config) # Pass the loaded config
                cycle_duration = time.monotonic() - cycle_start_time

                if updated_ratings is None:
                    logger.warning(f"Comparison cycle {i + 1} did not complete successfully or returned None (Duration: {cycle_duration:.2f}s).")
                    errors.append(f"Cycle {i + 1} failed.")
                    # Decide whether to continue or stop on failure
                    # break # Uncomment to stop on the first failure
                else:
                    success_count += 1
                    logger.info(f"Comparison cycle {i + 1} completed successfully (Duration: {cycle_duration:.2f}s).")

            except Exception as e:
                cycle_duration = time.monotonic() - cycle_start_time
                logger.exception(f"Critical error during comparison cycle {i + 1} (Duration: {cycle_duration:.2f}s): {e}")
                errors.append(f"Cycle {i + 1} critical error: {e}")
                break # Stop execution on critical errors within a cycle

        total_duration = time.monotonic() - total_start_time
        logger.info("-" * 40)
        logger.info(f"Evaluation Finished in {total_duration:.2f} seconds.")
        logger.info(f"Successful Cycles: {success_count}/{num_comparisons}")
        if errors:
            logger.error(f"{len(errors)} Error(s) Occurred:")
            for err_msg in errors:
                logger.error(f"  - {err_msg}")
        logger.info("-" * 40)

    except Exception as e:
        logger.exception(f"An unexpected error occurred during the main evaluation task: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- OllamaBench CLI Execution Start ---")
    try:
        # 1. Load Configuration
        print(f"Loading configuration from: {DEFAULT_CONFIG_PATH}")
        CONFIG = load_config(DEFAULT_CONFIG_PATH) # Load into global variable
        print("Configuration loaded.")

        # 2. Setup Logging (using loaded config)
        print("Setting up logging...")
        setup_logging(CONFIG)
        main_logger = logging.getLogger(__name__) # Get logger after setup

        # 3. Run Evaluation
        main_logger.info("Starting evaluation process...")
        run_evaluation_cli(CONFIG)
        main_logger.info("Evaluation process finished.")

    except FileNotFoundError as e:
        # Basic print/log if logging setup failed due to missing config
        print(f"ERROR: Configuration file not found. {e}", file=sys.stderr)
        logging.critical(f"Configuration file not found. {e}")
        sys.exit(1) # Exit with error code
    except (ValueError, yaml.YAMLError) as e:
        print(f"ERROR: Invalid configuration file. {e}", file=sys.stderr)
        logging.critical(f"Invalid configuration file. {e}")
        sys.exit(1)
    except Exception as e:
        # Catch-all for other unexpected errors during startup
        print(f"FATAL ERROR during initialization or execution: {e}", file=sys.stderr)
        # Attempt to log if possible, otherwise just print
        try:
             logging.critical(f"FATAL ERROR during initialization or execution: {e}", exc_info=True)
        except:
             pass # Ignore logging errors if logging itself failed
        sys.exit(1)

    print("--- OllamaBench CLI Execution End ---")
    sys.exit(0) # Explicitly exit with success code
