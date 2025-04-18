import logging
import time 

from utils.config import load_config
from scripts.run_comparison import run_comparison_cycle

CONFIG_PATH = './config/config.yaml'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def run_evaluation():
    logger.info("Received request for /api/run_eval (synchronous)")

    try:
        config = load_config(CONFIG_PATH)
        num_comparisons = config.get("comparison", {}).get("comparisons_per_run", 1)
        logger.info(f"Starting synchronous evaluation for {num_comparisons} comparisons.")

        final_ratings = None
        errors = []
        success_count = 0

        for i in range(num_comparisons):
            logger.info(f"Running comparison {i+1}/{num_comparisons}...")
            try:
                # Run one comparison cycle directly
                updated_ratings = run_comparison_cycle(config)
                if updated_ratings is None:
                    logger.warning(f"Comparison cycle {i+1} did not complete successfully or returned None.")
                    errors.append(f"Comparison cycle {i+1} failed.")
                else:
                    success_count += 1
                    logger.info(f"Comparison cycle {i+1} completed.")

            except Exception as e:
                logger.error(f"Unhandled exception in comparison cycle {i+1}: {e}", exc_info=True)
                errors.append(f"Error during comparison {i+1}: {e}")
                break

        logger.info(f"Synchronous evaluation finished. Successful cycles: {success_count}/{num_comparisons}.")


    except FileNotFoundError:
        logger.error(f"Configuration file '{CONFIG_PATH}' not found.")
    except Exception as e:
        logger.error(f"Failed to run evaluation task: {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    try:
        load_config(CONFIG_PATH)
    except Exception as e:
        logger.critical(f"Failed to load initial configuration: {e}. Exiting.")
        exit(1) # Exit if config is bad

    run_evaluation()
