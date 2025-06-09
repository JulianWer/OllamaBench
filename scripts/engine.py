import logging
from typing import Dict, Any, List

from scripts.generate_model_answers_for_category import generate_and_save_model_answers_for_category
from scripts.run_comparison import run_judge_comparison

logger = logging.getLogger(__name__)


def run_benchmark(
    config: Dict[str, Any],
    categories_to_run: List[str],
    generate_answers: bool = True,
    run_judgement: bool = True,
):
    """
    Orchestrates the entire benchmark process for a given list of categories.

    This function centralizes the logic for generating model answers and running
    judge comparisons, which can be called by both the CLI and the web dashboard.

    Args:
        config: The application's configuration dictionary.
        categories_to_run: A list of category strings to be evaluated.
        generate_answers: If True, models will generate answers for the prompts.
        run_judgement: If True, the judge model will compare the generated answers.
    """
    if not categories_to_run:
        logger.warning("run_benchmark called with no categories to run. Exiting.")
        return

    logger.info(
        f"Starting benchmark for categories: {categories_to_run}. "
        f"Generate answers: {generate_answers}, Run judgement: {run_judgement}"
    )
    
    success_count = 0
    error_count = 0

    for i, category in enumerate(categories_to_run, 1):
        logger.info(f"--- Processing category {i}/{len(categories_to_run)}: '{category}' ---")
        try:
            if generate_answers:
                generate_and_save_model_answers_for_category(config=config, category=category)
                success_count += 1
            else:
                logger.info(f"Skipping answer generation for category '{category}'.")
                error_count += 1

            if run_judgement:
                run_judge_comparison(config, category)
                success_count += 1
            else:
                logger.info(f"Skipping judgement for category '{category}'.")
                error_count += 1

        except Exception as e:
            logger.exception(
                f"A critical error occurred while processing category '{category}'. "
                f"Moving to the next category. Error: {e}"
            )

    logger.info("Benchmark run completed for all specified categories.")
    return success_count, error_count