import logging
import random
from typing import Dict, Any, Optional, Union, List # Added List
from datasets import load_dataset, Dataset, IterableDataset

from evaluation.ranking import update_elo, Category, ModelRatingsType
from models.judge_llm import judge_responses
from models.llms import generate_responses_sequentially, get_two_random_models
from utils.file_operations import load_current_json, save_elo_results

logger = logging.getLogger(__name__)

PROMPT_DATASET_CACHE: Optional[Union[Dataset, IterableDataset]] = None
PROMPT_DATASET_NAME: Optional[str] = None
PROMPT_COLUMN: Optional[str] = None

def _load_prompt_dataset(config: Dict[str, Any]) -> Optional[Union[Dataset, IterableDataset]]:
    global PROMPT_DATASET_CACHE, PROMPT_DATASET_NAME, PROMPT_COLUMN

    dataset_name = config.get("PROMPT_DATASET")
    prompt_column = config.get("PROMPT_DATASET_COLUMN")

    if not dataset_name or not prompt_column:
        logger.error("PROMPT_DATASET or PROMPT_DATASET_COLUMN not specified in config.")
        return None

    if PROMPT_DATASET_CACHE is not None and PROMPT_DATASET_NAME == dataset_name and PROMPT_COLUMN == prompt_column:
        logger.debug(f"Using cached dataset '{dataset_name}'.")
        return PROMPT_DATASET_CACHE

    logger.info(f"Loading prompt dataset '{dataset_name}'...")
    try:
        # Load the dataset (consider adding streaming=True for large datasets)
        ds = load_dataset(dataset_name, split="test") # Stick to train split for consistency
        if prompt_column not in ds.column_names:
             logger.error(f"Prompt column '{prompt_column}' not found in dataset '{dataset_name}'. Available columns: {ds.column_names}")
             return None

        # Cache the dataset and its info
        PROMPT_DATASET_CACHE = ds
        PROMPT_DATASET_NAME = dataset_name
        PROMPT_COLUMN = prompt_column
        logger.info(f"Dataset '{dataset_name}' loaded successfully.")
        return ds
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}': {e}", exc_info=True)
        # Reset cache info on failure
        PROMPT_DATASET_CACHE = None
        PROMPT_DATASET_NAME = None
        PROMPT_COLUMN = None
        return None

def _get_random_prompt(config: Dict[str, Any]) -> Optional[str]:
    """
    Gets a random prompt from the configured dataset column.
    Assumes the column contains a list and uses the first element if it's a string.
    """
    dataset = _load_prompt_dataset(config)
    prompt_column = config.get("PROMPT_DATASET_COLUMN")

    if dataset is None or prompt_column is None:
        return None

    try:
        retrieved_prompt_data = None
        # Efficiently get a random index if it's a standard Dataset
        if isinstance(dataset, Dataset):
            if len(dataset) == 0:
                 logger.error(f"Dataset '{PROMPT_DATASET_NAME}' is empty.")
                 return None
            # --- Get random data ---
            random_index = random.randint(0, len(dataset) - 1)
            retrieved_prompt_data = dataset[random_index][prompt_column]
            logger.debug(f"Selected random data (index {random_index}) from column '{prompt_column}'. Type: {type(retrieved_prompt_data)}")

        # Handle IterableDataset (less efficient random access)
        elif isinstance(dataset, IterableDataset):
             logger.warning("Fetching random prompt from IterableDataset, might be slow.")
             num_to_sample = 1000 # Sample from the first 1000 prompts
             sampled_items = [item[prompt_column] for i, item in zip(range(num_to_sample), dataset) if prompt_column in item]
             if not sampled_items:
                 logger.error(f"Could not sample any items from column '{prompt_column}' in the IterableDataset.")
                 return None
             retrieved_prompt_data = random.choice(sampled_items)
             logger.debug(f"Selected random data (sampled) from column '{prompt_column}'. Type: {type(retrieved_prompt_data)}")

        else:
            logger.error(f"Unsupported dataset type: {type(dataset)}")
            return None

        # --- Process retrieved data (assuming it might be a list) ---
        if isinstance(retrieved_prompt_data, list):
            if len(retrieved_prompt_data) > 0:
                first_element = retrieved_prompt_data[0]
                if isinstance(first_element, str):
                    prompt = first_element
                    logger.debug(f"Retrieved data was a list. Using first element as string prompt: {prompt[:100]}...")
                    return prompt
                else:
                    logger.warning(f"Retrieved data was a list, but the first element is not a string (Type: {type(first_element)}). Value: {first_element}. Skipping this entry.")
                    return None
            else:
                logger.warning(f"Retrieved data was an empty list from column '{prompt_column}'. Skipping this entry.")
                return None
        elif isinstance(retrieved_prompt_data, str):
             # Handle case where it might sometimes be a string directly
             prompt = retrieved_prompt_data
             logger.debug(f"Retrieved data was already a string: {prompt[:100]}...")
             return prompt
        else:
            # Log other unexpected types
            logger.warning(f"Retrieved data from column '{prompt_column}' is not a list or string (Type: {type(retrieved_prompt_data)}). Value: {retrieved_prompt_data}. Skipping this entry.")
            return None

    except Exception as e:
        logger.error(f"Error getting random prompt: {e}", exc_info=True)
        return None


def run_comparison_cycle(config: Dict[str, Any]) -> Optional[ModelRatingsType]:
    """
    Runs a single comparison cycle: selects models, gets prompt, generates, judges, updates ELO.

    Args:
        config: The application configuration.

    Returns:
        The updated model ratings dictionary, or None if a critical step failed.
    """
    logger.info("Starting new comparison cycle...")

    # 1. Select two distinct random models
    models_tuple = get_two_random_models()
    if models_tuple is None:
        logger.error("Comparison cycle failed: Could not select two models.")
        return None
    model_a, model_b = models_tuple

    # 2. Get a random prompt (string) from the dataset
    #    _get_random_prompt now handles potential lists in the source column
    prompt = _get_random_prompt(config)
    if prompt is None:
        # Error logged within _get_random_prompt if retrieval failed or wasn't valid
        logger.error("Comparison cycle failed: Could not retrieve a valid string prompt.")
        return None # Comparison cycle cannot proceed without a valid prompt

    # 3. Generate responses concurrently, passing the config
    logger.info(f"Generating responses for prompt: '{prompt[:100]}...'")
    responses = generate_responses_sequentially(
        model_list=[model_a, model_b],
        message=prompt, # Should be a string here if not None
        config=config, # Pass the config dictionary
        system_prompt=config.get("GENERATION_SYSTEM_PROMPT") # Optional system prompt
    )

    response_a = responses.get(model_a)
    response_b = responses.get(model_b)

    # Check if generation failed for either model
    if response_a is None or response_b is None:
        logger.error(f"Comparison cycle failed: Failed to generate response from one or both models ({model_a}: {'OK' if response_a else 'FAIL'}, {model_b}: {'OK' if response_b else 'FAIL'}). Prompt: '{prompt[:100]}...'")
        # Note: generate_response logs errors internally if API calls fail
        return None # Cannot judge if a response is missing

    # 4. Judge the responses
    judge_model = config.get("JUDGE_MODEL")
    if not judge_model:
         logger.error("Comparison cycle failed: JUDGE_MODEL not specified in config.")
         return None

    score_a = judge_responses(
        judge_model=judge_model,
        response1=response_a,
        response2=response_b,
        prompt=prompt, # Pass the valid string prompt
        system_prompt=config.get("JUDGE_SYSTEM_PROMPT"),
    )

    if score_a is None:
        logger.error("Comparison cycle failed: Judging failed or returned an invalid verdict.")
        # judge_responses logs details internally
        return None # Cannot update ELO without a valid score

    # 5. Load current ratings
    current_results = load_current_json(config["paths"]["results_file"], config["paths"]["lock_file"])
    # Initialize model_ratings safely
    model_ratings: ModelRatingsType = current_results.get("models", {}) if isinstance(current_results, dict) else {}


    # 6. Update ELO ratings
    category_str = config.get("DEFAULT_CATEGORY", "general") # Use default category from config
    try:
        # Ensure category exists in Enum if using strict validation
        category = Category.from_string(category_str)
    except ValueError:
        logger.warning(f"DEFAULT_CATEGORY '{category_str}' in config is not a predefined Category enum member. Proceeding with the string.")
        category = category_str # Use the string directly

    update_elo(
        model_ratings=model_ratings, # Updated in-place
        model_a=model_a,
        model_b=model_b,
        category=category, # Pass the determined category
        score_a=score_a,
        k_factor=config.get("elo", {}).get("k_factor", 32),
        initial_rating=config.get("elo", {}).get("initial_rating", 1500.0)
    )

    # 7. Save updated ratings
    if not save_elo_results(model_ratings, config):
         logger.error("Comparison cycle partially failed: Could not save updated ELO results.")
         # Decide if returning the in-memory update is acceptable or signal failure
         return None # Signal failure if saving is critical

    logger.info(f"Comparison cycle completed successfully for {model_a} vs {model_b}.")
    return model_ratings # Return the updated ratings

