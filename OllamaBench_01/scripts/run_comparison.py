import logging
from typing import Dict, Any, Optional, Tuple, List, Union # Added List, Union

# Project specific imports
from evaluation.ranking import update_elo, ModelRatingsType
from models.judge_llm import judge_responses
from models.llms import generate_responses_sequentially, get_two_random_models
# Import the correct function name from the reverted get_data.py
from utils.get_data import get_random_prompt_and_ground_truth # Use the function returning a tuple
from utils.file_operations import save_elo_results, load_json_file

logger = logging.getLogger(__name__)

# Helper function to extract text remains useful
def _extract_prompt_text(prompt_data: Any) -> Optional[str]:
    """
    Extracts a single string prompt from potentially complex prompt data.
    Prioritizes the last user turn, falls back to concatenating all turns' content.
    """
    if isinstance(prompt_data, str):
        return prompt_data
    elif isinstance(prompt_data, list):
        user_turns_content = []
        all_turns_content = []
        for turn in prompt_data:
            if isinstance(turn, dict) and isinstance(turn.get("content"), str):
                content = turn["content"]
                all_turns_content.append(content)
                if turn.get("role") == "user":
                    user_turns_content.append(content)
            elif isinstance(turn, str):
                 all_turns_content.append(turn)
        if user_turns_content:
            logger.debug("Extracted prompt text from the last 'user' turn.")
            return user_turns_content[-1]
        elif all_turns_content:
            logger.warning(f"No 'user' role found in prompt turns. Falling back to concatenating content from all {len(all_turns_content)} turns/strings.")
            return "\n".join(all_turns_content)
        else:
            logger.warning(f"Prompt data is a list but contains no extractable string content: {prompt_data}")
            return None
    else:
        logger.warning(f"Unsupported prompt data type: {type(prompt_data)}. Cannot extract text.")
        return None


def run_comparison_cycle(config: Dict[str, Any]) -> Optional[ModelRatingsType]:
    """
    Executes a single comparison cycle.
    (Adapted for the reverted get_data.py version returning a tuple)
    """
    logger.info("=" * 20 + " Starting Comparison Cycle " + "=" * 20)
    cycle_success = False
    updated_ratings_snapshot: Optional[ModelRatingsType] = None

    try:
        # 1. Select Models
        logger.info("Step 1: Selecting models...")
        models_tuple = get_two_random_models(config)
        if models_tuple is None:
            logger.error("Cycle failed: Could not select two models.")
            return None
        model_a, model_b = models_tuple
        logger.info(f"Models selected: '{model_a}' vs '{model_b}'")

        # 2. Get Prompt Details (using the reverted get_data function)
        logger.info("Step 2: Fetching prompt and ground truth...")
        # Call the function that returns a tuple
        prompt_data_tuple = get_random_prompt_and_ground_truth(config)

        if prompt_data_tuple is None:
            logger.error("Cycle failed: Could not retrieve valid prompt/ground truth tuple.")
            return None

        # --- Correctly unpack the tuple ---
        prompt_content, ground_truth = prompt_data_tuple
        # --- End Correction ---

        # Category is not returned by the reverted function, use default from config
        category = config.get("CURRENT_CATEGORY")
        logger.info(f"Using default category: '{category}' (as it's not provided by get_random_prompt_and_ground_truth)")

        # Extract a single string representation for logging/judging if needed
        # This helper function still works with the extracted prompt_content
        prompt_text_for_judge = _extract_prompt_text(prompt_content)
        if prompt_text_for_judge is None:
             # This might happen if prompt_content was an unusable list/type
             logger.error(f"Cycle failed: Could not extract usable prompt text from data: {prompt_content}")
             return None

        logger.info(f"Prompt category: '{category}'. Ground Truth available: {'Yes' if ground_truth else 'No'}.")
        logger.debug(f"Prompt Text (for judge): {prompt_text_for_judge[:150]}...")

        # 3. Generate Responses
        logger.info(f"Step 3: Generating responses from '{model_a}' and '{model_b}'...")
        generation_system_prompt = config.get("generation_system_prompt")
        # Pass the potentially raw prompt_content (str or list)
        logger.info(f"Prompt Content'{prompt_content}'")

        responses = generate_responses_sequentially(
            model_list=[model_a, model_b],
            prompt=prompt_content,
            config=config,
            system_prompt=generation_system_prompt
        )

        response_a = responses.get(model_a)
        response_b = responses.get(model_b)

        if response_a is None or response_b is None:
            logger.error(f"Cycle failed: Response generation failed ({model_a}: {'OK' if response_a else 'FAIL'}, {model_b}: {'OK' if response_b else 'FAIL'}).")
            return None

        logger.info(f"Responses generated successfully for '{model_a}' and '{model_b}'.")

        # 4. Judge Responses
        judge_model = config.get("JUDGE_MODEL")
        judge_system_prompt = config.get("judge_system_prompt")
        if not judge_model:
            logger.error("Cycle failed: JUDGE_MODEL not specified in configuration.")
            return None

        logger.info(f"Step 4: Judging responses using '{judge_model}'...")
        score_a = judge_responses(
            judge_model=judge_model,
            response1=response_a,
            response2=response_b,
            prompt=prompt_text_for_judge, # Use the extracted text prompt
            config=config,
            ground_truth=ground_truth,
            judge_system_prompt=judge_system_prompt,
        )

        if score_a is None:
            logger.error("Cycle failed: Judging process failed or returned an invalid verdict.")
            return None

        verdict_map = {1.0: f"{model_a} wins", 0.0: f"{model_b} wins", 0.5: "Tie"}
        logger.info(f"Judgment result: {verdict_map[score_a]} (Score A: {score_a})")

        # 5. Update ELO Ratings
        logger.info(f"Step 5: Updating ELO ratings for category '{category}'...")
        k_factor = config.get("elo", {}).get("k_factor", 32)
        initial_rating = config.get("elo", {}).get("initial_rating", 1000.0)

        # Load current ratings before update
        logger.debug("Loading current ratings before ELO update...")
        current_results_data = load_json_file(config["paths"]["results_file"], config["paths"]["lock_file"])
        ratings_obj_for_update = {}
        if isinstance(current_results_data, dict) and isinstance(current_results_data.get("models"), dict):
             ratings_obj_for_update = current_results_data.get("models", {})
             logger.debug(f"Loaded {len(ratings_obj_for_update)} models from results file for ELO calculation.")
        else:
             logger.warning("Could not load valid 'models' data from results file. ELO calculation will use initial ratings.")

        # update_elo modifies ratings_obj_for_update in place
        new_rating_a, new_rating_b = update_elo(
            model_ratings=ratings_obj_for_update,
            model_a=model_a,
            model_b=model_b,
            category=category, # Pass category string directly
            score_a=score_a,
            k_factor=k_factor,
            initial_rating=initial_rating
        )

        if new_rating_a is None or new_rating_b is None:
             logger.error("Cycle failed: ELO update function indicated an error (likely invalid score).")
             return None

        updated_ratings_snapshot = ratings_obj_for_update
        logger.info(f"ELO ratings updated in memory: {model_a} -> {new_rating_a:.1f}, {model_b} -> {new_rating_b:.1f}")

        # 6. Save Results
        logger.info("Step 6: Saving updated ELO results...")
        if not save_elo_results(updated_ratings_snapshot, config):
             logger.error("Failed to save updated ELO results to file. Ratings may be inconsistent.")
        else:
             logger.info("ELO results saved successfully.")

        cycle_success = True

    except Exception as e:
        logger.exception(f"An unexpected error terminated the comparison cycle: {e}")
        return None # Return None on unexpected errors during the cycle
    finally:
        logger.info("=" * 20 + f" Comparison Cycle {'Finished' if cycle_success else 'Failed'} " + "=" * 20)

    return updated_ratings_snapshot if cycle_success else None

