import logging
import os
import random
from itertools import combinations
from typing import Dict, Any, List, Optional, Tuple

from evaluation.ranking import (MatchType, calculate_mELO_ratings as calculate_ratings, ModelRatingsType)
from models.factory import create_judge_llm
from models.judge_llm import JudgeLLM
from utils.config import ConfigService
from utils.file_operations import load_json_file, save_elo_results
from utils.get_data import load_prompt_dataset, extract_prompt_info_from_dataset_item, extract_prompt_text

logger = logging.getLogger(__name__)

def _get_model_answers_for_prompt(prompt_id: Any, category: str, config_service: ConfigService) -> Optional[List[Dict[str, Any]]]:
    """Loads all existing model answers for a specific prompt."""
    base_output_dir = config_service.get_path("output_dir")
    if not base_output_dir:
        logger.error("Path 'output_dir' not configured.")
        return None

    category_dir = os.path.join(base_output_dir, str(category))
    safe_prompt_id_filename = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in str(prompt_id)) + ".json"
    answers_file_path = os.path.join(category_dir, safe_prompt_id_filename)

    logger.info(f"Loading answers for prompt ID {prompt_id} from: {answers_file_path}")
    return load_json_file(answers_file_path)

def _perform_judgement_for_pair(
    judge: JudgeLLM,
    prompt_text: str,
    ground_truth: Optional[str],
    model_a_name: str,
    model_b_name: str,
    response_a: str,
    response_b: str
) -> Optional[float]:
    """Runs the judge on a pair of responses, handling position bias, and returns a single score for model A."""
    # handle Position bias 
    verdict_1 = judge.evaluate(
        user_prompt=prompt_text,
        response_a=response_a,
        response_b=response_b,
        ground_truth=ground_truth
    )
    verdict_2 = judge.evaluate(
        user_prompt=prompt_text,
        response_a=response_b,
        response_b=response_a,
        ground_truth=ground_truth
    )

    if verdict_1 is None or verdict_2 is None:
        logger.warning(f"Judging failed for one or both positions for pair ({model_a_name} vs {model_b_name}). Skipping.")
        return None
    
    final_score_model_a = 0.5
    if verdict_1 != verdict_2 and verdict_1 != 0.5 and verdict_2 != 0.5:
        logger.warning(f"Contradictory verdicts for {model_a_name} vs {model_b_name}. Verdict1={verdict_1}, Verdict2={verdict_2}. Using first verdict.")
        final_score_model_a = verdict_1
    elif verdict_1 == verdict_2:
        final_score_model_a = 0.5

    logger.info(f"Judgement for pair ({model_a_name} vs {model_b_name}): Score for A = {final_score_model_a}")
    return final_score_model_a

def _process_prompt_comparisons(
    prompt: Dict[str, Any],
    category: str,
    config_service: ConfigService,
    judge: 'JudgeLLM'
) -> List[MatchType]:
    """Processes all comparisons for a single prompt and returns a list of matches."""
    matches = []
    
    prompt_content, ground_truth, prompt_id = extract_prompt_info_from_dataset_item(
        config_service.get_full_config(), prompt
    )
    if prompt_id is None:
        logger.warning("Skipping prompt with no ID.")
        return []

    existing_answers = _get_model_answers_for_prompt(prompt_id, category, config_service)
    if not isinstance(existing_answers, list) or len(existing_answers) < 2:
        return []

    prompt_text_for_judge = extract_prompt_text(prompt_content)
    if prompt_text_for_judge is None:
        return []

    all_pairs = list(combinations(existing_answers, 2))
    num_comparisons_per_prompt = config_service.get_full_config().get("comparison", {}).get("comparisons_per_prompt", 30)
    
    random.shuffle(all_pairs)
    pairs_to_process = all_pairs[:num_comparisons_per_prompt]
    
    logger.info(f"Found {len(existing_answers)} answers for prompt {prompt_id}. Processing {len(pairs_to_process)}/{len(all_pairs)} pairs.")

    for response_obj_1, response_obj_2 in pairs_to_process:
        model_a, model_b = response_obj_1.get("model"), response_obj_2.get("model")
        response_a, response_b = response_obj_1.get("response"), response_obj_2.get("response")

        if not all([model_a, model_b, response_a is not None, response_b is not None, model_a != model_b]):
            continue

        final_score = _perform_judgement_for_pair(
            judge, prompt_text_for_judge, ground_truth, model_a, model_b, response_a, response_b
        )
        if final_score is not None:
            matches.append((model_a, model_b, category, final_score))
            
    return matches

def run_judge_comparison(config_dict: Dict[str, Any], category: str) -> Optional[ModelRatingsType]:
    config_service = ConfigService()
    
    judge = create_judge_llm(config_service)
    if not judge:
        logger.critical("Could not create Judge LLM. Aborting comparison.")
        return None

    dataset = load_prompt_dataset(config_dict, category)
    if not dataset:
        return None

    logger.info(f"--- Starting judge comparisons for category: '{category}' ---")
    
    all_matches_for_category: List[MatchType] = []
    for prompt in dataset:
        try:
            matches_from_prompt = _process_prompt_comparisons(prompt, category, config_service, judge)
            all_matches_for_category.extend(matches_from_prompt)
        except Exception as e:
            prompt_id = prompt.get(config_service.dataset_config.get("columns", {}).get("id", "N/A"))
            logger.exception(f"An unexpected error occurred while processing prompt ID {prompt_id}: {e}")

    if not all_matches_for_category:
        logger.warning(f"No valid matches were generated for category '{category}'. Skipping ELO update.")
        return None

    logger.info(f"Collected {len(all_matches_for_category)} matches for '{category}'. Starting batch rating calculation.")

    results_file = config_service.get_path("results_file")
    lock_file = config_service.get_path("lock_file")
    current_results = load_json_file(results_file, lock_file) or {}
    model_ratings_store: ModelRatingsType = current_results.get("models", {})
    
    mELO_config = config_service.mELO_config
    calculate_ratings(
        model_ratings=model_ratings_store,
        all_matches=all_matches_for_category,
        initial_rating=mELO_config.get("initial_rating", 1000.0),
        learning_rate=mELO_config.get("learning_rate", 100),
        epochs=mELO_config.get("epochs", 300),
    )

    if save_elo_results(model_ratings_store, config_dict):
        logger.info(f"Successfully saved updated ratings for '{category}' to '{results_file}'.")
    else:
        logger.error(f"Failed to save ratings for '{category}'.")
        
    return model_ratings_store