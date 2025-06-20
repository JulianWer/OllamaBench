from itertools import combinations
import logging
import os
import random
from typing import Dict, Any, List, Optional

from evaluation.ranking import MatchType, calculate_mELO_ratings_by_gradient_descent
from utils.get_data import load_prompt_dataset, extract_prompt_info_from_dataset_item

try:
    from evaluation.ranking import ModelRatingsType
    from models.judge_llm import JudgeLLM
    from utils.file_operations import save_elo_results, load_json_file
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import all necessary modules. Using placeholders. Ensure your project structure and PYTHONPATH are correct.")
    ModelRatingsType = Dict[str, Any]
    class JudgeLLM:
        def __init__(self, *args, **kwargs): pass
        def evaluate(self, *args, **kwargs): return None
    def save_elo_results(*args, **kwargs): return False
    def load_json_file(*args, **kwargs): return None


logger = logging.getLogger(__name__)

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


def run_judge_comparison(config: Dict[str, Any], category: str) -> Optional[ModelRatingsType]:
    """
    Executes comparison cycles for a given category.
    Selects random prompts and then random pairs of existing model answers for those prompts.
    """
    all_matches_for_category: List[MatchType] = [] # Collect all matches for batch elo update

    # --- Load Config ---
    num_pair_comparisons_per_prompt = config.get("comparison", {}).get("comparisons_per_prompt", 30) 

    paths_config = config.get("paths", {})
    base_output_dir = paths_config.get("output_dir", "model_responses") 
    
    k_factor = config.get("elo", {}).get("k_factor", 32)
    initial_rating_elo = config.get("elo", {}).get("initial_rating", 1000.0)
    
    llm_runtime_api_url = config.get("LLM_runtime", {}).get("api_base_url")
    judge_model_name = config.get('judge_llm',{}).get("name")
    judge_temp = config.get("judge_llm", {}).get('generation_options',{}).get("temperature", 0.0) 
    judge_has_reasoning = config.get('judge_llm',{}).get("has_reasoning", True) 
    judge_options = config.get("judge_llm", {}).get('generation_options',{})
    judge_system_prompt = config.get("judge_llm", {}).get('system_prompt')

    
    mELO_config = config.get("mELO", {}) 
    initial_rating = mELO_config.get("initial_rating", 1000.0)
    learning_rate = mELO_config.get("learning_rate", 0.5) 
    epochs = mELO_config.get("epochs", 300) 


    if not llm_runtime_api_url:
        logger.error("API base URL ('api_base_url') not configured.")
        return None
    if not judge_model_name:
        logger.error("Judge model not configured.")
        return None

    judge = JudgeLLM(
        api_url=llm_runtime_api_url,
        model_name=judge_model_name,
        temperature=judge_temp,
        has_reasoning=judge_has_reasoning,
        options=judge_options,
        system_prompt=judge_system_prompt
    )
    
    overall_run_success = True
    
    dataset = load_prompt_dataset(config, category)
    
    logger.info(f"Starting comparisons for category '{category}'.")

    for prompt in dataset:
        cycle_success_flag_for_logging = False 
        prompt_processing_details = {"prompt_id": "N/A", "status": "Skipped", "pairs_processed_attempts": 0, "pairs_succeeded": 0}
        
        try:
            prompt_data_tuple = extract_prompt_info_from_dataset_item(config,dataset_item=prompt)

            if prompt_data_tuple is None:
                logger.error("Failed to retrieve a valid prompt/ground truth tuple. Skipping this cycle.")
                prompt_processing_details["status"] = "Failed (No Prompt)"
                overall_run_success = False
                continue 
            
            prompt_content, ground_truth, prompt_id = prompt_data_tuple
            prompt_processing_details["prompt_id"] = prompt_id
            logger.info(f"Retrieved prompt ID: {prompt_id}")

            category_specific_output_dir = os.path.join(base_output_dir, str(category)) # Ensure string
            # create file name for prompt
            safe_prompt_id_filename = "".join(c if c.isalnum() or c in ['_','-'] else '_' for c in str(prompt_id)) + ".json"
            output_file_path = os.path.join(category_specific_output_dir, safe_prompt_id_filename)

            logger.info(f"Loading existing answers for prompt ID {prompt_id} from: {output_file_path}")
            existing_answers = load_json_file(output_file_path) 

            if not isinstance(existing_answers, list) or len(existing_answers) < 2:
                logger.warning(f"Not enough existing answers (found {len(existing_answers) if isinstance(existing_answers, list) else 0}, need at least 2) for prompt ID {prompt_id}. Skipping comparisons for this prompt.")
                prompt_processing_details["status"] = "Skipped (Not enough answers)"
                continue

            indices = list(range(len(existing_answers)))
            # Create list of combinations
            pair_indices_iterator = list(combinations(indices, 2))
            
            if not pair_indices_iterator:
                logger.warning(f"No model pairs could be formed from the {len(existing_answers)} answers for prompt ID {prompt_id}. Skipping.")
                prompt_processing_details["status"] = "Skipped (No pairs formed)"
                continue

            logger.info(f"Found {len(existing_answers)} answers, forming {len(pair_indices_iterator)} possible unique pairs for prompt ID {prompt_id}.")
            
            num_successful_pairs_for_prompt = 0
            # Inner loop: Random pairs 
            actual_comparisons_to_run = min(num_pair_comparisons_per_prompt, len(pair_indices_iterator))
            prompt_processing_details["pairs_processed_attempts"] = actual_comparisons_to_run

            logger.info(f"Attempting to process up to {actual_comparisons_to_run} random pairs for this prompt.")
            random.shuffle(pair_indices_iterator)
            
            for pair_selection_attempt in range(actual_comparisons_to_run):
                if not pair_indices_iterator: 
                    logger.info("No more unique pairs to compare for this prompt.")
                    break
                
                index1, index2 = pair_indices_iterator.pop(0) 

                selected_model_response_object_1 = existing_answers[index1]
                selected_model_response_object_2 = existing_answers[index2]

                model_a = selected_model_response_object_1.get("model")
                model_b = selected_model_response_object_2.get("model")
                response_a = selected_model_response_object_1.get("response")
                response_b = selected_model_response_object_2.get("response")
                
                if not all([model_a, model_b, response_a is not None, response_b is not None]):
                    logger.warning(f"Skipping pair for prompt {prompt_id} due to missing model name or answer: "
                                   f"M1({model_a}, ans_exists={response_a is not None}), M2({model_b}, ans_exists={response_b is not None})")
                    continue

                if model_a == model_b: 
                    logger.info(f"Skipping pair for prompt {prompt_id}: model_a and model_b are the same ('{model_a}').")
                    continue
                
                prompt_text_for_judge = _extract_prompt_text(prompt_content)
                if prompt_text_for_judge is None:
                    logger.error(f"Could not extract usable prompt text from data for prompt ID {prompt_id}: {prompt_content}")
                    prompt_processing_details["status"] = "Failed (Prompt Extraction Error)"
                    overall_run_success = False
                    break 

                logger.info(f"Comparing models for prompt ID {prompt_id}: '{model_a}' vs '{model_b}'. Attempt {pair_selection_attempt + 1}/{actual_comparisons_to_run}.")

                # Judge Responses (without Positionbias)
                verdict_1 = judge.evaluate(
                    user_prompt=prompt_text_for_judge, 
                    response_a=response_a,
                    response_b=response_b,
                    ground_truth=ground_truth
                )
                verdict_2 = judge.evaluate(
                    user_prompt=prompt_text_for_judge, 
                    response_a=response_b, # Switch Answers
                    response_b=response_a,
                    ground_truth=ground_truth
                )
                
                # 1.0 for A (first response), 0.0 for B (second response), 0.5 for Tie
                final_score_model_a = 0.5 
                if verdict_1 is None or verdict_2 is None:
                    logger.warning(f"Judging failed for one or both positions for prompt {prompt_id}, pair ({model_a} vs {model_b}). Skipping ELO update for this pair.")
                    continue 

                if verdict_1 != verdict_2 and verdict_1 != 0.5 and verdict_2 != 0.5:
                    final_score_model_a = verdict_1 
              
                logger.info(f"Judgement for prompt {prompt_id}, pair ({model_a} vs {model_b}): Verdict1 (A vs B)={verdict_1}, Verdict2 (B vs A)={verdict_2}. Final score for A = {final_score_model_a}")
                match_data: MatchType = (model_a, model_b, category, final_score_model_a)
                all_matches_for_category.append(match_data)
            
            # Update status
            prompt_processing_details["pairs_succeeded"] = num_successful_pairs_for_prompt
            if prompt_processing_details["pairs_processed_attempts"] > 0:
                if num_successful_pairs_for_prompt == prompt_processing_details["pairs_processed_attempts"]:
                    logger.info(f"Successfully processed all {prompt_processing_details['pairs_processed_attempts']} attempted model pairs for prompt {prompt_id}.")
                    prompt_processing_details["status"] = "Success"
                    cycle_success_flag_for_logging = True
                elif num_successful_pairs_for_prompt > 0:
                    logger.warning(f"Processed {num_successful_pairs_for_prompt}/{prompt_processing_details['pairs_processed_attempts']} model pairs for prompt {prompt_id}. Some pairs may have failed or were skipped.")
                    prompt_processing_details["status"] = f"Partial Success ({num_successful_pairs_for_prompt}/{prompt_processing_details['pairs_processed_attempts']})"
                    cycle_success_flag_for_logging = True 
                else:
                    logger.error(f"Failed to process any model pairs for prompt {prompt_id} out of {prompt_processing_details['pairs_processed_attempts']} attempts.")
                    prompt_processing_details["status"] = "Failed (All pairs failed)"
                    overall_run_success = False
            else: 
                logger.info(f"No pair processing was attempted for prompt {prompt_id} (e.g. not enough answers or pairs).")

    
        except Exception as e:
            logger.exception(f"An unexpected error terminated the comparison cycle for prompt ID {prompt_processing_details.get('prompt_id', 'N/A')}: {e}")
            prompt_processing_details["status"] = "Failed (Exception)"
            overall_run_success = False

        finally:
            log_status = "Finished" if cycle_success_flag_for_logging or prompt_processing_details["status"].startswith("Skipped") else "Failed"
            logger.info(f"Details for prompt ID '{prompt_processing_details['prompt_id']}': Status: {prompt_processing_details['status']}, Attempted Pairs: {prompt_processing_details['pairs_processed_attempts']}, Succeeded Pairs: {prompt_processing_details['pairs_succeeded']}.")
            logger.info("=" * 50)
        
            
    # --- Batch m-ELO Update for the Category ---
    if not all_matches_for_category:
        logger.info(f"No matches were collected for category '{category}'. Skipping m-ELO update.")
        if not overall_run_success: return None 
        
        current_ratings_data = load_json_file(config["paths"]["results_file"], config["paths"]["lock_file"])
        return current_ratings_data.get("models") if isinstance(current_ratings_data, dict) else None

    logger.info(f"Collected {len(all_matches_for_category)} matches in total for category '{category}'. Starting m-ELO batch update.")
    
    current_elo_data_file_content = load_json_file(config["paths"]["results_file"], config["paths"]["lock_file"])
    mELO_model_ratings_store: ModelRatingsType = {}

    if isinstance(current_elo_data_file_content, dict) and isinstance(current_elo_data_file_content.get("models"), dict):
        mELO_model_ratings_store = current_elo_data_file_content.get("models", {})
        logger.info(f"Loaded existing model ratings from {config["paths"]["results_file"]}.")
    else:
        logger.warning(f"Could not load valid 'models' data from {config["paths"]["results_file"]} or file is empty/new. m-ELO will initialize ratings for category '{category}'.")
        current_elo_data_file_content = {} 

    calculate_mELO_ratings_by_gradient_descent(
        model_ratings=mELO_model_ratings_store,
        all_matches=all_matches_for_category,
        initial_rating=initial_rating,
        learning_rate=learning_rate,
        epochs=epochs,
    )

    logger.info(f"m-ELO batch update for category '{category}' completed.")


    save_elo_results(mELO_model_ratings_store, config)
    logger.info(f"Updated ELO ratings for category '{category}' saved to '{config["paths"]["results_file"]}'.")

    if not overall_run_success:
        logger.error(f"One or more errors occurred during the run for category '{category}'.")
    
    return mELO_model_ratings_store

