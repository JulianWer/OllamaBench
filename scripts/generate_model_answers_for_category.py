import logging
import os
from typing import Dict, Any, List

from models.factory import create_llm
from models.llms import LLM
from utils.get_data import load_prompt_dataset, prepare_prompt_for_llm
from utils.file_operations import save_json_file, load_json_file
from utils.config import ConfigService

logger = logging.getLogger(__name__)

def _ensure_dir(file_path: str):
    """Ensures that the directory for a given file_path exists."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logger.info(f"Directory created: {directory}")
        except OSError as e:
            logger.error(f"Error creating directory {directory}: {e}")
            raise

def _process_single_prompt(
    prompt_entry: Dict[str, Any],
    model: LLM,
    output_file_path: str,
    prompt_id_col: str,
    prompt_content_col: str,
) -> bool:
    """
    Generates and saves a response for a single prompt, avoiding regeneration if it exists.
    """
    prompt_id = prompt_entry.get(prompt_id_col)
    raw_prompt_content = prompt_entry.get(prompt_content_col)

    if prompt_id is None or raw_prompt_content is None:
        logger.warning(f"Skipping entry for model '{model.model_name}' due to missing prompt ID or content.")
        return False

    # Efficiency Check: Avoid regenerating existing answers
    existing_answers = load_json_file(output_file_path) or []
    if any(ans.get("model") == model.model_name for ans in existing_answers if isinstance(ans, dict)):
        logger.info(f"Answer for prompt ID '{prompt_id}' by model '{model.model_name}' already exists. Skipping.")
        return True

    final_prompt_for_llm = prepare_prompt_for_llm(raw_prompt_content)
    if final_prompt_for_llm is None:
        return False

    logger.debug(f"Generating response for prompt ID '{prompt_id}' with model '{model.model_name}'.")
    response_text = model.generate_response(prompt=final_prompt_for_llm)

    if response_text is None:
        logger.warning(f"Failed to generate response from '{model.model_name}' for prompt ID '{prompt_id}'.")
        return False

    new_response_entry = {"response": response_text, "model": model.model_name, "prompt_id": prompt_id}
    
    all_answers = [ans for ans in existing_answers if isinstance(ans, dict)]
    all_answers.append(new_response_entry)

    if save_json_file(all_answers, output_file_path):
        logger.info(f"Response for prompt '{prompt_id}' from '{model.model_name}' saved to '{output_file_path}'.")
        return True
    
    return False

def _process_prompts_for_model(
    model_name: str,
    dataset: List[Dict[str, Any]],
    config_service: ConfigService,
    category_output_dir: str,
):
    """Iterates through all prompts for a single model and orchestrates response generation."""
    logger.info(f"Processing prompts for model: '{model_name}' in category: '{os.path.basename(category_output_dir)}'")
    
    model = create_llm(model_name, config_service)
    if not model:
        logger.error(f"Could not create LLM for '{model_name}'. Skipping.")
        return

    dataset_config = config_service.dataset_config
    prompt_id_col = dataset_config.get("columns", {}).get("id")
    prompt_content_col = dataset_config.get("columns", {}).get("prompt")

    try:
        _ensure_dir(os.path.join(category_output_dir, "dummy.file"))
    except Exception as e:
        logger.error(f"Cannot create/access directory '{category_output_dir}' for model '{model_name}'. Skipping. Error: {e}")
        return

    success_count, fail_count = 0, 0
    for prompt_entry in dataset:
        safe_prompt_id = str(prompt_entry.get(prompt_id_col, 'unknown_id')).replace(os.sep, "_")
        output_file_path = os.path.join(category_output_dir, f"{safe_prompt_id}.json")

        if _process_single_prompt(
            prompt_entry, model, output_file_path,
            prompt_id_col, prompt_content_col
        ):
            success_count += 1
        else:
            fail_count += 1
    
    logger.info(f"Processing for model '{model_name}' complete. Succeeded/Skipped: {success_count}, Failed: {fail_count}.")

def generate_and_save_model_answers_for_category(config: Dict[str, Any], category: str):
    """Main orchestrator for generating model answers for all prompts in a category."""
    logger.info(f"--- Starting answer generation for category: '{category}' ---")
    
    config_service = ConfigService()
    config = config_service.get_full_config() # Use the full config dict for legacy compatibility if needed

    models_to_run = config_service.comparison_llms_config.get("names")
    if not models_to_run:
        logger.error("No models configured under 'comparison_llms'. Aborting.")
        return

    dataset = load_prompt_dataset(config, category)
    if not dataset:
        logger.error(f"Could not load dataset for category '{category}'. Aborting.")
        return

    base_output_dir = config_service.get_path("output_dir")
    if not base_output_dir:
        logger.error("Output directory 'output_dir' not configured in paths. Aborting.")
        return
        
    category_output_dir = os.path.join(base_output_dir, category)
    logger.info(f"Output directory for this run: '{category_output_dir}'")

    for model_name in models_to_run:
        _process_prompts_for_model(
            model_name=model_name,
            dataset=dataset,
            config_service=config_service,
            category_output_dir=category_output_dir,
        )

    logger.info(f"--- Finished answer generation for category '{category}' ---")