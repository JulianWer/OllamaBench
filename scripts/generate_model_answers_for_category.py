import logging
import os
from typing import Dict, Any, Optional, List, Union

from models.llms import LLM
from utils.get_data import load_prompt_dataset
from utils.file_operations import save_json_file, load_json_file

logger = logging.getLogger(__name__)

def _ensure_dir(file_path: str):
    """
    Ensures that the directory for a given file_path exists.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logger.info(f"Directory created: {directory}")
        except OSError as e:
            logger.error(f"Error creating directory {directory}: {e}")
            raise

def _prepare_prompt_for_llm(raw_prompt_content: Any) -> Union[str, List[Dict[str, str]], None]:
    """
    Prepares the raw prompt content into the format expected by the LLM.
    Handles strings, lists of strings, and lists of dictionaries.
    """
    if isinstance(raw_prompt_content, str):
        return raw_prompt_content
    elif isinstance(raw_prompt_content, list):
        # Already in the correct message format (e.g., [{'role': 'user', 'content': '...'}])
        if all(isinstance(item, dict) and "role" in item and "content" in item for item in raw_prompt_content):
            return raw_prompt_content
        # List of simple strings, will be converted to message format
        elif all(isinstance(item, str) for item in raw_prompt_content):
            return [{"role": "user", "content": text} for text in raw_prompt_content]
    
    logger.warning(f"Unsupported prompt content type: {type(raw_prompt_content)}. Skipping.")
    return None

def _process_single_prompt(
    prompt_entry: Dict[str, Any],
    model: LLM,
    output_dir: str,
    prompt_id_col: str,
    prompt_content_col: str,
) -> bool:
    """
    Generates and saves a response for a single prompt, avoiding
    regeneration if a response already exists.
    """
    prompt_id = prompt_entry.get(prompt_id_col)
    raw_prompt_content = prompt_entry.get(prompt_content_col)

    if prompt_id is None or raw_prompt_content is None:
        logger.warning(f"Skipping entry for model '{model.model_name}' due to missing prompt_id or content.")
        return False

    safe_prompt_id_filename = str(prompt_id).replace(os.sep, "_") + ".json"
    output_file_path = os.path.join(output_dir, safe_prompt_id_filename)

    # --- Efficiency Check: Avoid regenerating existing answers ---
    existing_answers = load_json_file(output_file_path) or []
    if any(ans.get("model") == model.model_name for ans in existing_answers if isinstance(ans, dict)):
        logger.info(f"Answer for prompt ID '{prompt_id}' by model '{model.model_name}' already exists. Skipping generation.")
        return True

    final_prompt_for_llm = _prepare_prompt_for_llm(raw_prompt_content)
    if final_prompt_for_llm is None:
        return False

    logger.debug(f"Generating response for prompt ID '{prompt_id}' with model '{model.model_name}'.")
    response_text = model.generate_response(prompt=final_prompt_for_llm)

    if response_text is None:
        logger.warning(f"Response generation from model '{model.model_name}' for prompt ID '{prompt_id}' failed.")
        return False

    new_response_entry = {
        "response": response_text,
        "model": model.model_name,
        "prompt_id": prompt_id
    }
    
    # Add the new response to the list of existing answers
    all_answers = [ans for ans in existing_answers if isinstance(ans, dict)]
    all_answers.append(new_response_entry)

    if save_json_file(all_answers, output_file_path):
        logger.info(f"Response for prompt '{prompt_id}' from '{model.model_name}' saved to '{output_file_path}'.")
        return True
    
    return False

def _process_prompts_for_model(
    model_name: str,
    dataset: Any,
    config: Dict[str, Any],
    category_output_dir: str,
    prompt_id_col: str,
    prompt_content_col: str,
):
    """
    Iterates through all prompts for a single model and orchestrates response generation.
    """
    logger.info(f"Processing prompts for model: '{model_name}' in category: '{os.path.basename(category_output_dir)}'")
    
    runtime_cfg = config.get("LLM_runtime", {})
    gen_opts = config.get("generation_options", {})
    
    model = LLM(
        api_url=runtime_cfg.get("api_base_url"),
        model_name=model_name,
        has_reasoning=config.get("LLMS_HAVE_REASONING", True),
        temperature=gen_opts.get("temperature", 0.0)
    )

    try:
        # Ensures the category-specific output directory exists
        _ensure_dir(os.path.join(category_output_dir, "dummy.file"))
    except Exception as e:
        logger.error(f"Cannot create/access directory '{category_output_dir}' for model '{model_name}'. Skipping. Error: {e}")
        return

    success_count, fail_count = 0, 0
    for prompt_entry in dataset:
        if _process_single_prompt(
            prompt_entry, model, category_output_dir,
            prompt_id_col, prompt_content_col
        ):
            success_count += 1
        else:
            fail_count += 1
    
    logger.info(f"Processing for model '{model_name}' complete. Succeeded/Skipped: {success_count}, Failed: {fail_count}.")

def generate_and_save_model_answers_for_category(config: Dict[str, Any], category: str):
    """
    Main orchestrator for generating model answers for all prompts in a category.
    """
    logger.info(f"--- Starting answer generation for category: '{category}' ---")

    models_to_run: Optional[List[str]] = config.get("COMPARISON_MODELS")
    if not models_to_run:
        logger.error("No models configured under 'COMPARISON_MODELS'. Aborting.")
        return

    dataset = load_prompt_dataset(config, category)
    if not dataset:
        logger.error(f"Could not load dataset for category '{category}'. Aborting generation task.")
        return
        
    # --- Correctly read column names from the configuration ---
    columns_cfg = config.get("dataset", {}).get("columns", {})
    prompt_id_col = columns_cfg.get("id")
    prompt_content_col = columns_cfg.get("prompt")
    
    if not prompt_id_col or not prompt_content_col:
        logger.error("Column names for 'id' or 'prompt' are not configured in config.yaml under dataset -> columns. Aborting.")
        return

    base_output_dir = config.get("paths", {}).get("output_dir", "model_responses")
    category_output_dir = os.path.join(base_output_dir, category)
    logger.info(f"Output directory for this run: '{category_output_dir}'")

    for model_name in models_to_run:
        _process_prompts_for_model(
            model_name=model_name,
            dataset=dataset,
            config=config,
            category_output_dir=category_output_dir,
            prompt_id_col=prompt_id_col,
            prompt_content_col=prompt_content_col,
        )

    logger.info(f"--- Finished answer generation for category '{category}' ---")
