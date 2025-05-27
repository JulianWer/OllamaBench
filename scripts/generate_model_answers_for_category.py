import logging
import os
from typing import Dict, Any, Optional, List, Union

from models.llms import LLM
from utils.get_data import load_prompt_dataset
from utils.file_operations import _deep_merge_dicts, save_json_file, load_json_file

logger = logging.getLogger(__name__)

def _ensure_dir(file_path: str):
    """
    Ensures that the directory for a given file_path exists.
    If file_path includes a filename, it ensures the directory containing the file exists.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        except OSError as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise 

def _load_and_validate_dataset(config: Dict[str, Any], category: str) -> Optional[Any]: 
    """
    Loads the dataset for the given category and validates essential columns.
    Returns the loaded dataset or None if loading/validation fails.
    """
    logger.info(f"Attempting to load dataset for category '{category}'...")

    dataset = load_prompt_dataset(config, category)
    if dataset is None:
        logger.error(f"Failed to load dataset for category '{category}'.")
        return None

    try:
        from utils.get_data import get_active_category_column, get_active_prompt_id_column, get_active_prompt_column
    except ImportError:
        logger.error("Could not import get_active_category_column, get_active_prompt_id_column from utils.get_data.")
        return None

    logger.info(f"Successfully loaded dataset '{get_active_category_column()}' for '{category}'. It contains {len(dataset)} prompts.")

    # Validate necessary columns
    prompt_id_column_name = get_active_prompt_id_column()
    actual_prompt_content_column_name = get_active_prompt_column() 

    if actual_prompt_content_column_name not in dataset.column_names:
        logger.error(f"Prompt content column '{actual_prompt_content_column_name}' not found in dataset. Available: {dataset.column_names}.")
        return None
    if prompt_id_column_name not in dataset.column_names:
        logger.error(f"Prompt ID column '{prompt_id_column_name}' not found in dataset. Available: {dataset.column_names}.")
        return None
    
    return dataset

def _prepare_prompt_for_llm(raw_prompt_content: Any, prompt_id: Any, model_name: str, content_column_name: str) -> Union[str, List[Dict[str, str]], None]:
    """
    Prepares the raw prompt content into the format expected by `generate_response`.
    Handles string prompts and lists of strings (interpreted as user turns).
    Returns the formatted prompt or None if preparation fails.
    """
    if isinstance(raw_prompt_content, str):
        return raw_prompt_content
    elif isinstance(raw_prompt_content, list):
        if all(isinstance(item, str) for item in raw_prompt_content):
            if raw_prompt_content:
                return [{"role": "user", "content": text_turn} for text_turn in raw_prompt_content]
            else:
                logger.warning(f"Prompt_id '{prompt_id}' for model '{model_name}': Column '{content_column_name}' is an empty list. Skipping.")
                return None
        elif all(isinstance(item, dict) and "role" in item and "content" in item for item in raw_prompt_content):
            return raw_prompt_content # Already in correct format
        else:
            logger.warning(f"Prompt_id '{prompt_id}' for model '{model_name}': Column '{content_column_name}' has list with unsupported structure. Skipping. Data: {raw_prompt_content}")
            return None
    else:
        logger.warning(f"Prompt_id '{prompt_id}' for model '{model_name}': Content in '{content_column_name}' is of unsupported type {type(raw_prompt_content)}. Skipping.")
        return None

# --- Single Prompt Processing ---
def _process_single_prompt(
    prompt_entry: Dict[str, Any],
    model:LLM,
    model_specific_output_dir: str,
    prompt_id_col: str,
    prompt_content_col: str,
) -> bool:
    """
    Generates a response for a single prompt entry. If a file for this prompt_id
    already exists, it appends the new answer to a list of answers. Otherwise,
    it creates a new list with the current answer.
    The result is saved as a JSON file.
    Returns True if successful, False otherwise.
    """
    prompt_id = prompt_entry.get(prompt_id_col)
    raw_prompt_content = prompt_entry.get(prompt_content_col)

    if prompt_id is None:
        logger.warning(f"Skipping prompt for model '{model.model_name}' due to missing '{prompt_id_col}'. Data: {prompt_entry}")
        return False
    
    # Create file Path to save
    safe_prompt_id_filename = str(prompt_id).replace(os.sep, "_") + ".json"
    output_file_path = os.path.join(model_specific_output_dir, safe_prompt_id_filename)


    if raw_prompt_content is None:
        logger.warning(f"Skipping prompt_id '{prompt_id}' for model '{model.model_name}' due to missing content in '{prompt_content_col}'.")
        return False

    final_prompt_for_llm = _prepare_prompt_for_llm(raw_prompt_content, prompt_id, model.model_name, prompt_content_col)
    
    if final_prompt_for_llm is None:
        logger.warning(f"Prompt_id '{prompt_id}' for model '{model.model_name}': Final prompt material is None after processing. Skipping.")
        return False

    logger.debug(f"Generating response for prompt_id '{prompt_id}' with model '{model.model_name}'.")
    response_text = model.generate_response(
        prompt=final_prompt_for_llm,
    )

    if response_text is not None:
        new_response_entry = {
            "response": response_text,
            "model": model.model_name,
            "prompt_id": prompt_id 
        }
        
        existing_answers = load_json_file(output_file_path) 
        is_answer_replaced = False
        all_answers = []
        if isinstance(existing_answers, list):
            all_answers.extend(existing_answers)
            logger.debug(f"Loaded {len(existing_answers)} existing answers for prompt_id '{prompt_id}'.")
        elif existing_answers is not None:
            # If the file exists but is not a list (e.g., old format, single dict)
            # wrap it in a list.
            logger.warning(f"Existing answer file for prompt_id '{prompt_id}' is not a list. Wrapping it. Content: {type(existing_answers)}")
            all_answers.append(existing_answers) # This might need adjustment based on actual old format

        
        for i, existing_answer in enumerate(all_answers):
            if isinstance(existing_answer, dict) and \
            existing_answer.get("model") == new_response_entry["model"] and \
            existing_answer.get("prompt_id") == new_response_entry["prompt_id"]:
                
                logger.info(f"Replacing existing answer for model '{new_response_entry['model']}' and prompt_id '{new_response_entry['prompt_id']}'.")
                all_answers[i] = new_response_entry  
                is_answer_replaced = True
                break
        if not is_answer_replaced:
            all_answers.append(new_response_entry)
        
        
        if save_json_file(all_answers, output_file_path, lock_file_path=None): 
            logger.info(f"Saved/Appended JSON response for prompt_id '{prompt_id}' from '{model.model_name}' to '{output_file_path}'. Total answers: {len(all_answers)}.")
            return True
        else:
            pass 
    else:
        logger.warning(f"Failed to generate response from model '{model.model_name}' for prompt_id '{prompt_id}'.")
    
    return False

# --- Process All Prompts for a Single Model ---
def _process_prompts_for_model(
    model_name: str,
    dataset: Any,
    config: Dict[str, Any],
    category_specific_output_dir: str,
    prompt_id_col: str,
    prompt_content_col: str,
):
    """
    Iterates through all prompts in the dataset for a single model,
    generating and saving/appending responses.
    """
    logger.info(f"Processing prompts for model: '{model_name}' in category directory '{category_specific_output_dir}'")
    llm_runtime_api_url = config.get("LLM_runtime", {}).get("api_base_url", {})
    model_temp = config.get("generation_options", {}).get("temperature", 0.0)
    model_has_reasoning = config.get("LLMS_HAVE_REASONING", True)



    model = LLM(api_url=llm_runtime_api_url,model_name=model_name,has_reasoning=model_has_reasoning, temperature=model_temp)
    try:
        # Ensure the base directory for this category's responses exists.
        # The file name part is not strictly necessary here if category_specific_output_dir is already just the dir.
        _ensure_dir(os.path.join(category_specific_output_dir, "dummy.txt")) 
    except Exception as e:
        logger.error(f"Cannot create/access directory {category_specific_output_dir} for model {model_name}. Skipping. Error: {e}")
        return

    prompts_processed_count = 0
    prompts_failed_count = 0

    for prompt_entry in dataset: 
        # Note: The third argument to _process_single_prompt is the output directory for this category.
        # Individual prompt files (e.g. <prompt_id>.json) will be created inside this directory.
        if _process_single_prompt(
            prompt_entry,model, category_specific_output_dir,
            prompt_id_col, prompt_content_col
        ):
            prompts_processed_count += 1
        else:
            prompts_failed_count += 1
    
    logger.info(f"Model '{model_name}' processing complete for category. Successful: {prompts_processed_count}, Failed: {prompts_failed_count}.")

# --- Main Orchestrator Function ---
def generate_and_save_model_answers_for_category(config: Dict[str, Any], category: str):
    """
    For each model specified in the configuration, this function generates 
    answers for every prompt found in the dataset corresponding to the given category.
    Responses are saved into individual JSON files named after their respective prompt_id
    within a category-specific directory. If a file for a prompt_id already exists,
    the new answer is appended to a list of answers in that file.
    """
    logger.info(f"Starting JSON answer generation process for category: '{category}'")

    # Get Model Configuration
    models_to_run: Optional[List[str]] = config.get("COMPARISON_MODELS")
    if not models_to_run:
        logger.error("No models in config under 'COMPARISON_MODELS'. Aborting.")
        return

    # Load and Validate Dataset
    dataset = _load_and_validate_dataset(config, category)
    if dataset is None:
        return 

    from utils.get_data import get_active_prompt_column, get_active_prompt_id_column
    prompt_id_column_name = get_active_prompt_id_column()
    actual_prompt_content_column_name = get_active_prompt_column()
    
    # Define Base Output Directory
    paths_config = config.get("paths", {})
    base_output_dir = paths_config.get("output_dir", {})
    logger.info(f"Base directory for saving JSON responses: '{base_output_dir}'")
    # Each category will have its own subdirectory under base_output_dir
    category_specific_output_dir = os.path.join(base_output_dir, category)
    logger.info(f"Output directory for category '{category}': '{category_specific_output_dir}'")


    for model_name in models_to_run:
        _process_prompts_for_model(
            model_name,
            dataset,
            config,
            category_specific_output_dir,
            prompt_id_column_name,
            actual_prompt_content_column_name,
        )

    logger.info(f"All configured models processed for category '{category}'. JSON responses saved/appended.")
