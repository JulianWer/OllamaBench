import random
import logging
import json
import requests
import time
from typing import List, Optional, Dict, Tuple, Any, Iterator, Union

# Project specific imports
from utils.chat_with_ollama import chat_with_ollama # Use the enhanced chat function

logger = logging.getLogger(__name__)

# --- Constants ---
FALLBACK_MODELS = ['llama3:latest', 'mistral:latest']
AVAILABLE_MODELS = ['qwen2.5:7b', 'mistral:7b', 'llama3.2:3b', 'llama3.1:8b', 'phi4-mini:3.8b', 'gemma3:4b', 'deepcoder:1.5b', 'deepscaler:1.5b']


# --- Model Discovery ---
def get_installed_models(config: Dict[str, Any]) -> List[str]:
    """
    Retrieves the list of installed models from the Ollama API.
    Uses configuration for API endpoint details. Falls back to a default list on error.
    """
    ollama_config = config.get("ollama", {})
    base_url = ollama_config.get("api_base_url", "http://localhost:11434")
    tags_path = ollama_config.get("tags_api_path", "/api/tags")
    timeout = ollama_config.get("default_timeout", 30)

    api_url = f"{base_url.rstrip('/')}{tags_path}"
    logger.info(f"Attempting to retrieve installed models from Ollama API: {api_url}")

    try:
        response = requests.get(api_url, timeout=timeout)
        response.raise_for_status()
        models_data = response.json()

        if "models" in models_data and isinstance(models_data["models"], list):
            model_names = AVAILABLE_MODELS
            if model_names:
                logger.info(f"Successfully retrieved {len(model_names)} installed models from API: {model_names}")
                return model_names
            else:
                logger.warning("Ollama API returned 'models' list, but it was empty or contained no names.")
                return FALLBACK_MODELS
        else:
             logger.warning(f"Unexpected structure in response from Ollama API ({api_url}). Response: {models_data}")
             return FALLBACK_MODELS

    except requests.exceptions.Timeout:
        logger.error(f"Timeout error while retrieving models from Ollama API ({api_url}). Using fallback list.")
        return FALLBACK_MODELS
    except requests.exceptions.ConnectionError:
         logger.error(f"Connection error while retrieving models from Ollama API ({api_url}). Using fallback list.")
         return FALLBACK_MODELS
    except requests.exceptions.RequestException as e:
        logger.error(f"Error retrieving models from Ollama API ({api_url}): {e}. Using fallback list.")
        return FALLBACK_MODELS
    except json.JSONDecodeError:
         logger.error(f"Failed to decode JSON response from Ollama model list API ({api_url}). Using fallback list.")
         return FALLBACK_MODELS
    except Exception as e:
        logger.exception(f"An unexpected error occurred while retrieving models from Ollama API: {e}. Using fallback list.")
        return FALLBACK_MODELS

def get_two_random_models(config: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    Selects two distinct random models from the list of installed models.
    Fetches the list from the API via get_installed_models().
    """
    model_pool = get_installed_models(config)

    if not model_pool or len(model_pool) < 2:
        logger.error(f"Cannot select two models: Only {len(model_pool)} model(s) available/found ({model_pool}).")
        return None

    try:
        model_a, model_b = random.sample(model_pool, 2)
        logger.info(f"Selected random models for comparison: '{model_a}' vs '{model_b}'")
        return model_a, model_b
    except ValueError:
        logger.error(f"Failed to sample 2 models from the pool (size {len(model_pool)}): {model_pool}")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred during random model selection: {e}")
        return None

# --- Response Generation ---
def generate_response(
    model: str,
    prompt: Union[str, List[Dict[str, str]]], # Accept string or 'turns' list
    config: Dict[str, Any],
    system_prompt: Optional[str] = None,
    stream_output: bool = False, # Default to False unless live output needed
) -> Optional[str]:
    """
    Generates a response from a specified Ollama model using the chat API.

    Args:
        model: The name of the Ollama model to use.
        prompt: The user prompt. Expected to be cleaned: either a string or a list
                of message dictionaries like {"role": ..., "content": ...}.
        config: The application configuration dictionary.
        system_prompt: An optional system prompt to guide the model's behavior.
        stream_output: If True, prints the response chunks to the console as they arrive.
                       The function still returns the complete response string.

    Returns:
        The complete generated response content as a string, or None if generation fails.
    """
    if not model: logger.error("Cannot generate response: Model name is empty."); return None
    if not prompt: logger.error(f"Cannot generate response for model '{model}': Prompt is empty."); return None

    # --- Prepare Messages ---
    messages: List[Dict[str, str]] = []
    if system_prompt and isinstance(system_prompt, str): messages.append({"role": "system", "content": system_prompt})
    elif system_prompt: logger.warning(f"System prompt for model '{model}' ignored (expected string, got {type(system_prompt)}).")

    # Handle different prompt formats
    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    elif isinstance(prompt, list):
        # **REVISED VALIDATION:**
        # Trust that the prompt list was cleaned by get_random_prompt_details.
        # Perform a basic check that all items are dictionaries.
        is_list_of_dicts = all(isinstance(turn, dict) for turn in prompt)

        if is_list_of_dicts:
            # Assume structure like {"role": ..., "content": ...} is present due to prior cleaning.
            # Filter out any potential empty dicts just in case? Optional.
            valid_turns = [turn for turn in prompt if turn] # Removes empty dicts if any
            if len(valid_turns) != len(prompt):
                 logger.warning(f"Found empty dictionaries within the prompt list for model '{model}'. Filtering them out.")

            if not valid_turns: # Check if list became empty after filtering
                 logger.error(f"Multi-turn prompt for model '{model}' became empty after filtering potentially empty dicts. Original length: {len(prompt)}.")
                 return None

            messages.extend(valid_turns)
            logger.debug(f"Using multi-turn prompt ({len(valid_turns)} turns) for model '{model}'.")
        else:
            # This case indicates a problem upstream (cleaning failed or wasn't done)
            logger.error(f"Invalid multi-turn prompt format provided to generate_response for model '{model}'. Expected List[Dict[str, str]], but list contained non-dict items. Prompt (structure): {[type(p) for p in prompt]}")
            return None
    else:
        logger.error(f"Invalid prompt type for model '{model}'. Expected str or List[Dict[str, str]], got {type(prompt)}.")
        return None

    # --- Get API and Generation Options from Config ---
    ollama_config = config.get("ollama", {})
    api_base_url = ollama_config.get("api_base_url")
    chat_api_path = ollama_config.get("chat_api_path")
    timeout = ollama_config.get("default_timeout", 120)
    max_retries = ollama_config.get("max_retries", 3)
    retry_delay = ollama_config.get("retry_delay", 5)
    generation_options = config.get("generation_options", {})
    if not isinstance(generation_options, dict):
         logger.warning(f"Config 'generation_options' is not a dictionary. Using empty options.")
         generation_options = {}

    logger.info(f"Generating response from model '{model}'...")

    full_response_content = ""
    response_stream = None

    try:
        # Call the central chat function, requesting a stream
        response_stream = chat_with_ollama(
            messages=messages,
            model=model,
            ollama_api_base_url=api_base_url,
            ollama_chat_api_path=chat_api_path,
            stream=True, # Always stream
            options=generation_options,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay
        )

        if response_stream is None:
            logger.error(f"Failed to get response stream from model '{model}' after retries.")
            return None

        # Process the stream
        if stream_output: print(f"\n--- Live Output from {model} ---")
        start_time = time.monotonic()
        last_chunk = None # Keep track of the last chunk for 'done' flag

        for chunk in response_stream:
            last_chunk = chunk # Store the current chunk
            if isinstance(chunk, dict) and "error" in chunk:
                 logger.error(f"Error received during stream from model '{model}': {chunk['error']}")
                 if stream_output: print(f"\n[STREAM ERROR: {chunk['error']}]")
                 return None # Indicate failure

            content_part = chunk.get('message', {}).get('content', '')
            if content_part:
                full_response_content += content_part
                if stream_output: print(content_part, end='', flush=True)

            # Check for the 'done' flag in the chunk
            if chunk.get('done'):
                done_reason = chunk.get('done_reason', 'unknown')
                logger.debug(f"Stream for model '{model}' marked as done. Reason: {done_reason}")
                break

        duration = time.monotonic() - start_time
        if stream_output: print("\n--- End Live Output ---")

        # Check if any content was received, even if 'done' flag might be missing in last chunk
        if not full_response_content and (last_chunk is None or not last_chunk.get('done')):
             logger.warning(f"Stream for model '{model}' ended without 'done' flag and no content received.")

        logger.info(f"Successfully generated response from model '{model}' (Duration: {duration:.2f}s). Length: {len(full_response_content)} chars.")
        return full_response_content.strip()

    except Exception as e:
        logger.exception(f"An unexpected error occurred while processing the response stream for model '{model}': {e}")
        return None
    finally:
        pass


def generate_responses_sequentially(
    model_list: List[str],
    prompt: Union[str, List[Dict[str, str]]],
    config: Dict[str, Any],
    system_prompt: Optional[str] = None
) -> Dict[str, Optional[str]]:
    """
    Generates responses from a list of models sequentially, one after the other.
    """
    responses: Dict[str, Optional[str]] = {}
    if not model_list: logger.warning("generate_responses_sequentially called with empty model list."); return {}

    logger.info(f"Generating responses sequentially for models: {model_list}")

    for model_name in model_list:
        logger.info(f"--- Generating for model: {model_name} ---")
        response = generate_response(
            model=model_name,
            prompt=prompt,
            config=config,
            system_prompt=system_prompt,
            stream_output=True # Show live output for each model
        )
        responses[model_name] = response
        if response is None: logger.warning(f"Failed to generate response for model '{model_name}' in sequential run.")

    logger.info(f"Finished sequential generation for {len(model_list)} models.")
    return responses
