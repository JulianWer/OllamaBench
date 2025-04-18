import random
import logging
from typing import List, Optional, Dict, Tuple, Any
import json

import requests

from utils.chat_with_ollama import chat_with_ollama


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Setup basic logging

def get_installed_models_api() -> List[str]:
     try:
         response = requests.get('http://localhost:11434/api/tags')
         response.raise_for_status()
         models_data = response.json().get('models', [])
         model_names = [model['name'] for model in models_data]
         logger.info(f"Found installed models via API: {model_names}")
         return model_names
     except Exception as e:
         logger.error(f"Failed to get installed Ollama models via API: {e}", exc_info=True)
         return []

def get_two_random_models() -> Optional[Tuple[str, str]]:
    models = ['qwen2.5:7b','mistral:7b','llama3.2:3b','llama3.1:8b']
    if len(models) < 2:
        logger.warning(f"Cannot select two distinct models, less than two models available ({len(models)} found).")
        return None
    try:
        model_a, model_b = random.sample(models, 2)
        logger.info(f"Selected random models for comparison: '{model_a}' and '{model_b}'")
        return model_a, model_b
    except ValueError:
        logger.error("Error selecting two random models despite having enough models.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error selecting random models: {e}", exc_info=True)
        return None


def generate_response(model: str, message: str, system_prompt: Optional[str] = None) -> Optional[str]:
    messages = []
    if system_prompt:
        if not isinstance(system_prompt, str):
             logger.warning(f"System prompt for model '{model}' is not a string: {type(system_prompt)}. Skipping.")
        else:
             messages.append({"role": "system", "content": system_prompt})

    if not isinstance(message, str):
        logger.error(f"User message for model '{model}' is not a string: {type(message)}. Cannot generate response.")
        return None
    messages.append({"role": "user", "content": message})

    logger.debug(f"Generating response from model '{model}' using chat_with_ollama with messages: {json.dumps(messages)}")

    try:
        response_content = chat_with_ollama(
            messages=messages,
            model=model,
            stream=False,
            options={"temperature": 0.0}
        )
        if response_content is None:
            logger.warning(f"chat_with_ollama returned None for model '{model}'. Check previous logs for details.")
            return None

        logger.debug(f"Successfully received response from '{model}' via chat_with_ollama.")
        return response_content

    except Exception as e:
        logger.error(f"Unexpected error calling chat_with_ollama for model '{model}': {e}", exc_info=True)
        return None


def generate_responses_sequentially(
    model_list: List[str],
    message: str,
    config: Dict[str, Any],
    system_prompt: Optional[str] = None
) -> Dict[str, Optional[str]]:
    responses = {}
    logger.info(f"Running generate_responses_sequentially for models: {model_list}")

    if not model_list:
        logger.warning("generate_responses_sequentially called with empty model_list.")
        return {}

    for model in model_list:
        logger.info(f"Generating response for model '{model}' sequentially...")
        try:
            result = generate_response(model, message, system_prompt)
            responses[model] = result 
            if result is None:
                 logger.warning(f"Sequential generation failed or returned None for model '{model}'.")
            else:
                 logger.info(f"Successfully generated response for '{model}' sequentially.")
        except Exception as e:
            logger.error(f"Exception occurred while sequentially generating response for model '{model}': {e}", exc_info=True)
            responses[model] = None 

    return responses