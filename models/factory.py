import logging
from typing import Dict, Any, Optional

from models.llms import LLM
from models.judge_llm import JudgeLLM
from utils.config import ConfigService

logger = logging.getLogger(__name__)

def create_llm(model_name: str, config_service: ConfigService) -> Optional[LLM]:
    """
    Factory function to create an instance of a comparison LLM.

    Args:
        model_name: The name/tag of the model to create.
        config_service: The configuration service instance.

    Returns:
        An initialized LLM instance or None if config is missing.
    """
    runtime_config = config_service.llm_runtime_config
    llm_config = config_service.comparison_llms_config

    api_url = runtime_config.get("api_base_url")
    if not api_url:
        logger.error("Cannot create LLM: 'api_base_url' is not configured.")
        return None

    temperature = llm_config.get("generation_options", {}).get("temperature", 0.0)
    has_reasoning = llm_config.get("has_reasoning", True)
    
    logger.debug(f"Creating LLM instance for '{model_name}' with temp={temperature}, reasoning={has_reasoning}.")
    return LLM(
        api_url=api_url,
        model_name=model_name,
        has_reasoning=has_reasoning,
        temperature=temperature
    )

def create_judge_llm(config_service: ConfigService) -> Optional[JudgeLLM]:
    """
    Factory function to create an instance of the Judge LLM.

    Args:
        config_service: The configuration service instance.

    Returns:
        An initialized JudgeLLM instance or None if config is missing.
    """
    runtime_config = config_service.llm_runtime_config
    judge_config = config_service.judge_llm_config

    api_url = runtime_config.get("api_base_url")
    model_name = judge_config.get("name")

    if not api_url or not model_name:
        logger.error("Cannot create JudgeLLM: 'api_base_url' or judge 'name' is not configured.")
        return None

    temperature = judge_config.get("generation_options", {}).get("temperature", 0.0)
    has_reasoning = judge_config.get("has_reasoning", False)
    options = judge_config.get("generation_options", {})
    system_prompt = judge_config.get('system_prompt')

    logger.debug(f"Creating JudgeLLM instance for '{model_name}' with temp={temperature}, reasoning={has_reasoning}.")
    return JudgeLLM(
        api_url=api_url,
        model_name=model_name,
        temperature=temperature,
        has_reasoning=has_reasoning,
        options=options,
        system_prompt=system_prompt
    )