# OllamaBench_01/models/llms.py
"""
Defines the LLM class for interacting with different language models via Ollama.
"""

import logging
from typing import Any, Dict, Optional

# Relative import from sibling directory 'utils'
from utils.chat_with_ollama import chat_with_ollama

logger = logging.getLogger(__name__)


class LLM:
    """
    Represents a language model accessible via the Ollama API.
    """

    def __init__(
        self,
        api_url: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the LLM instance.

        Args:
            api_url: The base URL of the Ollama API.
            model_name: The name of the language model to use.
            system_prompt: An optional default system prompt for this model.
            temperature: The default temperature setting for generation.
            options: Default additional Ollama options for this model.
        """
        self.api_url = api_url
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.options = options if options else {}
        logger.info(f"LLM instance created for model: {self.model_name} at {self.api_url}")

    def generate_response(
        self,
        prompt: str,
        system_message_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
        options_override: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Generates a response from the LLM for a given prompt.

        Allows overriding default settings for this specific call.

        Args:
            prompt: The user's prompt.
            system_message_override: A system message to use for this call,
                                     overriding the instance default.
            temperature_override: A temperature value to use for this call.
            options_override: Ollama options to use for this call, potentially
                              merging with or replacing instance defaults.

        Returns:
            The generated text response as a string, or None if generation fails.
        """
        system_msg = system_message_override if system_message_override is not None else self.system_prompt
        temp = temperature_override if temperature_override is not None else self.temperature
        # Combine options: override takes precedence
        current_options = self.options.copy()
        if options_override:
            current_options.update(options_override)

        logger.info(f"Generating response using model {self.model_name} with prompt: '{prompt[:50]}...'")

        api_response = chat_with_ollama(
            api_url=self.api_url,
            model=self.model_name,
            prompt=prompt,
            system_message=system_msg,
            temperature=temp,
            options=current_options,
            request_timeout=600,
            stream=False,
        )

        if api_response and "message" in api_response and "content" in api_response["message"]:
            response_content = api_response["message"]["content"]
            logger.debug(f"Response received from {self.model_name}: '{response_content[:100]}...'")
            return response_content
        else:
            logger.error(f"Failed to generate response from model {self.model_name}.")
            return None

