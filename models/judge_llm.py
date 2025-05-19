# OllamaBench_01/models/judge_llm.py
"""
Defines the JudgeLLM class for evaluating and comparing model responses
using a specific prompt format that expects a simple verdict (A, B, or C).
"""

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple, Union

# Relative import from sibling directory 'utils'
# Ensure this path is correct based on your project structure
try:
    from ..utils.chat_with_ollama import chat_with_ollama
except ImportError:
    # Fallback for running script directly or different structure
    from utils.chat_with_ollama import chat_with_ollama


logger = logging.getLogger(__name__)


def _build_judge_prompt(
    prompt: str,
    response1: str,
    response2: str,
    ground_truth: Optional[str] = None,
    custom_system_prompt: Optional[str] = None, 
) -> str:
    """Constructs the prompt for the judge LLM."""
    
    judge_prompt_parts = [
        custom_system_prompt,
        "You are an automated evaluation system. Your SOLE task is to determine which AI assistant's response is better based on the user's query and potentially a ground truth answer.",
        "\n**Evaluation Criteria:**",
        "* Helpfulness and relevance to the user's query.",
        "* Accuracy and correctness of information.",
        "* Depth, detail, and creativity.",
        "* Adherence to instructions in the user query.",
    ]

    if ground_truth:
        judge_prompt_parts.extend([
            "* Closeness to the provided Ground Truth Answer (if applicable).",
            "\n**[Ground Truth Answer]**",
            ground_truth,
            "**[End of Ground Truth Answer]**"
        ])

    judge_prompt_parts.extend([
        "\n**[User Question]**",
        prompt,
        "**[The Start of Assistant A’s Answer]**",
        response1,
        "**[The End of Assistant A’s Answer]**",
        "**[The Start of Assistant B’s Answer]**",
        response2,
        "**[The End of Assistant B’s Answer]**",
        "\n**Output Format:**",
        "Your output MUST be exactly one of the following three options, and ABSOLUTELY NOTHING ELSE:",
        "* `[[A]]` if Assistant A provided the better response.",
        "* `[[B]]` if Assistant B provided the better response.",
        "* `[[C]]` if the responses are of comparable quality or tied.",
        "\n**IMPORTANT RULES:**",
        "* Do NOT provide any explanation or justification.",
        "* Do NOT include any text before or after the verdict (`[[A]]`, `[[B]]`, or `[[C]]`).",
        "* Be objective. Do not let response length, assistant names, or the order of presentation influence your decision.",
        "* If a Ground Truth Answer was provided, consider it in your evaluation, but the primary focus remains on answering the User Question effectively.",
        "* Your ENTIRE response must be ONLY `[[A]]` OR `[[B]]` OR `[[C]]`.",
        "\nNow, evaluate the provided query, responses (and ground truth, if available) and output your verdict in the required format."
    ])
    judge_prompt = "\n".join(filter(None, judge_prompt_parts))
    formatted_prompt = judge_prompt
    return formatted_prompt.strip()


# --- JudgeLLM Class ---

class JudgeLLM:
    """
    A class to use an LLM as a judge for comparing two model responses,
    expecting a simple verdict output ('[[A]]', '[[B]]', or '[[C]]').
    """

    def __init__(
        self,
        api_url: str,
        model_name: str,
        system_prompt: Optional[str] = None, 
        temperature: float = 0.0, 
        has_reasoning: Optional[bool] = False,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the JudgeLLM.

        Args:
            api_url: The base URL of the Ollama API.
            model_name: The name of the judge LLM to use.
            system_prompt: A custom system prompt header to override the default.
            temperature: The temperature for the judge model's generation.
            options: Additional Ollama options for the judge model.
        """
        self.api_url = api_url
        self.model_name = model_name
        self.system_prompt = system_prompt 
        self.temperature = temperature
        self.has_reasoning= has_reasoning
        self.options = options if options else {}
        logger.info(f"JudgeLLM initialized with model: {self.model_name} at {self.api_url}")
        logger.debug(f"Using Judge System Prompt Header: {self.system_prompt}")


    def _parse_judge_verdict(self,response_text: str) -> Optional[float]:
        """
        Parses the judge LLM's response to extract the verdict.
        Handles variations like extra whitespace or text around the verdict.

        Returns:
            1.0 for 'A', 0.0 for 'B', 0.5 for 'C', or None if parsing fails.
        """
        if not response_text:
            logger.warning("Judge response is empty.")
            return None

        cleaned_response = response_text.strip()

        # 1. Check for exact match first (most reliable)
        if cleaned_response == "[[A]]": return 1.0
        if cleaned_response == "[[B]]": return 0.0
        if cleaned_response == "[[C]]": return 0.5

        # 2. Use regex to find the pattern [[A]], [[B]], or [[C]], ignoring case and surrounding text
        # This regex looks for [[ followed by A, B, or C (case-insensitive), followed by ]]
        match = re.search(r'\[\[\s*(A|B|C)\s*\]\]', cleaned_response, re.IGNORECASE)
        if match:
            verdict = match.group(1).upper()
            logger.warning(f"Parsed verdict '{verdict}' using regex from potentially noisy response: '{cleaned_response[:100]}...'")
            if verdict == "A": return 1.0
            if verdict == "B": return 0.0
            if verdict == "C": return 0.5
            # Should not happen if regex is correct, but as a safeguard:
            logger.error(f"Regex matched but extracted unexpected verdict '{verdict}'.")
            return None

        logger.error(f"Failed to parse judge verdict. Could not find [[A]], [[B]], or [[C]] in response: '{cleaned_response[:200]}...'")
        return None


    def evaluate(
        self,
        user_prompt: str,
        response_a: str,
        response_b: str,
        ground_truth: Optional[str] = None 
    ) -> Optional[str]: 
        """
        Evaluates two responses using the judge LLM.

        Args:
            user_prompt: The original user prompt.
            response_a: The response from the first model.
            response_b: The response from the second model.
            ground_truth: Optional ground truth answer for comparison.

        Returns:
            The judge's verdict ('[[A]]', '[[B]]', or '[[C]]') as a string,
            or None if the evaluation fails or parsing fails.
        """
        judge_prompt = _build_judge_prompt(
            prompt=user_prompt,
            response1=response_a,
            response2=response_b,
            ground_truth=ground_truth,
            custom_system_prompt=self.system_prompt 
        )
        logger.debug(f"Constructed Judge Prompt:\n----\n{judge_prompt}\n----")

        logger.info(f"Requesting judgment from model {self.model_name}...")

        api_response = chat_with_ollama(
            api_url=self.api_url,
            model=self.model_name,
            prompt=judge_prompt,
            temperature=self.temperature,
            has_reasoning=self.has_reasoning,
            options=self.options,
            stream=False, 
        )

        # Check the response type (should be dict if stream=False)
        if not isinstance(api_response, dict):
             logger.error(f"Failed to get a valid dictionary response from the judge LLM API. Got type: {type(api_response)}")
             return None

        if "message" not in api_response or "content" not in api_response["message"]:
            logger.error(f"Judge LLM API response missing 'message' or 'content': {api_response}")
            return None

        raw_judgment_text = api_response["message"]["content"]
        logger.debug(f"Raw judgment response received: {raw_judgment_text}")

        # Parse the raw text for the verdict using the updated parser
        parsed_verdict = self._parse_judge_verdict(raw_judgment_text)

        if parsed_verdict is not None:
            verdict_map = {1.0: "A", 0.0: "B", 0.5: "C"}
            logger.info(f"Judgment received': [[{verdict_map[parsed_verdict]}]] (Score: {parsed_verdict})")
            return parsed_verdict
        else:
            # Parsing failed, error already logged by _parse_judge_verdict
            logger.error("Failed to parse the judgment verdict from the response.")
            return None


