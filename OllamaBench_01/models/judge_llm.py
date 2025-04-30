import re
import logging
from typing import Optional, Dict, Any, List

# Project specific imports
from utils.chat_with_ollama import chat_with_ollama # Use the enhanced chat function

logger = logging.getLogger(__name__)

# --- Constants ---
# Default prompt structure - could be loaded from config or template file
DEFAULT_SYSTEM_PROMPT = "You are an impartial AI judge evaluating the quality of two AI assistant responses based on a user query and potentially a ground truth answer."

JUDGE_PROMPT_TEMPLATE = """
{system_prompt_header}

**Evaluation Task:**
Your objective is to determine which AI assistant's response ('Assistant A' or 'Assistant B') is superior according to the criteria below.

**Evaluation Criteria:**
* **Relevance & Helpfulness:** How well does the response address the user's query? Is it helpful and on-topic?
* **Accuracy & Correctness:** Is the information provided accurate and factually correct?
* **Depth & Detail:** Does the response provide sufficient depth, detail, and insight? (Avoid rewarding verbosity alone).
* **Clarity & Structure:** Is the response well-organized, clear, and easy to understand?
* **Instruction Following:** Did the response adhere to any specific instructions or constraints in the user query?
{ground_truth_criteria}

**Input Data:**

**[User Query]**
{prompt}
**[End User Query]**

{ground_truth_section}

**[Assistant A's Response]**
{response1}
**[End Assistant A's Response]**

**[Assistant B's Response]**
{response2}
**[End Assistant B's Response]**

**Your Verdict:**
Based on your evaluation, choose **exactly one** of the following options. Your entire output must consist *only* of the chosen option, with no additional text, explanation, or formatting.

* `[[A]]` (If Assistant A is better)
* `[[B]]` (If Assistant B is better)
* `[[C]]` (If the responses are of comparable quality or tied)

**Output your verdict now:**
"""

GROUND_TRUTH_CRITERIA_TEXT = "* **Ground Truth Alignment:** How closely does the response align with the provided Ground Truth Answer (if applicable)?"
GROUND_TRUTH_SECTION_TEMPLATE = """
**[Ground Truth Answer]**
{ground_truth}
**[End Ground Truth Answer]**
"""

# --- Helper Function ---
def _build_judge_prompt(
    prompt: str,
    response1: str,
    response2: str,
    ground_truth: Optional[str] = None,
    custom_system_prompt: Optional[str] = None
) -> str:
    """Constructs the prompt for the judge LLM."""
    
    judge_prompt_parts = [
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

    system_header = custom_system_prompt if custom_system_prompt else DEFAULT_SYSTEM_PROMPT
    gt_criteria = GROUND_TRUTH_CRITERIA_TEXT if ground_truth else ""
    gt_section = GROUND_TRUTH_SECTION_TEMPLATE.format(ground_truth=ground_truth) if ground_truth else ""

    # Basic cleaning of inputs to avoid breaking the prompt structure
    clean_prompt = str(prompt).replace("[End User Query]", "[ End User Query ]") # Avoid premature ending
    clean_resp1 = str(response1).replace("[End Assistant A's Response]", "[ End Assistant A's Response ]")
    clean_resp2 = str(response2).replace("[End Assistant B's Response]", "[ End Assistant B's Response ]")
    clean_gt = str(ground_truth).replace("[End Ground Truth Answer]", "[ End Ground Truth Answer ]") if ground_truth else ""


    formatted_prompt = judge_prompt
    return formatted_prompt.strip()


def _parse_judge_verdict(response_text: str) -> Optional[float]:
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

    # 3. Fallback: Check if the cleaned response *contains* the verdict string clearly
    # Be cautious with this, as it might misinterpret explanations.
    # Only use if the response is very short or clearly intended as just the verdict.
    # Example check (could be refined):
    # if "[[A]]" in cleaned_response and len(cleaned_response) < 20: return 1.0
    # ... (similar for B and C)

    logger.error(f"Failed to parse judge verdict. Could not find [[A]], [[B]], or [[C]] in response: '{cleaned_response[:200]}...'")
    return None


# --- Main Judge Function ---
def judge_responses(
    judge_model: str,
    response1: str,
    response2: str,
    prompt: str,
    config: Dict[str, Any], # Pass the main config
    ground_truth: Optional[str] = None,
    judge_system_prompt: Optional[str] = None, # Allow override via argument
) -> Optional[float]:
    """
    Evaluates two LLM responses using a specified judge model via the Ollama API.

    Args:
        judge_model: Name of the judge model (e.g., "llama3.1:8b").
        response1: The response generated by the first model (Assistant A).
        response2: The response generated by the second model (Assistant B).
        prompt: The original user query or prompt given to the models.
        config: The main application configuration dictionary, used for API details.
        ground_truth: Optional ground truth answer for reference.
        judge_system_prompt: Optional system prompt override for the judge.

    Returns:
        A float score representing the verdict:
            - 1.0 if Assistant A's response is judged better.
            - 0.0 if Assistant B's response is judged better.
            - 0.5 if the responses are judged as a tie/comparable.
        Returns None if the judging process fails (API error, parsing error, etc.).
    """
    if not judge_model:
        logger.error("Judge model name cannot be empty.")
        return None
    if response1 is None or response2 is None:
        logger.error("Cannot judge responses: One or both responses are None.")
        return None

    # --- Get API configuration ---
    ollama_config = config.get("ollama", {})
    api_base_url = ollama_config.get("api_base_url")
    chat_api_path = ollama_config.get("chat_api_path")
    timeout = ollama_config.get("default_timeout", 120) # Use specific judge timeout?
    max_retries = ollama_config.get("max_retries", 3)
    retry_delay = ollama_config.get("retry_delay", 5)

    # --- Prepare the prompt for the judge ---
    # Use argument override or default system prompt
    system_prompt_to_use = judge_system_prompt if judge_system_prompt is not None else DEFAULT_SYSTEM_PROMPT
    full_judge_prompt = _build_judge_prompt(prompt, response1, response2, ground_truth, system_prompt_to_use)

    # Message format for chat_with_ollama
    # Note: The judge prompt itself contains the structure, so we send it as user content.
    # Alternatively, parts could be structured into system/user roles if the judge model expects that.
    messages: List[Dict[str, str]] = [
        # Optional: Could add the system prompt here if the model handles it better
        # {"role": "system", "content": system_prompt_to_use},
        {"role": "user", "content": full_judge_prompt}
    ]

    # --- Set judge-specific generation options (low temperature for deterministic output) ---
    judge_options = {
        "temperature": 0.0, # Make the judge deterministic
        "top_p": None,      # Disable nucleus sampling
        "top_k": None       # Disable top-k sampling
        # Add other options if needed, e.g., stop sequences?
    }

    logger.info(f"Requesting judgment from model '{judge_model}'...")
    # logger.debug(f"Judge Prompt:\n{full_judge_prompt}") # Log full prompt only if needed

    try:
        # Call the Ollama API (non-streaming for judge)
        judge_response_text = chat_with_ollama(
            messages=messages,
            model=judge_model,
            ollama_api_base_url=api_base_url,
            ollama_chat_api_path=chat_api_path,
            stream=False, # Judge response should be short and non-streamed
            options=judge_options,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay
        )

        if judge_response_text is None:
            logger.error(f"Judge model '{judge_model}' failed to return a response after retries.")
            return None

        logger.debug(f"Raw response from judge '{judge_model}': '{judge_response_text}'")

        # --- Parse the verdict ---
        verdict_score = _parse_judge_verdict(judge_response_text)

        if verdict_score is not None:
            verdict_map = {1.0: "A", 0.0: "B", 0.5: "C"}
            logger.info(f"Judgment received from '{judge_model}': [[{verdict_map[verdict_score]}]] (Score: {verdict_score})")
            return verdict_score
        else:
            # Parsing failed, error already logged by _parse_judge_verdict
            return None

    except Exception as e:
        logger.exception(f"An unexpected error occurred during the call to judge model '{judge_model}': {e}")
        return None
