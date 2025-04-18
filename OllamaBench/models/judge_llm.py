import re
import ollama
import logging
from typing import Optional

from utils.chat_with_ollama import chat_with_ollama

logger = logging.getLogger(__name__)

def judge_responses(
    judge_model: str,
    response1: str,
    response2: str,
    prompt: str,
    system_prompt: Optional[str] = None,
) -> Optional[float]:
    
    judge_prompt = f"""
    You are an automated evaluation system. Your SOLE task is to determine which AI assistant's response is better based on the user's query.

    **Evaluation Criteria:**
    * Helpfulness and relevance to the user's query.
    * Accuracy and correctness of information.
    * Depth, detail, and creativity.
    * Adherence to instructions in the user query.

    **Input Format:**
    You will receive the user query and the responses from Assistant A and Assistant B.
    [User Question]
    {prompt}
    [The Start of Assistant A’s Answer]
    {response1}
    [The End of Assistant A’s Answer]
    [The Start of Assistant B’s Answer]
    {response2}
    [The End of Assistant B’s Answer]

    **Output Format:**
    Your output MUST be exactly one of the following three options, and ABSOLUTELY NOTHING ELSE:
    * `[[A]]` if Assistant A provided the better response.
    * `[[B]]` if Assistant B provided the better response.
    * `[[C]]` if the responses are of comparable quality or tied.

    **IMPORTANT RULES:**
    * Do NOT provide any explanation or justification.
    * Do NOT include any text before or after the verdict (`[[A]]`, `[[B]]`, or `[[C]]`).
    * Be objective. Do not let response length, assistant names, or the order of presentation influence your decision.
    * Your ENTIRE response must be ONLY `[[A]]` OR `[[B]]` OR `[[C]]`.

    **Examples of Correct Output (entire response from the judge):**
    * [[A]]
    * [[C]]
    * [[B]]

    Now, evaluate the provided query and responses and output your verdict in the required format.
    """

    messages = []
    # if system_prompt:
    #     messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": judge_prompt})

    logger.info(f"Judging responses using model '{judge_model}'...")
    try:
        result = chat_with_ollama(messages=messages, 
                                  model=judge_model,
                                  stream=False,
                                  options={"temperature": 0.0})
        response_text = result
        logger.debug(f"Judge model '{judge_model}' raw response: {response_text}")

        verdict_match = re.search(r'\[\[(A|B|C)\]\]', response_text)

        if verdict_match:
            verdict = verdict_match.group(1)
            logger.info(f"Judge '{judge_model}' verdict: {verdict}")
            if verdict == "A":
                return 1.0
            elif verdict == "B":
                return 0.0
            elif verdict == "C":
                return 0.5
        else:
            response_lower = response_text.lower()
            if response_lower.startswith('a'):
                 logger.warning(f"Verdict format deviation (used fallback): Judge '{judge_model}' response started with 'A'. Interpreting as A.")
                 return 1.0
            elif response_lower.startswith('b'):
                 logger.warning(f"Verdict format deviation (used fallback): Judge '{judge_model}' response started with 'B'. Interpreting as B.")
                 return 0.0
            elif response_lower.startswith('c'):
                 logger.warning(f"Verdict format deviation (used fallback): Judge '{judge_model}' response started with 'C'. Interpreting as C.")
                 return 0.5
            else:
                logger.error(f"Could not extract a valid verdict ([[A]], [[B]], or [[C]]) from judge '{judge_model}'. Response: {response_text}")
                return None

    except Exception as e:
        logger.error(f"Error during Ollama API call to judge model '{judge_model}': {e}", exc_info=True)
        return None
