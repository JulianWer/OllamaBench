import requests
import json
import logging
import time
import re
from collections import deque
from typing import List, Dict, Optional, Union, Iterator, Any

logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_CHAT_API_PATH = "/api/chat"
DEFAULT_TIMEOUT = 120  # Increased default timeout
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5 # Delay in seconds

# --- Constants for Repetition Check ---
# Streaming check
MIN_CHUNK_LEN_FOR_STREAM_CHECK = 5 # Ignore very short chunks for repetition checks
STREAM_REPETITION_THRESHOLD = 3    # How many consecutive identical chunks trigger detection?
# Non-streaming check
NON_STREAM_REPETITION_MIN_LEN = 30 # Min length of substring to check for repetition
NON_STREAM_REPETITION_MIN_REPEATS = 3 # How many times must the substring repeat?


# --- Helper Function ---
def _build_api_url(base_url: Optional[str], api_path: Optional[str]) -> str:
    """Constructs the full API URL from base and path."""
    base = (base_url or DEFAULT_OLLAMA_BASE_URL).rstrip('/')
    path = (api_path or DEFAULT_CHAT_API_PATH).lstrip('/')
    return f"{base}/{path}"

def _check_for_excessive_repetition(
    text: str,
    min_len: int = NON_STREAM_REPETITION_MIN_LEN,
    min_repeats: int = NON_STREAM_REPETITION_MIN_REPEATS
) -> bool:
    """
    Checks for directly consecutive repetitions of longer text parts in the final string.
    Args:
        text: The text content to check.
        min_len: Minimum length of the substring to be considered repeating.
        min_repeats: Minimum number of times the substring must repeat consecutively.
    Returns:
        True if excessive repetition is detected, False otherwise.
    """
    if not text or len(text) < min_len * min_repeats:
        return False
    # Regex explanation:
    # (.{min_len,}?) : Capture group 1: Any character (.), repeated min_len or more times ({min_len,}).
    #                 The '?' makes the repetition non-greedy, potentially finding shorter repeats first.
    # \1             : Backreference to whatever was captured in group 1.
    # {min_repeats-1,}: Match the backreference (the captured group) at least (min_repeats - 1) more times.
    #                 So, group 1 + (min_repeats - 1) repetitions = min_repeats total occurrences.
    pattern = re.compile(r'(.{' + str(min_len) + r',}?)\1{' + str(min_repeats - 1) + r',}')
    if pattern.search(text):
        logger.warning(f"Excessive repetition detected in non-stream response. Pattern: {pattern.pattern} matched.")
        # To see *what* matched (can be verbose):
        # match = pattern.search(text)
        # if match:
        #    logger.debug(f"Repetitive sequence found: '{match.group(1)}' repeated")
        return True
    return False


def _process_stream(response: requests.Response, model: str) -> Iterator[Dict[str, Any]]:
    """
    Helper generator to process the streaming response, including a check for repetitive content.
    Raises:
        StopIteration: If repetitive content is detected based on thresholds.
    """
    buffer = ""
    incomplete_json = "" # Buffer for handling chunks split across lines
    # --- State for Repetition Check ---
    last_chunks = deque(maxlen=STREAM_REPETITION_THRESHOLD)

    try:
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                buffer += decoded_line + "\n" # Keep track of raw stream for debugging

                full_line_data = incomplete_json + decoded_line
                incomplete_json = "" # Reset fragment buffer

                try:
                    chunk = json.loads(full_line_data)
                    # Basic chunk structure validation
                    if isinstance(chunk, dict) and "message" in chunk and isinstance(chunk["message"], dict):
                        content = chunk.get('message', {}).get('content', '')

                        # --- Repetition Check Logic ---
                        if content: # Only check if there's actual content
                             current_content_stripped = content.strip()
                             if len(current_content_stripped) >= MIN_CHUNK_LEN_FOR_STREAM_CHECK:
                                 # Check if buffer is full and all elements match the current one
                                 if len(last_chunks) == STREAM_REPETITION_THRESHOLD and all(c == current_content_stripped for c in last_chunks):
                                     logger.warning(f"Repetitive stream content detected from model '{model}'. Stopping stream. Last chunk content: '{current_content_stripped}'")
                                     # Stop the generator cleanly, signaling the issue
                                     raise StopIteration("Repetitive stream content detected")
                                 # Add current stripped content to the tracking deque
                                 last_chunks.append(current_content_stripped)
                             elif len(current_content_stripped) > 0:
                                # If a short, non-empty chunk arrives, it breaks the sequence. Clear the checker.
                                last_chunks.clear()
                             # else: an empty or whitespace-only chunk, doesn't affect repetition check status
                        else:
                             # If content is empty, it might break a sequence, clear checker
                             last_chunks.clear()
                        # --- End Repetition Check ---

                        yield chunk # Yield the original chunk
                    else:
                         logger.warning(f"Stream chunk for model '{model}' has unexpected structure: {chunk}")

                except json.JSONDecodeError:
                    incomplete_json = full_line_data
                    logger.debug(f"Incomplete JSON detected for model '{model}', buffering: '{incomplete_json}'")
                    continue # Skip to the next line

    except requests.exceptions.ChunkedEncodingError as stream_err:
        logger.error(f"Connection broken during stream processing for model '{model}': {stream_err}. Received buffer: {buffer[:500]}...", exc_info=True)
        raise # Re-raise to signal the issue to the caller
    except StopIteration as si: # Catch the specific StopIteration from the repetition check
        logger.warning(f"Stream stopped due to: {si}")
        # Optionally yield a final error chunk before stopping?
        # yield {"error": str(si), "done": True, "message": {"role":"assistant", "content":""}}
        return # Cleanly end the generator
    except Exception as stream_err:
        logger.error(f"Generic error during stream processing for model '{model}': {stream_err}. Received buffer: {buffer[:500]}...", exc_info=True)
        raise # Re-raise other errors

    if incomplete_json:
         logger.warning(f"Stream for model '{model}' ended with incomplete JSON data in buffer: '{incomplete_json}'")


# --- Main Chat Function ---
def chat_with_ollama(
    messages: List[Dict[str, str]],
    model: str,
    ollama_api_base_url: Optional[str] = None,
    ollama_chat_api_path: Optional[str] = None,
    stream: bool = False,
    options: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: int = DEFAULT_RETRY_DELAY
) -> Union[str, Iterator[Dict[str, Any]], None]:
    """
    Sends a chat request to the Ollama API with configuration, timeout, retry logic,
    and basic repetition detection.

    Args:
        messages: A list of message dictionaries, e.g., [{"role": "user", "content": "Hi"}].
        model: The name of the Ollama model to use (e.g., "llama3:latest").
        ollama_api_base_url: Base URL of the Ollama API. Uses default if None.
        ollama_chat_api_path: Path for the chat endpoint. Uses default if None.
        stream: If True, return an iterator yielding response chunks. Checks for repetitive chunks.
                If False, return the complete response content as a string. Checks final content for repetition.
        options: Additional options for the Ollama API (e.g., temperature, top_p).
        timeout: Request timeout in seconds for each attempt.
        max_retries: Maximum number of retry attempts on network/server failures.
                     Repetition errors typically do not trigger retries.
        retry_delay: Delay in seconds between retries.

    Returns:
        - If stream=True: An iterator yielding JSON chunk dictionaries from the API.
                         The iterator may end prematurely if repetition is detected.
        - If stream=False: The complete assistant message content as a string, or None if failed.
        - None: If the request fails after all retries, returns invalid data, or detects repetition
                in non-streaming mode.
    """
    api_url = _build_api_url(ollama_api_base_url, ollama_chat_api_path)

    if not isinstance(messages, list) or not all(isinstance(m, dict) and "role" in m and "content" in m for m in messages):
        logger.error(f"Invalid 'messages' format provided for model '{model}'. Expected List[Dict[str, str]].")
        return None

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream
    }
    if options is not None:
        if isinstance(options, dict):
             payload["options"] = options
        else:
             logger.warning(f"Invalid 'options' type provided (expected dict, got {type(options)}). Ignoring options.")


    logger.debug(f"Sending request to {api_url} for model '{model}'. Stream: {stream}. Timeout: {timeout}s. Payload keys: {list(payload.keys())}")

    last_exception: Optional[Exception] = None

    for attempt in range(max_retries):
        logger.info(f"Attempt {attempt + 1}/{max_retries} to contact model '{model}' at {api_url}")
        try:
            response = requests.post(
                api_url,
                json=payload,
                stream=stream,
                timeout=timeout
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            logger.debug(f"Attempt {attempt + 1}: Received response status {response.status_code} from model '{model}'.")

            # --- Stream Processing ---
            if stream:
                # Return the generator which now includes repetition checks
                return _process_stream(response, model)

            # --- Non-Stream Processing ---
            else:
                try:
                    response_data = response.json()
                    message_content = response_data.get('message', {}).get('content')

                    if message_content is None:
                        logger.warning(f"No 'content' found in non-stream response for model '{model}'. Full response: {response_data}")
                        last_exception = ValueError("Missing 'content' in response")
                        continue # Try next attempt if retries left

                    if not isinstance(message_content, str):
                        logger.warning(f"Model '{model}' response content is not a string (type: {type(message_content)}). Attempting conversion.")
                        try:
                            message_content = str(message_content)
                        except Exception as conv_e:
                            logger.error(f"Failed to convert response content to string for model '{model}'. Error: {conv_e}. Raw content: {response_data.get('message', {}).get('content')}")
                            last_exception = conv_e
                            continue # Try next attempt

                    # --- Non-Streaming Repetition Check ---
                    message_content_stripped = message_content.strip()
                    if _check_for_excessive_repetition(message_content_stripped):
                         logger.error(f"Model '{model}' non-stream response failed repetition check.")
                         last_exception = ValueError("Repetitive content detected in non-stream response")
                         # Repetition error -> break retry loop, likely won't be fixed by retrying
                         break
                    # --- End Repetition Check ---

                    logger.info(f"Successfully received non-stream response from model '{model}'.")
                    return message_content_stripped # Return the stripped content

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON response from Ollama (model '{model}'). Status: {response.status_code}. Response text: {response.text[:500]}...", exc_info=True)
                    last_exception = e
                    break # Don't retry JSON decode errors

        except requests.exceptions.Timeout as e:
            last_exception = e
            logger.warning(f"Attempt {attempt + 1}/{max_retries} timed out for model '{model}' after {timeout}s.")
            # Continue to next retry attempt
        except requests.exceptions.ConnectionError as e:
            last_exception = e
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed with connection error for model '{model}': {e}")
            # Continue to next retry attempt
        except requests.exceptions.RequestException as e: # Includes HTTPError
            last_exception = e
            status_code = e.response.status_code if e.response is not None else 'N/A'
            response_text = e.response.text[:500] if e.response is not None else 'N/A'
            logger.error(f"Request failed for model '{model}' (Attempt {attempt + 1}/{max_retries}). Status: {status_code}. Response: {response_text}", exc_info=False) # exc_info=False to avoid huge tracebacks for common HTTP errors
            logger.debug("Full traceback for RequestException:", exc_info=True) # Add debug log for full traceback if needed
            # Break on HTTP errors (like 4xx, 5xx) as retrying might not help
            break
        except Exception as e: # Catch-all for unexpected errors during the request/response handling phase
             last_exception = e
             logger.exception(f"An unexpected error occurred during chat_with_ollama for model '{model}' (Attempt {attempt + 1}/{max_retries}): {e}")
             break # Break on unexpected errors

        # Wait before retrying if not the last attempt and we didn't break
        if attempt < max_retries - 1:
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    # --- End of Retries / Loop Exit ---
    logger.error(f"Failed to get a valid response from model '{model}' at {api_url} after {attempt + 1} attempt(s).")
    if last_exception:
        logger.error(f"Last encountered error: {type(last_exception).__name__}: {last_exception}")
    return None


# --- Example Usage ---
if __name__ == "__main__":
    # Setup basic logging to see output
    logging.basicConfig(
        level=logging.INFO, # Change to DEBUG for more verbose output
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_model = "llama3:latest" # Make sure this model is served by your Ollama instance
    # Example prompt that *might* sometimes trigger repetition depending on the model/state
    # test_messages = [{"role": "user", "content": "Repeat the word 'test' fifty times."}]
    test_messages = [{"role": "user", "content": "Explain the concept of ELO rating in simple terms."}]
    test_messages_potentially_repetitive = [{"role": "user", "content": "Write a story that just repeats the phrase 'the cat sat on the mat' over and over again."}]


    print(f"\n--- Testing Non-Streaming Request for {test_model} ---")
    non_stream_response = chat_with_ollama(messages=test_messages, model=test_model, stream=False)
    if non_stream_response:
        print("\nResponse Received (Non-Streaming):")
        print(non_stream_response)
    else:
        print("\nFailed to get non-streaming response or repetition detected.")

    print(f"\n--- Testing Streaming Request for {test_model} ---")
    stream_iterator = chat_with_ollama(messages=test_messages, model=test_model, stream=True)
    if stream_iterator:
        print("\nStreaming Response:")
        full_streamed_content = ""
        try:
            for chunk in stream_iterator:
                # Check if the stream processor yielded an error object (optional way, StopIteration is cleaner)
                # if chunk.get("error"):
                #    print(f"\n[Stream Error: {chunk.get('error')}]")
                #    break
                content = chunk.get('message', {}).get('content', '')
                if content:
                    print(content, end='', flush=True)
                    full_streamed_content += content
                # The 'done' flag indicates the end of the stream from Ollama's perspective
                # but our _process_stream might stop earlier due to repetition via StopIteration
                if chunk.get('done'):
                    print("\n[Ollama Stream Done Flag Received]")
                    # Note: done=True might appear even if StopIteration was raised,
                    # depending on timing. The loop already exited if StopIteration occurred.
            print("\n--- End of Stream Iteration ---")
            print(f"Full streamed content length: {len(full_streamed_content)}")
        except Exception as e:
            # Catch potential errors raised from _process_stream if not handled internally
            print(f"\nError during stream processing in example: {type(e).__name__}: {e}")
    else:
        print("\nFailed to get stream iterator (initial request failed).")


    # --- Test with potentially repetitive prompt ---
    print(f"\n--- Testing Streaming Request for {test_model} (Potentially Repetitive) ---")
    stream_iterator_rep = chat_with_ollama(messages=test_messages_potentially_repetitive, model=test_model, stream=True)
    if stream_iterator_rep:
        print("\nStreaming Response (Potentially Repetitive):")
        full_streamed_content_rep = ""
        try:
            for chunk in stream_iterator_rep:
                content = chunk.get('message', {}).get('content', '')
                if content:
                    print(content, end='', flush=True)
                    full_streamed_content_rep += content
                if chunk.get('done'):
                     print("\n[Ollama Stream Done Flag Received]")

            print("\n--- End of Stream Iteration (Potentially Repetitive) ---")
            print(f"Full streamed content length: {len(full_streamed_content_rep)}")
        except Exception as e:
            print(f"\nError during stream processing in example: {type(e).__name__}: {e}")
    else:
        print("\nFailed to get stream iterator (initial request failed).")