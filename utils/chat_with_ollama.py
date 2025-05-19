import json
import logging
from typing import Any, Dict, List, Optional, Union, Iterator

import requests


logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT = 400

def chat_with_ollama(
    api_url: str,
    model: str,
    prompt: str,
    system_message: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    has_reasoning: Optional[bool] = False,
    options: Optional[Dict[str, Any]] = None,
    max_retries: Optional[int]= 3,
    request_timeout: Optional[float] = DEFAULT_REQUEST_TIMEOUT
) -> Union[Optional[Dict[str, Any]], Iterator[Optional[Dict[str, Any]]]]:
    """
    Sends a chat request to the Ollama API.

    If stream is False (default), returns the full response dictionary.
    If stream is True, returns an iterator yielding response chunks.

    Args:
        api_url: The base URL of the Ollama API (e.g., "http://localhost:11434").
        model: The name of the model to use.
        prompt: The user's prompt.
        system_message: An optional system message to guide the model.
        temperature: The temperature setting for generation.
        max_tokens: The maximum number of tokens to generate (optional).
                     Note: Direct 'max_tokens' is not a standard Ollama option.
                     It might be mapped to 'num_predict' or similar in 'options'.
        stream: Whether to stream the response.
        options: Additional Ollama options (e.g., {"num_ctx": 4096}).
        max_retries: If the the model takes to long and the timeout kicks in, the model gets retiries
        has_reasoning: Should the model have the ability for reasoning?

    Returns:
        - If stream is False: A dictionary containing the full API response, or None on error.
        - If stream is True: An iterator yielding dictionary chunks of the API response,
                             or None if the initial request fails. Yields None on stream error.
    """
    api_chat_url = f"{api_url.rstrip('/')}/api/chat"
    messages: List[Dict[str, str]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": f'{prompt}{'/no_think' if not has_reasoning else ''}' })
    
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,  # Set stream based on the parameter
        "options": options.copy() if options else {}, # Use a copy to avoid modifying the original
    }

    # Add temperature to options if not already present by the user
    if "temperature" not in payload["options"]:
        payload["options"]["temperature"] = temperature

    if max_tokens is not None:
        # Ollama doesn't have a direct 'max_tokens' like OpenAI.
        # 'num_predict' is the closest, controlling max tokens to generate.
        # User can also pass 'num_predict' directly in 'options'.
        if 'num_predict' not in payload["options"]:
            payload["options"]["num_predict"] = max_tokens
            logger.debug(f"'max_tokens' ({max_tokens}) was mapped to 'num_predict' in options.")
        else:
            logger.warning(
                f"'max_tokens' ({max_tokens}) was provided, but 'num_predict' "
                f"({payload['options']['num_predict']}) already exists in 'options'. "
                f"'num_predict' from 'options' will be used."
            )
        logger.warning(
            "The 'max_tokens' parameter is interpreted as 'num_predict' for Ollama. "
            "Ensure this aligns with your model's capabilities."
        )


    api_response_obj = None # Initialize to ensure it's defined for finally block if stream is True
    for attempt in range(max_retries):
        try:
            logger.debug(f"Sending request to {api_chat_url} with payload: {json.dumps(payload)}")
            api_response_obj = requests.post(
                api_chat_url,
                json=payload,
                timeout=request_timeout,  # Increased timeout for potentially long generations
                stream=stream   # Pass stream argument to requests.post
            )
            api_response_obj.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            if stream:
                # Define a generator function for streaming
                def stream_generator() -> Iterator[Optional[Dict[str, Any]]]:
                    nonlocal api_response_obj # Allow modification of api_response_obj in outer scope for finally
                    try:
                        for line in api_response_obj.iter_lines():
                            if line:
                                try:
                                    decoded_line = line.decode('utf-8')
                                    chunk = json.loads(decoded_line)
                                    yield chunk
                                    # Ollama's stream typically ends when 'done' is true in a chunk
                                    if chunk.get("done"):
                                        logger.debug("Stream finished (done=true received).")
                                        break
                                except json.JSONDecodeError:
                                    logger.error(f"Error decoding JSON stream chunk: {decoded_line}")
                                    yield None # Signal an error in the stream
                                except Exception as e_chunk:
                                    logger.error(f"Error processing stream chunk: {e_chunk}")
                                    yield None # Signal an error
                            # else: empty keep-alive line, ignore
                    finally:
                        if api_response_obj:
                            api_response_obj.close() # Ensure the response is closed
                            logger.debug("Stream response object closed.")
                return stream_generator()  # Return the generator
            else:
                # Non-streaming logic
                response_data = api_response_obj.json()
                logger.debug(f"Received non-streamed response: {response_data}")
                return response_data

        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error: Could not connect to Ollama API at {api_chat_url}.")
            return None
        except requests.exceptions.Timeout:
            logger.error(f"Timeout error: Request to {api_chat_url} timed out.")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during chat request to Ollama API ({api_chat_url}): {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status code: {e.response.status_code}")
                try:
                    logger.error(f"Response text: {e.response.text}")
                except Exception: # Handle cases where response.text might not be available or decodable
                    logger.error("Response text could not be displayed.")
            return None # For both stream and non-stream initial request failure
        except json.JSONDecodeError as e: # Should only apply to non-streamed .json() call
            logger.error(f"Error decoding JSON response from {api_chat_url}: {e}. Response text: {api_response_obj.text if api_response_obj else 'N/A'}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during chat with Ollama: {e}", exc_info=True)
            return None
        finally:
            # If not streaming and an error occurred after response but before .json(), or if stream is True but error before generator is fully consumed.
            # The stream_generator's finally block handles closing for successful streaming.
            # For non-streaming, requests typically closes the connection unless an error prevents .json()
            if not stream and api_response_obj and not api_response_obj.raw.closed:
                api_response_obj.close()
                logger.debug("Non-stream response object closed in outer finally.")