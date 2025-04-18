import requests
import json
import logging
from typing import List, Dict, Optional, Union, Iterator

logger = logging.getLogger(__name__) 

def chat_with_ollama(messages: List[Dict[str, str]],
                     model: str = 'llama3',
                     ollama_api_url: str = 'http://localhost:11434/api/chat', 
                     stream: bool = False,
                     options: Optional[Dict] = None) -> Union[str, Iterator[Dict], None]:

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream
    }
    if options:
        payload["options"] = options

    logger.debug(f"Sending request to {ollama_api_url} with payload: {json.dumps(payload)}")

    try:
        response = requests.post(ollama_api_url, json=payload, stream=stream)
        response.raise_for_status()  
        logger.debug(f"Received response status: {response.status_code}")

        if stream:
            def generate_chunks() -> Iterator[Dict]:
                lines = ""
                try:
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            lines += decoded_line
                            try:
                                chunk = json.loads(decoded_line)
                                logger.debug(f"Stream chunk received: {chunk}")
                                yield chunk
                            except json.JSONDecodeError:
                                logger.warning(f"Skipping non-JSON line in stream: {decoded_line}")
                                continue
                except Exception as stream_err:
                     logger.error(f"Error during stream processing: {stream_err}. Received lines: {lines}", exc_info=True)
            return generate_chunks()
        else:
            try:
                response_data = response.json()
                logger.debug(f"Non-stream response received: {response_data}")
                message_content = response_data.get('message', {}).get('content')
                if message_content is None:
                     logger.warning(f"No 'content' found in response message for model '{model}'. Full response: {response_data}")
                     return None
                if not isinstance(message_content, str):
                    logger.warning(f"Model '{model}' returned non-string content via custom function. Type: {type(message_content)}. Attempting conversion.")
                    try:
                        message_content = str(message_content)
                    except Exception as conv_e:
                        logger.error(f"Could not convert content to string for model '{model}'. Conversion error: {conv_e}. Raw: {response_data}")
                        return None 
                return message_content.strip() 
            except json.JSONDecodeError:
                logger.error(f"Could not decode JSON response from Ollama. Status: {response.status_code}. Response text: {response.text}", exc_info=True)
                return None 

    except requests.exceptions.ConnectionError:
        logger.error(f"Connection Error: Could not connect to Ollama API at {ollama_api_url}. Is Ollama running?", exc_info=True)
        return None
    except requests.exceptions.Timeout:
        logger.error(f"Timeout Error: Request to Ollama API timed out. URL: {ollama_api_url}", exc_info=True)
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request Error: An error occurred interacting with Ollama API at {ollama_api_url}. Status: {e.response.status_code if e.response else 'N/A'}. Response: {e.response.text if e.response else 'N/A'}", exc_info=True)
        return None