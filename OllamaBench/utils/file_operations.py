import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import fcntl # For file locking (Unix-based systems)
import time
import errno
logger = logging.getLogger(__name__)

class FileLock:
    def __init__(self, lock_file_path: str, timeout: int = 10):
        self._lock_file_path = lock_file_path
        self._lock_file = None
        self._timeout = timeout

    def __enter__(self):
        start_time = time.time()
        while True:
            try:
                # Create lock file directory if it doesn't exist
                os.makedirs(os.path.dirname(self._lock_file_path), exist_ok=True)
                # Open the lock file
                self._lock_file = open(self._lock_file_path, 'w')
                # Try to acquire an exclusive, non-blocking lock
                fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                logger.debug(f"Acquired lock on {self._lock_file_path}")
                return self # Not strictly necessary, but common practice
            except IOError as e:
                # If lock is already held (EACCES or EAGAIN), wait and retry
                if e.errno == errno.EACCES or e.errno == errno.EAGAIN:
                    if time.time() - start_time >= self._timeout:
                        logger.error(f"Timeout acquiring lock on {self._lock_file_path}")
                        if self._lock_file:
                            self._lock_file.close()
                        raise TimeoutError(f"Could not acquire lock on {self._lock_file_path} within {self._timeout} seconds.")
                    logger.debug(f"Waiting for lock on {self._lock_file_path}...")
                    time.sleep(0.1) # Wait a short period before retrying
                else:
                    # Other IOErrors are unexpected
                    if self._lock_file:
                        self._lock_file.close()
                    logger.error(f"Unexpected IOError acquiring lock: {e}")
                    raise
            except Exception as e:
                 # Catch any other exceptions during lock acquisition
                if self._lock_file:
                    self._lock_file.close()
                logger.error(f"Unexpected error acquiring lock: {e}")
                raise


    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._lock_file:
            # Release the lock and close the file
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            self._lock_file.close()
            logger.debug(f"Released lock on {self._lock_file_path}")
            # Optionally remove the lock file, though leaving it is harmless
            # try:
            #     os.remove(self._lock_file_path)
            # except OSError:
            #     pass
        # Return False to propagate exceptions if any occurred within the 'with' block
        return False


# --- Data Handling ---

ModelRatingsType = Dict[str, Dict[str, Dict[str, float]]] # Define type alias for clarity

def _set_structure(result: ModelRatingsType) -> Dict[str, Any]:
    """Adds timestamp to the results structure."""
    return {
        "models": result,
        "timestamp": datetime.now().isoformat()
    }

def load_current_json(result_file: str, lock_file: str) -> Optional[Dict[str, Any]]:
    logger.debug(f"Attempting to load JSON from {result_file} with lock {lock_file}")
    try:
        with FileLock(lock_file):
            logger.debug(f"Lock acquired for reading {result_file}")
            if not os.path.exists(result_file):
                logger.warning(f"Results file '{result_file}' not found. Returning None.")
                return None
            with open(result_file, "r", encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Successfully loaded JSON from {result_file}")
            return data
    except FileNotFoundError:
         # Should be caught by os.path.exists, but good practice to handle anyway
        logger.warning(f"Results file '{result_file}' not found.")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file '{result_file}': {e}")
        return None
    except (TimeoutError, IOError, Exception) as e:
        logger.error(f"Error loading current JSON from '{result_file}': {e}")
        return None # Return None on lock timeout or other errors

def _deep_merge_dicts(dict1: dict, dict2: dict) -> dict:
    """
    Recursively merges dict2 into dict1. Overwrites existing keys in dict1.
    """
    merged = dict1.copy() # Start with a copy of the first dictionary
    for key, value in dict2.items():
        if key in merged:
            if isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = _deep_merge_dicts(merged[key], value)
            elif isinstance(merged[key], list) and isinstance(value, list):
                 # Simple list merge (append unique items) - adjust if needed
                 merged[key].extend([item for item in value if item not in merged[key]])
            else:
                # If types differ or not dict/list, overwrite
                merged[key] = value
        else:
            merged[key] = value
    return merged

def save_elo_results(result: ModelRatingsType, config: Dict[str, Any]) -> bool:
    """
    Saves the updated ELO results to the JSON file with file locking,
    merging with existing data.

    Args:
        result: The latest ELO ratings calculated.
        config: The application configuration dictionary.

    Returns:
        True if saving was successful, False otherwise.
    """
    result_file = config["paths"]["results_file"]
    lock_file = config["paths"]["lock_file"]
    logger.debug(f"Attempting to save results to {result_file} with lock {lock_file}")

    try:
        with FileLock(lock_file):
            logger.debug(f"Lock acquired for writing {result_file}")
            # Load existing data *after* acquiring the lock
            current_data = load_current_json(result_file, lock_file) # Pass lock file to avoid re-locking issues

            # Ensure the directory exists
            os.makedirs(os.path.dirname(result_file), exist_ok=True)

            new_structured_result = _set_structure(result=result)

            if current_data and "models" in current_data:
                 # Merge the new ratings into the existing ratings
                updated_models = _deep_merge_dicts(current_data.get("models", {}), new_structured_result["models"])
                updated_data = _set_structure(result=updated_models)
                 # Preserve other top-level keys from current_data if they exist
                for key, value in current_data.items():
                    if key not in updated_data:
                         updated_data[key] = value
                updated_data["timestamp"] = new_structured_result["timestamp"] # Ensure latest timestamp
            else:
                # If no current data or it's invalid, use the new result directly
                updated_data = new_structured_result
                if current_data:
                    logger.warning(f"Existing data in {result_file} was invalid or missing 'models' key. Overwriting.")


            # Write the merged data back to the file
            with open(result_file, "w", encoding='utf-8') as f:
                json.dump(updated_data, f, indent=4, ensure_ascii=False)

            logger.info(f"ELO results successfully saved to: {result_file}")
            return True

    except (TimeoutError, IOError, Exception) as e:
        logger.error(f"Failed to save ELO results to '{result_file}': {e}")
        return False

