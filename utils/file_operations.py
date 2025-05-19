import os
import json
import logging
import datetime as dt
from typing import Dict, Any, Optional, Union, List, TypeAlias
import time
import filelock # Use filelock for robust locking
from contextlib import contextmanager

# Import Dataset types for type hinting if needed, but avoid circular dependency if possible
# from datasets import Dataset, IterableDataset

logger = logging.getLogger(__name__)

# --- Type Alias for Model Ratings Structure ---
# Using TypeAlias for better readability if Python 3.12+ is used,
# otherwise stick to the Dict structure.
# Example (adjust structure as needed):
# ModelRatingsType: TypeAlias = Dict[str, # Model Name
#                                   Dict[str, # "categorie" key
#                                        Dict[str, float] # Category Name -> ELO Rating
#                                       ]
#                                  ]
# For broader compatibility:
ModelRatingsType = Dict[str, Dict[str, Dict[str, float]]]

# --- JSON Serialization Helper ---
def _json_serial(obj: Any) -> str:
    """JSON serializer for objects not serializable by default json code (like datetime)."""
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

# --- Locking Context Manager ---
@contextmanager
def file_lock_manager(lock_file_path: str, timeout: int = 10, mode: str = 'exclusive'):
    """Provides a context manager for acquiring file locks."""
    if not lock_file_path:
        # If no lock path is provided, yield None (no lock acquired)
        yield None
        return

    lock_dir = os.path.dirname(lock_file_path)
    if lock_dir:
        try:
            os.makedirs(lock_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory for lock file '{lock_file_path}': {e}. Lock may fail.")
            # Proceed cautiously, lock might still work if dir exists or is root

    lock = filelock.FileLock(lock_file_path, timeout=timeout)
    try:
        logger.debug(f"Attempting to acquire {mode} lock on '{lock_file_path}' (Timeout: {timeout}s)")
        # Choose between exclusive (write) and shared (read) lock
        # Note: filelock doesn't explicitly support shared locks on all platforms easily.
        # We'll use exclusive lock for both read and write for simplicity and safety,
        # assuming writes are infrequent enough or reads can tolerate waiting.
        # If high concurrent reads are needed, explore alternative locking mechanisms.
        lock.acquire() # Acquires exclusive lock
        logger.debug(f"Successfully acquired lock on '{lock_file_path}'.")
        yield lock # Yield the acquired lock object (or just True)
    except filelock.Timeout:
        logger.error(f"Timeout ({timeout}s) waiting for lock on file: '{lock_file_path}'")
        raise # Re-raise the timeout error to be handled by the caller
    except Exception as e:
         logger.error(f"Error acquiring lock on file '{lock_file_path}': {e}", exc_info=True)
         raise # Re-raise other lock acquisition errors
    finally:
        if lock.is_locked:
            lock.release()
            logger.debug(f"Released lock on '{lock_file_path}'.")


# --- Core File Operations (Internal - Assume Lock is Held) ---
def _read_json_internal(file_path: str) -> Optional[Any]:
    """Reads JSON from a file. Assumes caller handles locking and existence checks if needed."""
    try:
        # Check existence and size *after* acquiring lock if locking is used externally
        if not os.path.exists(file_path):
             logger.debug(f"File not found for reading: '{file_path}'")
             return None 
        if os.path.getsize(file_path) == 0:
             logger.warning(f"File is empty: '{file_path}'. Returning None.")
             return None

        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Successfully read and parsed JSON from '{file_path}'.")
        return data
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from file: '{file_path}'. Content might be corrupted.", exc_info=True)
        return None 
    except Exception as e:
        logger.error(f"Error reading file '{file_path}': {e}", exc_info=True)
        return None 

def _write_json_internal(data: Any, file_path: str) -> bool:
    """Writes data to a JSON file. Assumes caller handles locking."""
    try:
        # Ensure directory exists before writing
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # Write to a temporary file first, then rename for atomicity
        temp_file_path = file_path + f".{os.getpid()}.tmp"
        with open(temp_file_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False, default=_json_serial)

        # Rename the temporary file to the final destination
        os.replace(temp_file_path, file_path) # os.replace is atomic on most systems

        logger.debug(f"Successfully wrote JSON data to '{file_path}'.")
        return True
    except TypeError as e:
        logger.error(f"Data type error during JSON serialization for '{file_path}': {e}", exc_info=True)
        # Clean up temp file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
             os.remove(temp_file_path)
        return False
    except Exception as e:
        logger.error(f"Error writing JSON file '{file_path}': {e}", exc_info=True)
         # Clean up temp file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
             os.remove(temp_file_path)
        return False

# --- Public API for JSON Operations (Handles Locking) ---
def load_json_file(file_path: str, lock_file_path: Optional[str] = None, lock_timeout: int = 10) -> Optional[Any]:
    """
    Loads data from a JSON file with optional file locking (exclusive lock for read).

    Args:
        file_path: Path to the JSON file.
        lock_file_path: Optional path to the lock file. If None, no locking is performed.
        lock_timeout: Maximum time (seconds) to wait for the lock.

    Returns:
        Loaded data as a Python object, or None if the file doesn't exist,
        is empty, fails to parse, or if locking times out.
    """
    logger.debug(f"Attempting to load JSON from '{file_path}' (Lock: {lock_file_path or 'None'}).")
    try:
        with file_lock_manager(lock_file_path, timeout=lock_timeout, mode='read'): # Using exclusive lock even for read
            # Lock acquired (or no lock needed if lock_file_path is None)
            return _read_json_internal(file_path)
    except filelock.Timeout:
        logger.error(f"Failed to load '{file_path}' due to lock timeout.")
        return None
    except Exception as e:
        logger.error(f"Failed to load '{file_path}' due to unexpected error during locking/reading: {e}", exc_info=True)
        return None

def save_json_file(data: Any, file_path: str, lock_file_path: Optional[str] = None, lock_timeout: int = 10) -> bool:
    """
    Saves Python data to a JSON file with optional file locking (exclusive lock).
    Uses atomic write (write to temp file, then rename).

    Args:
        data: The Python object to serialize and save.
        file_path: Path to the target JSON file.
        lock_file_path: Optional path to the lock file. If None, no locking is performed.
        lock_timeout: Maximum time (seconds) to wait for the lock.

    Returns:
        True if saving was successful, False otherwise.
    """
    logger.debug(f"Attempting to save JSON to '{file_path}' (Lock: {lock_file_path or 'None'}).")
    try:
        with file_lock_manager(lock_file_path, timeout=lock_timeout, mode='write'):
            # Lock acquired (or no lock needed)
            return _write_json_internal(data, file_path)
    except filelock.Timeout:
        logger.error(f"Failed to save '{file_path}' due to lock timeout.")
        return False
    except Exception as e:
        logger.error(f"Failed to save '{file_path}' due to unexpected error during locking/writing: {e}", exc_info=True)
        return False


# --- Specific Logic for ELO Results (Read-Modify-Write) ---
def _deep_merge_dicts(dict1: dict, dict2: dict) -> dict:
    """Recursively merges dict2 into dict1. Modifies dict1."""
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            _deep_merge_dicts(dict1[key], value) 
        else:
            dict1[key] = value 
    return dict1 

def save_elo_results(new_ratings_fragment: ModelRatingsType, config: Dict[str, Any], lock_timeout: int = 10) -> bool:
    """
    Loads existing ELO results, merges new ratings, adds a timestamp, and saves back atomically with locking.

    Args:
        new_ratings_fragment: A dictionary containing only the new or updated model ratings
                               to be merged into the main results file.
        config: The application configuration dictionary containing paths.
        lock_timeout: Timeout for acquiring the file lock.

    Returns:
        True if the merge and save operation was successful, False otherwise.
    """
    result_file = config.get("paths", {}).get("results_file")
    lock_file = config.get("paths", {}).get("lock_file")

    if not result_file or not lock_file:
        logger.error("Cannot save ELO results: 'results_file' or 'lock_file' path missing in configuration.")
        return False

    logger.info(f"Attempting to merge and save ELO results to '{result_file}' (Lock: '{lock_file}')")

    try:
        with file_lock_manager(lock_file, timeout=lock_timeout, mode='write'):
            # --- Lock Acquired ---
            logger.debug(f"Lock acquired for read-modify-write on '{result_file}'.")

            # 1. Read existing data (inside the lock)
            current_data = _read_json_internal(result_file)

            # Initialize or validate structure
            if not isinstance(current_data, dict):
                logger.warning(f"No valid data found in '{result_file}' or file is new. Initializing structure.")
                current_data = {"models": {}, "timestamp": None}
            elif "models" not in current_data or not isinstance(current_data["models"], dict):
                 logger.warning(f"Existing data in '{result_file}' missing 'models' dictionary. Re-initializing 'models'.")
                 current_data["models"] = {}

            # 2. Merge new ratings fragment into existing models data
            logger.debug(f"Merging {len(new_ratings_fragment)} new/updated model ratings.")
            # Use deep merge - modifies current_data["models"] in place
            _deep_merge_dicts(current_data["models"], new_ratings_fragment)

            # 3. Update timestamp
            current_data["timestamp"] = dt.datetime.now(dt.timezone.utc).isoformat() # Use UTC timestamp
            logger.debug(f"Updated timestamp to {current_data['timestamp']}.")

            # 4. Write updated data back (inside the lock)
            success = _write_json_internal(current_data, result_file)
            if success:
                 logger.info(f"Successfully merged and saved ELO results to '{result_file}'.")
            else:
                 logger.error(f"Failed during the write step while saving ELO results to '{result_file}'.")
                 return False 

            # --- Lock Released automatically by context manager ---
            return True 

    except filelock.Timeout:
        logger.error(f"Failed to merge/save ELO results to '{result_file}' due to lock timeout.")
        return False
    except Exception as e:
        logger.error(f"Failed to merge/save ELO results to '{result_file}' due to unexpected error: {e}", exc_info=True)
        return False

# --- Dataset Saving (Wrapper around save_json_file) ---
# Note: Assumes dataset object can be converted to a list of dicts.
# Consider adding specific type hint for dataset if needed, e.g., Union[Dataset, IterableDataset]
def save_dataset_to_json(dataset: Any, file_path: str, lock_file_path: Optional[str] = None, lock_timeout: int = 10) -> bool:
    """
    Converts a dataset object to a list of dictionaries.
    If the target JSON file already exists and contains a list,
    the new items are appended to it. Otherwise, a new file is created (or overwritten if not a list).
    Saves it as a JSON file using locking.

    Args:
        dataset: The dataset object (e.g., Hugging Face Dataset/IterableDataset).
                 Must be convertible to list via `list(dataset)`.
        file_path: Path to the target JSON file.
        lock_file_path: Optional path to the lock file. If None, a default lock
                        (file_path + ".lock") will be used by save_json_file.
        lock_timeout: Maximum time (seconds) to wait for the lock.

    Returns:
        True if conversion and saving were successful, False otherwise.
    """
    logger.info(f"Attempting to convert and save/append dataset to '{file_path}'")

    new_data_list: List[Dict[str, Any]] = []
    try:
        start_time = time.monotonic()
        if hasattr(dataset, '__iter__') and not hasattr(dataset, '__len__'): 
             logger.warning(f"Converting IterableDataset to list for saving to '{file_path}'. This may use significant memory.")
        new_data_list = list(dataset) 
        duration = time.monotonic() - start_time
        logger.debug(f"Converted new dataset portion to list with {len(new_data_list)} entries in {duration:.2f}s.")
    except Exception as e:
        logger.error(f"Failed to convert dataset object to list for saving: {e}", exc_info=True)
        return False

    # Determine the actual lock file path to use for both load and save operations for consistency.
    # If no specific lock_file_path is provided, use one next to the data file.
    actual_lock_file = lock_file_path if lock_file_path else file_path + ".lock"

    final_data_to_save: List[Dict[str, Any]] = []

    try:
        # Acquire lock for the read-modify-write operation if appending
        with file_lock_manager(actual_lock_file, timeout=lock_timeout, mode='write'): # Exclusive lock for safety
            logger.debug(f"Lock acquired for dataset save/append operation on '{file_path}'.")
            
            # Try to load existing data
            existing_data = _read_json_internal(file_path) # Use internal read as lock is already held

            if isinstance(existing_data, list):
                logger.info(f"Existing dataset file '{file_path}' found with {len(existing_data)} items. Appending new data.")
                final_data_to_save.extend(existing_data)
                final_data_to_save.extend(new_data_list)
            else:
                if existing_data is not None:
                    logger.warning(f"Existing file '{file_path}' does not contain a JSON list (found {type(existing_data)}). It will be overwritten with the new dataset.")
                else:
                    logger.info(f"No existing dataset file at '{file_path}' or file is empty. Creating new file with new data.")
                final_data_to_save.extend(new_data_list)
            
            if not final_data_to_save and not new_data_list: # Original dataset was empty, and no existing data
                 logger.warning(f"Dataset is empty and no existing data. Saving an empty list to '{file_path}'.")
                 # final_data_to_save is already an empty list in this case.

            # Save the combined or new list
            success = _write_json_internal(final_data_to_save, file_path) # Use internal write

            if success:
                logger.info(f"Successfully saved dataset ({len(final_data_to_save)} total entries) to '{file_path}'.")
            else:
                logger.error(f"Failed to save the dataset list to '{file_path}' during write.")
            
            return success

    except filelock.Timeout:
        logger.error(f"Failed to save/append dataset to '{file_path}' due to lock timeout on '{actual_lock_file}'.")
        return False
    except Exception as e:
        logger.error(f"Failed to save/append dataset to '{file_path}' due to unexpected error: {e}", exc_info=True)
        return False
