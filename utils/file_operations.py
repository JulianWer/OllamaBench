import os
import json
import logging
import datetime as dt
from typing import Dict, Any, Optional, Union, List, TypeAlias
import time # For robust temporary file naming and logging long operations
import filelock # For process-safe file access
from contextlib import contextmanager

logger = logging.getLogger(__name__)
# BasicConfig should ideally be set up in the main application entry point.
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Type alias for the structure of model ratings.
# Example: Dict[ModelName, Dict["categorie", Dict[CategoryName, EloRatingFloat]]]
ModelRatingsType: TypeAlias = Dict[str, Dict[str, Dict[str, float]]]


# --- JSON Serialization Helper ---
def _json_serial(obj: Any) -> str:
    """JSON serializer for objects not serializable by default (e.g., datetime)."""
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    # Add other custom types here if needed
    raise TypeError(f"Type {type(obj)} not serializable for JSON.")

# --- Locking Context Manager ---
@contextmanager
def file_lock_manager(lock_file_path: Optional[str], timeout: int = 10):
    """
    Provides a context manager for acquiring an exclusive file lock.
    If lock_file_path is None, it operates without locking, yielding None.
    """
    if not lock_file_path:
        logger.debug("No lock file path provided; proceeding without file lock.")
        yield None # Indicate no lock was acquired/needed
        return

    # Ensure directory for lock file exists
    lock_dir = os.path.dirname(lock_file_path)
    if lock_dir and not os.path.exists(lock_dir): # Create only if a directory part exists
        try:
            os.makedirs(lock_dir, exist_ok=True)
            logger.debug(f"Created directory for lock file: '{lock_dir}'")
        except OSError as e:
            logger.error(f"Failed to create directory for lock file '{lock_file_path}': {e}. Locking might be affected.")

    lock = filelock.FileLock(lock_file_path, timeout=timeout)
    try:
        logger.debug(f"Attempting to acquire exclusive lock on '{lock_file_path}' (Timeout: {timeout}s)")
        lock.acquire() # Acquires an exclusive lock
        logger.debug(f"Successfully acquired lock on '{lock_file_path}'.")
        yield lock # Yield the lock object itself, can be useful
    except filelock.Timeout:
        logger.error(f"Timeout ({timeout}s) waiting for lock on file: '{lock_file_path}'")
        raise # Re-raise to be handled by the caller
    except Exception as e: # Catch any other error during lock acquisition
        logger.error(f"Error acquiring lock on file '{lock_file_path}': {e}", exc_info=True)
        raise
    finally:
        if lock.is_locked:
            try:
                lock.release()
                logger.debug(f"Released lock on '{lock_file_path}'.")
            except Exception as e_release: # Catch potential errors during release
                logger.error(f"Error releasing lock on '{lock_file_path}': {e_release}", exc_info=True)


# --- Core File Operations (Internal functions assuming lock is handled by caller) ---
def _read_json_internal(file_path: str) -> Optional[Any]:
    """Reads JSON from a file. Assumes caller handles locking and existence checks if critical before call."""
    if not os.path.exists(file_path):
        logger.debug(f"File not found for reading: '{file_path}' (checked within _read_json_internal).")
        return None
    if os.path.getsize(file_path) == 0:
        logger.warning(f"File is empty: '{file_path}'. Returning None.")
        return None

    try:
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Successfully read and parsed JSON from '{file_path}'.")
        return data
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from file: '{file_path}'. File might be corrupted.", exc_info=True)
        return None
    except Exception as e: # Catch other read errors
        logger.error(f"Error reading file '{file_path}': {e}", exc_info=True)
        return None

def _write_json_internal(data: Any, file_path: str) -> bool:
    """Writes data to a JSON file atomically. Assumes caller handles locking."""
    temp_file_path = "" # Initialize for potential use in finally block
    try:
        # Ensure directory exists before writing
        dir_name = os.path.dirname(file_path)
        if dir_name: # Only try to create if there's a directory part in the path
            os.makedirs(dir_name, exist_ok=True)

        # Write to a temporary file first for atomicity
        # Using a more robust temp file naming to avoid collisions
        temp_file_path = f"{file_path}.{os.getpid()}.{int(time.time_ns() / 1000)}.tmp"
        
        with open(temp_file_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False, default=_json_serial)
        
        os.replace(temp_file_path, file_path) # os.replace is atomic on most POSIX systems and Windows
        logger.debug(f"Successfully wrote JSON data to '{file_path}'.")
        return True
    except TypeError as e: # Catch errors specifically from json.dump if data is not serializable
        logger.error(f"Data type error during JSON serialization for '{file_path}': {e}", exc_info=True)
    except Exception as e: # Catch other write/rename errors
        logger.error(f"Error writing JSON file '{file_path}': {e}", exc_info=True)
    finally:
        # Clean up temporary file if it exists and an error occurred before os.replace
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Removed temporary file '{temp_file_path}' after an error during write process.")
            except OSError as e_remove:
                logger.error(f"Error removing temporary file '{temp_file_path}': {e_remove}")
    return False

# --- Public API for JSON Operations (These handle locking) ---
def load_json_file(file_path: str, lock_file_path: Optional[str] = None, lock_timeout: int = 10) -> Optional[Any]:
    """
    Loads data from a JSON file with optional file locking.
    Returns loaded data or None on failure (e.g., file not found, parse error, lock timeout).
    """
    logger.debug(f"Request to load JSON from '{file_path}' (Lock: {lock_file_path or 'None'}).")
    try:
        with file_lock_manager(lock_file_path, timeout=lock_timeout):
            # If lock_file_path was None, context manager yields None, code proceeds.
            # If lock acquired, code proceeds.
            # If lock times out, file_lock_manager raises Timeout, caught below.
            return _read_json_internal(file_path)
    except filelock.Timeout: # Explicitly catch Timeout from the manager
        # Error already logged by file_lock_manager
        return None
    except Exception as e: # Catch other errors that might escape the manager or _read_json_internal
        logger.error(f"Failed to load '{file_path}' due to an unexpected error: {e}", exc_info=True)
        return None

def save_json_file(data: Any, file_path: str, lock_file_path: Optional[str] = None, lock_timeout: int = 10) -> bool:
    """
    Saves Python data to a JSON file with optional file locking.
    Uses atomic write. Returns True on success, False on failure.
    """
    logger.debug(f"Request to save JSON to '{file_path}' (Lock: {lock_file_path or 'None'}).")
    try:
        with file_lock_manager(lock_file_path, timeout=lock_timeout):
            return _write_json_internal(data, file_path)
    except filelock.Timeout:
        return False # Error already logged
    except Exception as e:
        logger.error(f"Failed to save '{file_path}' due to an unexpected error: {e}", exc_info=True)
        return False

# --- Specific Logic for ELO Results (Read-Modify-Write with Deep Merge) ---
def _deep_merge_dicts(target_dict: dict, source_dict: dict) -> dict:
    """
    Recursively merges `source_dict` into `target_dict`.
    `target_dict` is modified in place.
    """
    for key, value in source_dict.items():
        if key in target_dict and isinstance(target_dict[key], dict) and isinstance(value, dict):
            _deep_merge_dicts(target_dict[key], value) # Recurse for nested dicts
        else:
            target_dict[key] = value # Overwrite or add new key
    return target_dict

def save_elo_results(new_ratings_fragment: ModelRatingsType, config: Dict[str, Any], lock_timeout: int = 10) -> bool:
    """
    Loads existing ELO results, deeply merges new ratings, updates a timestamp, and saves back.
    This entire operation is performed atomically with file locking.
    """
    paths_cfg = config.get("paths", {})
    result_file_path = paths_cfg.get("results_file")
    lock_file = paths_cfg.get("lock_file") # Specific lock file for ELO results

    if not result_file_path or not lock_file:
        logger.error("Cannot save ELO results: 'results_file' or 'lock_file' path missing in configuration.")
        return False

    logger.info(f"Attempting to merge and save ELO results to '{result_file_path}' (Using lock: '{lock_file}')")
    try:
        with file_lock_manager(lock_file, timeout=lock_timeout):
            # --- Lock Acquired for the ELO results file ---
            logger.debug(f"Lock acquired for ELO results file '{result_file_path}'. Performing read-modify-write.")
            
            current_data = _read_json_internal(result_file_path)

            # Initialize or validate structure of current_data
            if not isinstance(current_data, dict):
                logger.warning(f"No valid dictionary data in '{result_file_path}' or file is new. Initializing ELO results structure.")
                current_data = {"models": {}, "timestamp": None}
            
            if "models" not in current_data or not isinstance(current_data["models"], dict):
                logger.warning(f"Existing ELO data in '{result_file_path}' missing 'models' dictionary or it's not a dict. Re-initializing 'models'.")
                current_data["models"] = {} # Ensure 'models' is a dict

            # Deep merge the new ratings into the existing ones
            logger.debug(f"Merging {len(new_ratings_fragment)} new/updated model ratings into ELO results.")
            _deep_merge_dicts(current_data["models"], new_ratings_fragment)

            # Update timestamp
            current_data["timestamp"] = dt.datetime.now(dt.timezone.utc).isoformat() # Use UTC for consistency
            logger.debug(f"Updated ELO results timestamp to {current_data['timestamp']}.")

            # Write updated data back
            if _write_json_internal(current_data, result_file_path):
                logger.info(f"Successfully merged and saved ELO results to '{result_file_path}'.")
                return True
            else:
                logger.error(f"Failed during the _write_json_internal step for ELO results to '{result_file_path}'.")
                return False
            # --- Lock Released automatically by context manager ---
    except filelock.Timeout:
        # Error already logged by file_lock_manager
        return False
    except Exception as e:
        logger.error(f"Unexpected error during ELO results save operation for '{result_file_path}': {e}", exc_info=True)
        return False

# --- Dataset Saving (Handles list or Hugging Face Dataset, with append/overwrite logic) ---
def save_dataset_to_json(
    dataset_to_save: Union[List[Dict[str, Any]], Any], # Accepts list or HF Dataset-like object
    file_path: str,
    lock_file_path: Optional[str] = None, # Lock for this specific dataset file
    lock_timeout: int = 10
) -> bool:
    """
    Saves a dataset to a JSON file. Handles lists of dicts or Hugging Face Dataset objects.
    If the target JSON file exists and contains a list, new items are appended.
    Otherwise, a new file is created (or existing non-list file is overwritten).
    Uses file locking for process safety.
    """
    logger.info(f"Request to save/append dataset to '{file_path}'")

    new_data_list: List[Dict[str, Any]]
    if isinstance(dataset_to_save, list):
        # Ensure all items are dicts, or handle conversion if necessary
        if all(isinstance(item, dict) for item in dataset_to_save):
            new_data_list = dataset_to_save
        else:
            logger.warning(f"Input list for saving to '{file_path}' contains non-dictionary items. Attempting to filter.")
            new_data_list = [item for item in dataset_to_save if isinstance(item, dict)]
            if not new_data_list and dataset_to_save: # Original list was not empty but no dicts found
                 logger.error(f"No dictionary items found in the input list for '{file_path}'. Cannot save.")
                 return False
    elif hasattr(dataset_to_save, 'to_list') and callable(dataset_to_save.to_list): # Check for Hugging Face Dataset
        try:
            operation_start_time = time.monotonic()
            logger.debug(f"Converting Hugging Face Dataset-like object to list for saving to '{file_path}'. This might take time for large datasets.")
            new_data_list = dataset_to_save.to_list() # This can be memory intensive
            duration = time.monotonic() - operation_start_time
            logger.debug(f"Converted dataset to list with {len(new_data_list)} entries in {duration:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to convert Hugging Face Dataset-like object to list for '{file_path}': {e}", exc_info=True)
            return False
    else:
        logger.error(f"Unsupported dataset type for saving: {type(dataset_to_save)}. Must be a list of dictionaries or a Hugging Face Dataset-like object with a 'to_list' method.")
        return False

    # Determine the actual lock file path to use for this operation.
    # If no specific lock_file_path is provided, create one next to the data file.
    actual_dataset_lock_file = lock_file_path if lock_file_path else file_path + ".lock"
    
    final_data_to_save: List[Dict[str, Any]] = []

    try:
        with file_lock_manager(actual_dataset_lock_file, timeout=lock_timeout):
            logger.debug(f"Lock acquired for dataset save/append operation on '{file_path}'.")
            
            existing_data = _read_json_internal(file_path) # Internal read as lock is held

            if isinstance(existing_data, list):
                logger.info(f"Existing dataset file '{file_path}' found with {len(existing_data)} items. Appending {len(new_data_list)} new items.")
                final_data_to_save.extend(existing_data)
                final_data_to_save.extend(new_data_list)
            else: # No existing list data (file non-existent, empty, unreadable, or not a list)
                if existing_data is not None: # File existed but wasn't a list
                    logger.warning(f"Existing file '{file_path}' (type: {type(existing_data)}) is not a JSON list. It will be overwritten with the new dataset ({len(new_data_list)} items).")
                else: # File didn't exist or was empty/unreadable
                    logger.info(f"No existing valid list data at '{file_path}'. Creating new file with {len(new_data_list)} new items.")
                final_data_to_save.extend(new_data_list)
            
            if not final_data_to_save and not new_data_list:
                logger.warning(f"Dataset to save to '{file_path}' is empty, and no existing data was found. Saving an empty list.")
            
            # Attempt to write the combined or new list
            success = _write_json_internal(final_data_to_save, file_path) # Internal write
            if success:
                logger.info(f"Successfully saved dataset ({len(final_data_to_save)} total entries) to '{file_path}'.")
            else:
                logger.error(f"Failed to write the dataset list to '{file_path}' during the _write_json_internal step.")
            return success
            # --- Lock Released by context manager ---
    except filelock.Timeout:
        # Error already logged by file_lock_manager
        return False
    except Exception as e:
        logger.error(f"Unexpected error during dataset save/append operation for '{file_path}': {e}", exc_info=True)
        return False
