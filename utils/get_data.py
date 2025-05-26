import logging
import random
import os
import glob
from typing import Any, Dict, Optional, Tuple, Union, List, Set
import datasets
from datasets import load_dataset, Dataset, IterableDataset, DatasetDict, Features, Value

try:
    from .file_operations import save_dataset_to_json, load_json_file
except ImportError:
    from file_operations import save_dataset_to_json, load_json_file


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Global Cache
PROMPT_DATASET_CACHE: Optional[Union[Dataset, IterableDataset, DatasetDict]] = None
PROMPT_DATASET_NAME: Optional[str] = None
PROMPT_COLUMN: Optional[str] = None
GROUND_TRUTH_COLUMN: Optional[str] = None
CATEGORY_COLUMN: Optional[str] = None
PROMPT_ID_COLUMN: Optional[str] = None 

# --- Helper Functions  ---

def _get_dataset_config(config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], int, str, Optional[str], Optional[str]]:
    """Extract Dataset config"""
    dataset_name = config.get("PROMPT_DATASET")
    prompt_column = config.get("PROMPT_DATASET_COLUMN")
    ground_truth_column = config.get("GROUND_TRUTH_DATASET_COLUMN")
    category_column_name = config.get("PROMPT_DATASET_CATEGORY_COLUMN")
    prompt_id_column_name = config.get("PROMPT_ID_COLUMN") 
    num_of_entries = config.get("NUM_SAVE_DATASET_ENTRIES", 100)
    paths_config = config.get("paths", {})
    dataset_path_suffix = paths_config.get("dataset_file_suffix", "_prompts.json")
    category_base_dir = paths_config.get("dataset_category_dir")
    dataset_save_lock_path = paths_config.get("dataset_save_lock_file")
    logger.debug(f"Read PROMPT_ID_COLUMN from config: '{prompt_id_column_name}'")
    return (dataset_name, prompt_column, ground_truth_column, category_column_name,
            prompt_id_column_name, num_of_entries, dataset_path_suffix,
            category_base_dir, dataset_save_lock_path)

def _validate_columns(ds_columns: Set[str], required_cols: List[str], optional_cols: List[str] = None) -> bool:
    """Validate column names for dataset."""
    if optional_cols is None:
        optional_cols = []

    actual_required = [col for col in required_cols if col is not None]
    missing_columns = [col for col in actual_required if col not in ds_columns]
    if missing_columns:
        logger.error(f"Required columns {missing_columns} not found in dataset. Available columns: {list(ds_columns)}")
        return False
    logger.debug(f"Required columns ({actual_required}) found in dataset.")

    actual_optional = [col for col in optional_cols if col is not None]
    missing_optional_columns = [col for col in actual_optional if col not in ds_columns]
    if missing_optional_columns:
        logger.warning(f"Optional columns {missing_optional_columns} not found in dataset. Available columns: {list(ds_columns)}")
    else:
        if actual_optional:
            logger.debug(f"Optional columns ({actual_optional}) found in dataset or not configured.")
    return True


def _clean_data_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Clean a Dataset-Entry for constant types"""
    cleaned_entry = entry.copy()
    for key, value in cleaned_entry.items():
        if key == "year":
            try: cleaned_entry[key] = int(value) if value is not None and value != '' else None
            except (ValueError, TypeError): cleaned_entry[key] = None
        elif key == "hardness":
            try: cleaned_entry[key] = float(value) if value is not None and value != '' else None
            except (ValueError, TypeError): cleaned_entry[key] = None
        elif key == "turns":
            if not isinstance(value, list):
                 cleaned_entry[key] = [str(value)] if value is not None else []
            else:
                 cleaned_entry[key] = [str(item) if item is not None else "" for item in value]
    return cleaned_entry

def _determine_features_simplified(data: List[Dict[str, Any]]) -> Optional[Features]:
    if not data: return None
    all_keys = set()
    for entry in data:
        if isinstance(entry, dict): all_keys.update(entry.keys())
    if not all_keys: return None

    feature_dict = {}
    for key in all_keys:
        if key == "turns":
            feature_type = datasets.Sequence(Value("string"))
        elif key == "year": feature_type = Value("int64")
        elif key == "hardness": feature_type = Value("float64")
        else: feature_type = Value("string")
        feature_dict[key] = feature_type
    logger.debug(f"Using simplified schema for local loading: {feature_dict}")
    return Features(feature_dict)

def _load_from_local_json(category_base_dir: str, dataset_save_lock_path: Optional[str],
                          required_cols_for_op: List[str], optional_cols_for_op: List[str],
                          category) -> Optional[Dataset]:
    """Load and clean data from JSON file"""
    if not category_base_dir or not os.path.isdir(category_base_dir):
         logger.warning(f"Local category directory '{category_base_dir}' not found or not a directory.")
         return None

    logger.info(f"Attempting to load from local JSON files in '{category_base_dir}' for category '{category}'...")
    json_files = glob.glob(os.path.join(category_base_dir, f"*{category}_prompts.json")) # Ensure category is part of the pattern
    if not json_files:
        logger.info(f"No JSON files found for category '{category}' with suffix '_prompts.json' in '{category_base_dir}'.")
        return None
    logger.info(f"{len(json_files)} JSON files found for category '{category}': {[os.path.basename(f) for f in json_files]}")


    all_data_from_json: List[Dict] = []
    load_errors = False
    processed_files = 0
    for file_path in json_files:
        basename = os.path.basename(file_path)
        if (dataset_save_lock_path and os.path.abspath(file_path) == os.path.abspath(dataset_save_lock_path)) or \
           basename == "results.json":
            logger.debug(f"Skipping file: {basename}")
            continue
        if not os.path.isfile(file_path): continue

        logger.debug(f"Attempting to load file: {file_path}")
        try:
            loaded_data = load_json_file(file_path, lock_file_path=dataset_save_lock_path)
            if isinstance(loaded_data, list):
                if loaded_data:
                    cleaned_data = [_clean_data_entry(entry) for entry in loaded_data if isinstance(entry, dict)]
                    valid_cleaned_data = [d for d in cleaned_data if d is not None]
                    all_data_from_json.extend(valid_cleaned_data)
                    processed_files += 1
                else:
                     logger.warning(f"File '{basename}' is empty.")
            elif loaded_data is None:
                logger.warning(f"Loading '{basename}' returned None (possibly error or empty).")
                load_errors = True
            else:
                logger.error(f"Unexpected data type from '{basename}': {type(loaded_data)}. Skipping.")
                load_errors = True
        except Exception as e:
             logger.exception(f"Severe error processing file '{basename}': {e}")
             load_errors = True

    if not all_data_from_json:
        logger.error("Could not extract any valid data from local JSON files.")
        return None
    if load_errors:
        logger.warning("There were errors loading/processing some local JSON files.")

    logger.info(f"{len(all_data_from_json)} entries collected from {processed_files} files. Creating Dataset...")
    try:
        features = _determine_features_simplified(all_data_from_json)
        if features is None:
            logger.error("Could not determine a simplified schema from loaded JSON data.")
            return None
        ds_from_json = Dataset.from_list(all_data_from_json, features=features)
        logger.info(f"Dataset created with {len(ds_from_json)} rows and columns: {ds_from_json.column_names}")

        if not _validate_columns(set(ds_from_json.column_names), required_cols_for_op, optional_cols_for_op):
            if any(col not in ds_from_json.column_names for col in required_cols_for_op if col is not None):
                logger.error("Validation of required columns for local dataset failed. Cannot proceed with this dataset.")
                return None
        logger.info(f"Local dataset successfully created and validated.")
        return ds_from_json
    except Exception as e:
        logger.exception(f"Error creating Dataset from local JSON: {e}.")
        return None

def _save_categorized_samples(ds: Dataset, category_column_name: str, category_base_dir: str,
                              num_of_entries: int, dataset_path_suffix: str,
                              config: Dict[str, Any],
                              dataset_save_lock_path: Optional[str]):
    """Speichert ausgewählte Spalten von Samples des Datasets in JSON-Dateien."""
    if not isinstance(ds, Dataset):
        logger.warning("Categorized saving is only supported for 'Dataset' objects.")
        return

    columns_to_save = ds.column_names
    logger.info(f"Saving the following columns: {columns_to_save}")
    logger.info(f"Starting to save dataset samples by category '{category_column_name}' into '{category_base_dir}'...")
    try:
        os.makedirs(category_base_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create directory '{category_base_dir}': {e}. Skipping save.")
        return

    try:
        if category_column_name not in ds.column_names:
             logger.error(f"Category column '{category_column_name}' not found in dataset.")
             return
        unique_categories = set(ds[category_column_name])
        logger.info(f"{len(unique_categories)} unique categories found: {list(unique_categories)[:10]}...")

        for category in unique_categories:
            safe_category_name = "".join(c if c.isalnum() else "_" for c in str(category).lower())[:50]
            if not safe_category_name: safe_category_name = "unknown_category"
            base_filename = f"{safe_category_name}{dataset_path_suffix}"
            target_file_path = os.path.join(category_base_dir, base_filename)

            if os.path.exists(target_file_path):
                 logger.warning(f"File '{target_file_path}' already exists and will be overwritten.")

            try:
                filtered_ds = ds.filter(lambda ex: ex[category_column_name] == category, num_proc=1) 
                num_filtered = len(filtered_ds)
                if num_filtered > 0:
                    dataset_to_process = filtered_ds
                    if num_filtered > num_of_entries:
                        dataset_to_process = filtered_ds.shuffle(seed=42).select(range(num_of_entries))
                    if save_dataset_to_json(dataset_to_process, target_file_path, dataset_save_lock_path):
                        logger.info(f"Category '{category}' successfully saved to '{target_file_path}'.")
                    else:
                        logger.error(f"Error saving category '{category}' to '{target_file_path}'.")
            except Exception as filter_err:
                logger.exception(f"Error processing/saving for category '{category}': {filter_err}")
    except Exception as cat_err:
         logger.exception(f"General error during categorized saving: {cat_err}")


# --- Main-Laodingfunction ---
def _load_prompt_dataset(config: Dict[str, Any], current_category) -> Optional[Union[Dataset, IterableDataset]]:
    """
    Load Prompt-Dataset (raw), prioritize local JSON, then Cache, then Hugging Face.
    """
    global PROMPT_DATASET_CACHE, PROMPT_DATASET_NAME, PROMPT_COLUMN, GROUND_TRUTH_COLUMN, CATEGORY_COLUMN, PROMPT_ID_COLUMN

    (dataset_name, prompt_column, ground_truth_column, category_column_name,
     prompt_id_column_config, num_of_entries, dataset_path_suffix,
     category_base_dir, dataset_save_lock_path) = _get_dataset_config(config)

    logger.info(f"Starting dataset load. Configured PROMPT_ID_COLUMN: '{prompt_id_column_config}'")

    if not prompt_column:
        logger.critical("PROMPT_DATASET_COLUMN is missing in configuration. Cannot load dataset.")
        return None

    required_cols_for_op = [prompt_column]
    optional_cols_for_op = []
    if ground_truth_column: required_cols_for_op.append(ground_truth_column)
    if category_column_name: required_cols_for_op.append(category_column_name)
    if prompt_id_column_config: optional_cols_for_op.append(prompt_id_column_config)

    # --- Attempt 1.: From local JSON ---
    if category_base_dir:
        logger.info("--- Attempt 1: Load dataset from local JSON files ---")
        ds_from_json = _load_from_local_json(category_base_dir, dataset_save_lock_path,
                                             required_cols_for_op, optional_cols_for_op, current_category)
        if ds_from_json:
            logger.info("Dataset successfully loaded from local files and will be used.")
            PROMPT_DATASET_CACHE = ds_from_json
            PROMPT_DATASET_NAME = f"local_json:{category_base_dir}/{current_category}"
            PROMPT_COLUMN = prompt_column
            GROUND_TRUTH_COLUMN = ground_truth_column if ground_truth_column and ground_truth_column in ds_from_json.column_names else None
            CATEGORY_COLUMN = category_column_name if category_column_name and category_column_name in ds_from_json.column_names else None
            PROMPT_ID_COLUMN = prompt_id_column_config if prompt_id_column_config and prompt_id_column_config in ds_from_json.column_names else None
            logger.info(f"Local JSON load: PROMPT_ID_COLUMN set to '{PROMPT_ID_COLUMN}' (Config was '{prompt_id_column_config}', Found in DS: {prompt_id_column_config in ds_from_json.column_names if prompt_id_column_config else 'N/A'})")
            return ds_from_json
        else:
            logger.info("No valid dataset created from local files for category '{current_category}'.")

    # --- Attempt 2.:  From In-Memory-Cache (only for Hugging Face Dataset) ---
    logger.info("--- Attempt 2: Check cache for Hugging Face Dataset ---")
    is_cache_valid = (
        PROMPT_DATASET_CACHE is not None and
        PROMPT_DATASET_NAME == dataset_name and 
        PROMPT_COLUMN == prompt_column 
    )
    if is_cache_valid:
        # Validate columns in the cached dataset against current config (especially for optional ones)
        if _validate_columns(set(PROMPT_DATASET_CACHE.column_names), required_cols_for_op, optional_cols_for_op):
             logger.info(f"Using cached Hugging Face dataset '{dataset_name}'.")
             # Ensure global column names are correctly set based on the cache and current config
             PROMPT_COLUMN = prompt_column 
             GROUND_TRUTH_COLUMN = ground_truth_column if ground_truth_column and ground_truth_column in PROMPT_DATASET_CACHE.column_names else None
             CATEGORY_COLUMN = category_column_name if category_column_name and category_column_name in PROMPT_DATASET_CACHE.column_names else None
             PROMPT_ID_COLUMN = prompt_id_column_config if prompt_id_column_config and prompt_id_column_config in PROMPT_DATASET_CACHE.column_names else None
             logger.info(f"Cache hit: PROMPT_ID_COLUMN set to '{PROMPT_ID_COLUMN}' (Config was '{prompt_id_column_config}', Found in cached DS: {prompt_id_column_config in PROMPT_DATASET_CACHE.column_names if prompt_id_column_config else 'N/A'})")
             return PROMPT_DATASET_CACHE
        else:
             logger.warning("Cache found, but column validation failed against current config. Reloading.")
             PROMPT_DATASET_CACHE = None 
    else:
        logger.info("No valid cache for Hugging Face dataset found or cache invalidated.")


    # --- Attempt 3.: Load from Hugging Face (roh) ---
    if not dataset_name:
         logger.error("No local dataset found and PROMPT_DATASET (for Hugging Face) is not configured.")
         return None

    try:
        loaded_object = load_dataset(dataset_name, trust_remote_code=True) 
        ds: Optional[Union[Dataset, IterableDataset]] = None

        if isinstance(loaded_object, DatasetDict):
            split_preference = ["test", "validation", "train"]
            found_split = None
            for split_name in split_preference:
                if split_name in loaded_object:
                    ds = loaded_object[split_name]
                    found_split = split_name
                    break
            if not ds:
                 first_split_key = next(iter(loaded_object.keys()), None)
                 if first_split_key:
                     ds = loaded_object[first_split_key]
                     logger.warning(f"Using first available split: '{first_split_key}'.")
                 else:
                     logger.error(f"No splits found in DatasetDict '{dataset_name}'.")
                     return None
            logger.info(f"Using '{found_split or first_split_key}' split from DatasetDict '{dataset_name}'.")
        elif isinstance(loaded_object, (Dataset, IterableDataset)):
             ds = loaded_object
        else:
            logger.error(f"load_dataset returned an unexpected type: {type(loaded_object)}")
            return None

        if not _validate_columns(set(ds.column_names), required_cols_for_op, optional_cols_for_op):
             if any(col not in ds.column_names for col in required_cols_for_op if col is not None):
                 logger.error("Required columns not found in the Hugging Face dataset. Cannot use this dataset.")
                 return None

        PROMPT_DATASET_CACHE = ds
        PROMPT_DATASET_NAME = dataset_name
        PROMPT_COLUMN = prompt_column
        GROUND_TRUTH_COLUMN = ground_truth_column if ground_truth_column and ground_truth_column in ds.column_names else None
        CATEGORY_COLUMN = category_column_name if category_column_name and category_column_name in ds.column_names else None
        PROMPT_ID_COLUMN = prompt_id_column_config if prompt_id_column_config and prompt_id_column_config in ds.column_names else None
        logger.info(f"Hugging Face load: PROMPT_ID_COLUMN set to '{PROMPT_ID_COLUMN}' (Config was '{prompt_id_column_config}', Found in DS: {prompt_id_column_config in ds.column_names if prompt_id_column_config else 'N/A'})")
        logger.info(f"Raw dataset '{dataset_name}' successfully loaded from Hugging Face and cached.")

        can_save_categorized = (category_base_dir and category_column_name and
                                isinstance(ds, Dataset) and category_column_name in ds.column_names)
        if can_save_categorized:
             logger.info("Saving samples of the raw Hugging Face dataset locally by categories...")
             _save_categorized_samples(ds, category_column_name, category_base_dir,
                                       num_of_entries, dataset_path_suffix, config, dataset_save_lock_path)
        return ds

    except Exception as e:
        logger.exception(f"Critical error loading '{dataset_name}' from Hugging Face: {e}")
        PROMPT_DATASET_CACHE = None; PROMPT_DATASET_NAME = None; PROMPT_COLUMN = None
        GROUND_TRUTH_COLUMN = None; CATEGORY_COLUMN = None; PROMPT_ID_COLUMN = None
        return None
     
     
def extract_information_from_dataset_prompt(dataset_item:dict) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    
    global PROMPT_COLUMN, GROUND_TRUTH_COLUMN, CATEGORY_COLUMN, PROMPT_ID_COLUMN
    
    if not dataset_item:
        logger.warning("extract_dataset_item_info has empty or None for dataset_item.")
        return None, None, None, None

    prompt_content_col = PROMPT_COLUMN
    ground_truth_col = GROUND_TRUTH_COLUMN
    prompt_id_col = PROMPT_ID_COLUMN 

    
    prompt_data = dataset_item.get(prompt_content_col)
    if prompt_data is None:
        logger.debug(
            f"Prompt content not found for Key '{prompt_content_col}'. "
            f"Key with item: {list(dataset_item.keys())}"
        )
        
    if ground_truth_col:
        gt_data = dataset_item.get(ground_truth_col)
        if gt_data is None:
            logger.debug(
                f"Key for Ground-Truth '{ground_truth_col}' passed, but not found in Item. "
                f"Key with item: {list(dataset_item.keys())}"
            )
    prompt_id = dataset_item.get(prompt_id_col)
    if prompt_id is None:
        logger.debug(
            f"Prompt-ID is missing '{prompt_id_col}'. "
            f"Key with item: {list(dataset_item.keys())}"
        )
    
    return prompt_data, gt_data, prompt_id
 

    
     



