import logging
import os
import glob # For finding files
from typing import Any, Dict, Optional, Tuple, Union, List, Set

import datasets
from datasets import load_dataset, Dataset, IterableDataset, DatasetDict, Features, Value

# Assuming file_operations is in the same directory or accessible in PYTHONPATH
try:
    # Use relative import if this is part of a package
    from .file_operations import save_dataset_to_json, load_json_file
except ImportError:
    # Fallback for standalone script execution (less common for utils)
    from file_operations import save_dataset_to_json, load_json_file

logger = logging.getLogger(__name__)
# BasicConfig should ideally be set up in the main application entry point.
# Adding a check to prevent multiple configurations if this module is reloaded.
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# --- Configuration and Cache Management ---
class DatasetConfigManager:
    """
    Manages dataset configuration and caches the loaded dataset and its active properties.
    This helps in centralizing how dataset parameters are accessed post-loading.
    """
    def __init__(self, app_config: Dict[str, Any]):
        # Raw config values
        self.raw_dataset_name: Optional[str] = app_config.get("PROMPT_DATASET")
        self.raw_prompt_column: Optional[str] = app_config.get("PROMPT_DATASET_COLUMN")
        self.raw_ground_truth_column: Optional[str] = app_config.get("GROUND_TRUTH_DATASET_COLUMN")
        self.raw_category_column: Optional[str] = app_config.get("PROMPT_DATASET_CATEGORY_COLUMN")
        self.raw_prompt_id_column: Optional[str] = app_config.get("PROMPT_ID_COLUMN")
        self.num_save_entries: int = app_config.get("NUM_SAVE_DATASET_ENTRIES", 100) # Default to 100
        
        paths_config = app_config.get("paths", {})
        self.dataset_file_suffix: str = paths_config.get("dataset_file_suffix", "_prompts.json")
        self.category_base_dir: Optional[str] = paths_config.get("dataset_category_dir")
        self.dataset_save_lock_file: Optional[str] = paths_config.get("dataset_save_lock_file")

        # --- Cached state for the currently loaded dataset ---
        self._cached_dataset: Optional[Union[Dataset, IterableDataset, DatasetDict]] = None
        self._cached_dataset_source_name: Optional[str] = None # e.g., HF name or local path

        # Active (validated) column names for the _cached_dataset
        self._active_prompt_column: Optional[str] = None
        self._active_ground_truth_column: Optional[str] = None
        self._active_category_column: Optional[str] = None
        self._active_prompt_id_column: Optional[str] = None

    def _update_active_config(self, dataset: Union[Dataset, IterableDataset], source_name: str):
        """Internal: Updates active column names based on the successfully loaded dataset."""
        # Ensure dataset is not None and has column_names before proceeding
        if dataset is None or not hasattr(dataset, 'column_names'):
            logger.error(f"Cannot update active config; provided dataset is invalid for source '{source_name}'.")
            # Reset active columns to avoid using stale data
            self._active_prompt_column = None
            self._active_ground_truth_column = None
            self._active_category_column = None
            self._active_prompt_id_column = None
            self._cached_dataset_source_name = None # Or keep source_name but mark as invalid
            return

        ds_columns = set(dataset.column_names)
        self._active_prompt_column = self.raw_prompt_column # This is mandatory
        self._active_ground_truth_column = self.raw_ground_truth_column if self.raw_ground_truth_column and self.raw_ground_truth_column in ds_columns else None
        self._active_category_column = self.raw_category_column if self.raw_category_column and self.raw_category_column in ds_columns else None
        self._active_prompt_id_column = self.raw_prompt_id_column if self.raw_prompt_id_column and self.raw_prompt_id_column in ds_columns else None
        self._cached_dataset_source_name = source_name
        
        logger.info(
            f"Active configuration updated for dataset '{source_name}':\n"
            f"  Prompt Column: '{self._active_prompt_column}' (Raw: '{self.raw_prompt_column}')\n"
            f"  Ground Truth Column: '{self._active_ground_truth_column}' (Raw: '{self.raw_ground_truth_column}')\n"
            f"  Category Column: '{self._active_category_column}' (Raw: '{self.raw_category_column}')\n"
            f"  Prompt ID Column: '{self._active_prompt_id_column}' (Raw: '{self.raw_prompt_id_column}')"
        )

    def set_cached_dataset(self, dataset: Optional[Union[Dataset, IterableDataset]], source_name: Optional[str]):
        """Sets or clears the cached dataset and updates active configuration."""
        self._cached_dataset = dataset
        if dataset is not None and source_name is not None:
            self._update_active_config(dataset, source_name)
        else: # Clearing cache or invalid dataset
            self._active_prompt_column = None
            self._active_ground_truth_column = None
            self._active_category_column = None
            self._active_prompt_id_column = None
            self._cached_dataset_source_name = None
            logger.info("Cached dataset and active configuration cleared.")


    def get_cached_dataset(self, expected_source_name: Optional[str] = None) -> Optional[Union[Dataset, IterableDataset]]:
        """
        Returns the cached dataset if it matches the expected source name (e.g., Hugging Face dataset name).
        This is primarily for caching the full Hugging Face dataset.
        """
        if self._cached_dataset and \
           (expected_source_name is None or self._cached_dataset_source_name == expected_source_name) and \
           self._active_prompt_column == self.raw_prompt_column: # Ensure prompt column config hasn't changed
            # Additional check for validity of the cached object
            if hasattr(self._cached_dataset, 'column_names'):
                return self._cached_dataset
            else:
                logger.warning(f"Cached dataset for '{self._cached_dataset_source_name}' is invalid (missing column_names). Clearing cache.")
                self.set_cached_dataset(None, None) # Clear invalid cache
                return None
        return None

    def get_active_prompt_column(self) -> Optional[str]: return self._active_prompt_column
    def get_active_ground_truth_column(self) -> Optional[str]: return self._active_ground_truth_column
    def get_active_category_column(self) -> Optional[str]: return self._active_category_column
    def get_active_prompt_id_column(self) -> Optional[str]: return self._active_prompt_id_column
    def get_active_dataset_source_name(self) -> Optional[str]: return self._cached_dataset_source_name

# Global instance of the config manager for this module
_dataset_config_manager: Optional[DatasetConfigManager] = None

def _get_config_manager(app_config: Optional[Dict[str, Any]] = None) -> DatasetConfigManager:
    """Initializes or returns the global dataset config manager."""
    global _dataset_config_manager
    if _dataset_config_manager is None:
        if app_config is None:
            # This state should ideally be avoided by ensuring app_config is passed on first call.
            logger.error("DatasetConfigManager accessed before initialization with app_config.")
            raise ValueError("Application configuration must be provided for first-time DatasetConfigManager initialization.")
        _dataset_config_manager = DatasetConfigManager(app_config)
    elif app_config is not None:
        # If app_config is provided again, it might mean a re-configuration.
        # Re-initialize. More complex scenarios might need selective updates or a different pattern.
        logger.info("Re-initializing DatasetConfigManager with new application config.")
        _dataset_config_manager = DatasetConfigManager(app_config)
    return _dataset_config_manager

# --- Helper Functions ---
def _validate_columns(ds_columns: Set[str], required_cols: List[Optional[str]], optional_cols: List[Optional[str]] = None) -> bool:
    """Validates if required and optional columns are present in the dataset."""
    optional_cols = optional_cols or []
    
    actual_required = [col for col in required_cols if col] # Filter out None if a config value isn't set
    missing_required = [col for col in actual_required if col not in ds_columns]
    if missing_required:
        logger.error(f"Required columns {missing_required} not found in dataset. Available: {list(ds_columns)}")
        return False
    logger.debug(f"All required columns ({actual_required}) found in dataset.")

    actual_optional = [col for col in optional_cols if col]
    missing_optional = [col for col in actual_optional if col not in ds_columns]
    if missing_optional:
        logger.warning(f"Optional columns {missing_optional} not found. Available: {list(ds_columns)}")
    return True

def _clean_data_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Cleans a single dataset entry, ensuring consistent types for known problematic keys."""
    cleaned_entry = entry.copy()
    # Example cleanups, expand as needed based on dataset specifics
    for key, value in cleaned_entry.items():
        if key == "year":
            try: cleaned_entry[key] = int(value) if value is not None and str(value).strip() != '' else None
            except (ValueError, TypeError): cleaned_entry[key] = None; logger.debug(f"Could not convert '{value}' to int for key '{key}'")
        elif key == "hardness":
            try: cleaned_entry[key] = float(value) if value is not None and str(value).strip() != '' else None
            except (ValueError, TypeError): cleaned_entry[key] = None; logger.debug(f"Could not convert '{value}' to float for key '{key}'")
        elif key == "turns": # Assuming 'turns' should be a list of strings
            if not isinstance(value, list):
                cleaned_entry[key] = [str(value)] if value is not None else []
            else:
                cleaned_entry[key] = [str(item) if item is not None else "" for item in value]
    return cleaned_entry

def _determine_features_from_data(data: List[Dict[str, Any]]) -> Optional[Features]:
    """Determines a Hugging Face Features schema from a list of data dictionaries."""
    if not data: return None
    
    all_keys = set()
    for entry in data:
        if isinstance(entry, dict): all_keys.update(entry.keys())
    if not all_keys: return None

    feature_dict = {}
    # Define types for known special keys; default to string for others.
    # This can be expanded or made more dynamic if necessary.
    # Ensure prompt_id is treated as string for flexibility unless strictly numeric.
    special_key_types = {
        "turns": datasets.Sequence(Value("string")), # Example: if 'turns' is a list of strings
        "year": Value("int64"),
        "hardness": Value("float64"),
        "prompt_id": Value("string") # Common practice for IDs
    }
    for key in all_keys:
        feature_dict[key] = special_key_types.get(key, Value("string"))
        
    logger.debug(f"Determined features for local JSON loading: {feature_dict}")
    return Features(feature_dict)

def _load_from_local_category_json(dcm: DatasetConfigManager, current_category: str) -> Optional[Dataset]:
    """Loads dataset from a local JSON file specific to the current_category."""
    if not dcm.category_base_dir or not os.path.isdir(dcm.category_base_dir):
        logger.debug(f"Local category directory '{dcm.category_base_dir}' not found or not a directory.")
        return None

    category_file_name = f"{current_category.lower()}{dcm.dataset_file_suffix}"
    category_file_path = os.path.join(dcm.category_base_dir, category_file_name)

    logger.info(f"Attempting to load local JSON for category '{current_category}' from: '{category_file_path}'")

    if not os.path.isfile(category_file_path):
        logger.info(f"No local JSON file found at '{category_file_path}'.")
        return None

    loaded_data_list = load_json_file(category_file_path, lock_file_path=dcm.dataset_save_lock_file)

    if not isinstance(loaded_data_list, list) or not loaded_data_list:
        logger.warning(f"File '{category_file_path}' is empty, not a list, or failed to load. Cannot create dataset.")
        return None

    cleaned_data = [_clean_data_entry(entry) for entry in loaded_data_list if isinstance(entry, dict)]
    if not cleaned_data:
        logger.error(f"No valid dictionary entries found after cleaning data from '{category_file_path}'.")
        return None
        
    logger.info(f"{len(cleaned_data)} entries loaded from '{category_file_path}'. Creating Dataset...")
    
    try:
        features = _determine_features_from_data(cleaned_data)
        if not features:
            logger.error("Could not determine features from loaded local JSON data.")
            return None

        # Ensure all entries conform to the derived features, especially if data is heterogeneous
        conforming_data = [{key: entry.get(key) for key in features} for entry in cleaned_data]

        ds_from_json = Dataset.from_list(conforming_data, features=features)
        logger.info(f"Local dataset for '{current_category}' created with {len(ds_from_json)} rows. Columns: {ds_from_json.column_names}")

        required_cols = [dcm.raw_prompt_column] # Prompt column is always required
        optional_cols = [dcm.raw_ground_truth_column, dcm.raw_category_column, dcm.raw_prompt_id_column]
        if not _validate_columns(set(ds_from_json.column_names), required_cols, optional_cols):
            logger.error("Validation of columns for local dataset failed. Cannot use this dataset.")
            return None
        
        # Update active config in the manager for this successfully loaded local dataset
        dcm._update_active_config(ds_from_json, f"local_json:{category_file_path}")
        return ds_from_json
    except Exception as e:
        logger.exception(f"Error creating Dataset from local JSON '{category_file_path}': {e}")
        return None

def _save_samples_by_category(full_ds: Dataset, dcm: DatasetConfigManager):
    """Filters the full_ds by category and saves samples to local JSON files."""
    if not dcm.raw_category_column or dcm.raw_category_column not in full_ds.column_names:
        logger.error(f"Category column '{dcm.raw_category_column}' not in dataset. Cannot save by category.")
        return
    if not dcm.category_base_dir:
        logger.error("Category base directory not configured. Cannot save by category.")
        return

    logger.info(f"Saving samples from Hugging Face dataset by category '{dcm.raw_category_column}' to '{dcm.category_base_dir}'...")
    os.makedirs(dcm.category_base_dir, exist_ok=True)
    
    try:
        unique_categories = set(full_ds[dcm.raw_category_column])
        logger.info(f"Found {len(unique_categories)} unique categories in the dataset: {list(unique_categories)[:5]}...")

        for category_value in unique_categories:
            safe_category_name = "".join(c if c.isalnum() else "_" for c in str(category_value).lower())[:50]
            if not safe_category_name: safe_category_name = "unknown_category"
            
            target_filename = f"{safe_category_name}{dcm.dataset_file_suffix}"
            target_file_path = os.path.join(dcm.category_base_dir, target_filename)
            
            logger.debug(f"Processing category: '{category_value}' (Safe name: '{safe_category_name}')")
            
            # Filter for the current category
            # Using a potentially multi-processed filter if num_proc > 1
            num_proc = os.cpu_count() if os.cpu_count() and os.cpu_count() > 1 else None
            category_ds = full_ds.filter(lambda ex: ex[dcm.raw_category_column] == category_value, num_proc=num_proc)
            
            num_filtered = len(category_ds)
            if num_filtered == 0:
                logger.info(f"No entries for category '{category_value}'. Skipping save for this category.")
                continue

            dataset_to_save = category_ds
            if dcm.num_save_entries > 0 and num_filtered > dcm.num_save_entries:
                logger.info(f"Category '{category_value}' has {num_filtered} entries. Shuffling and selecting {dcm.num_save_entries}.")
                dataset_to_save = category_ds.shuffle(seed=42).select(range(dcm.num_save_entries))
            else:
                logger.info(f"Category '{category_value}' has {num_filtered} entries. Saving all.")

            # save_dataset_to_json handles append/overwrite logic and locking
            if save_dataset_to_json(dataset_to_save, target_file_path, dcm.dataset_save_lock_file):
                logger.info(f"Data for category '{category_value}' successfully saved/appended to '{target_file_path}'.")
            else:
                logger.error(f"Failed to save/append data for category '{category_value}' to '{target_file_path}'.")
    except Exception as e:
        logger.exception(f"General error during categorized saving of Hugging Face dataset: {e}")

# --- Main Loading Function ---
def load_prompt_dataset(app_config: Dict[str, Any], current_category: str) -> Optional[Union[Dataset, IterableDataset]]:
    """
    Loads a dataset for a specific category, managing configuration and caching.
    Sources, in order of preference:
    1. Local JSON file for the specific `current_category`.
    2. In-memory cache of the full Hugging Face dataset (if previously loaded and applicable).
    3. Hugging Face Hub (loads full dataset, then filters for `current_category`).
       - If loaded from Hugging Face, samples are saved locally by category for future fast loads.
    """
    dcm = _get_config_manager(app_config) # Initialize or get existing manager

    if not dcm.raw_prompt_column:
        logger.critical("PROMPT_DATASET_COLUMN is not configured. Cannot load dataset.")
        return None

    # --- Attempt 1: Load from specific local JSON for the current_category ---
    logger.info(f"--- Attempt 1: Load dataset from local JSON for category '{current_category}' ---")
    # This function will update dcm's active columns if successful
    ds_from_local_category = _load_from_local_category_json(dcm, current_category)
    if ds_from_local_category:
        logger.info(f"Dataset for '{current_category}' loaded from local JSON: '{dcm.get_active_dataset_source_name()}'")
        return ds_from_local_category
    logger.info(f"No local JSON dataset found for '{current_category}'.")

    # --- Attempt 2: Check In-Memory Cache for the full Hugging Face Dataset ---
    logger.info(f"--- Attempt 2: Check cache for full Hugging Face Dataset ('{dcm.raw_dataset_name}') ---")
    cached_full_hf_ds = dcm.get_cached_dataset(expected_source_name=dcm.raw_dataset_name)
    
    if cached_full_hf_ds: # This implies cached_full_hf_ds is not None and has .column_names
        logger.info(f"Full Hugging Face dataset '{dcm.raw_dataset_name}' found in cache.")
        if not dcm.raw_category_column or dcm.raw_category_column not in cached_full_hf_ds.column_names:
            logger.warning(f"Category column '{dcm.raw_category_column}' not in cached HF dataset. Cannot filter from cache.")
        else:
            try:
                logger.info(f"Filtering cached HF dataset for category '{current_category}' using column '{dcm.raw_category_column}'.")
                # Type check already implicitly handled by get_cached_dataset and hasattr checks within it.
                filtered_ds = cached_full_hf_ds.filter(lambda ex: ex[dcm.raw_category_column] == current_category)
                if len(filtered_ds) > 0:
                    logger.info(f"Successfully filtered cached HF dataset for '{current_category}' ({len(filtered_ds)} entries).")
                    dcm._update_active_config(filtered_ds, dcm.raw_dataset_name + f" [filtered for {current_category}]")
                    return filtered_ds
                else:
                    logger.warning(f"No entries for category '{current_category}' in cached HF dataset.")
            except Exception as e:
                logger.exception(f"Error filtering cached HF dataset for '{current_category}': {e}. Invalidating cache.")
                dcm.set_cached_dataset(None, None) # Invalidate cache on error
    else:
        logger.info("No valid cache for full Hugging Face dataset, or config mismatch.")

    # --- Attempt 3: Load from Hugging Face Hub (full dataset) ---
    if not dcm.raw_dataset_name:
        logger.error("No local category dataset found, and PROMPT_DATASET (for Hugging Face) is not configured.")
        return None
    
    logger.info(f"--- Attempt 3: Load full dataset '{dcm.raw_dataset_name}' from Hugging Face Hub ---")
    full_hf_ds: Optional[Union[Dataset, IterableDataset]] = None # Initialize
    try:
        load_params = {"path": dcm.raw_dataset_name, "trust_remote_code": True}
        
        # Specific handling for "VAGOsolutions/MT-Bench-TrueGerman"
        # This dataset might require specifying which file to load if it doesn't have a default config
        # that correctly points to 'question_de.jsonl' or if it tries to load all JSONL files.
        if dcm.raw_dataset_name == "VAGOsolutions/MT-Bench-TrueGerman":
            logger.info(f"Applying specific data_files configuration for {dcm.raw_dataset_name} to target 'question_de.jsonl'.")
            # This tells load_dataset to look for 'question_de.jsonl' within the dataset repository
            # and treat it as the 'train' split. The dataset's loading script (if any)
            # or the default json loader should handle this.
            load_params["data_files"] = {"train": "question_de.jsonl"}
            # If this still causes issues, it might be necessary to specify the data type, e.g., by adding:
            # load_params["type"] = "json" # Or just "json" if path becomes the type
            # However, usually, if 'path' is a Hub identifier, its script handles file types.
            # If 'data_files' is used with a Hub identifier, it typically means the dataset's
            # loading script will use these file names relative to the dataset root.

        logger.info(f"Calling load_dataset with params: {load_params}")
        loaded_object = load_dataset(**load_params)
        logger.debug(f"load_dataset returned object of type: {type(loaded_object)}")
        
        if isinstance(loaded_object, DatasetDict):
            logger.debug(f"Loaded object is a DatasetDict. Keys: {list(loaded_object.keys())}")
            # If we specified data_files={"train": ...}, 'train' should be a key.
            split_preference = ["train", "test", "validation"] 
            chosen_split_name = None
            for name in split_preference:
                if name in loaded_object:
                    full_hf_ds = loaded_object[name]
                    chosen_split_name = name
                    logger.info(f"Selected split '{chosen_split_name}' from DatasetDict.")
                    break
            if not full_hf_ds: # Fallback to first available split if preferred ones not found
                first_key = next(iter(loaded_object.keys()), None)
                if first_key: 
                    full_hf_ds = loaded_object[first_key]
                    chosen_split_name = first_key
                    logger.warning(f"Using first available split from DatasetDict: '{chosen_split_name}'.")
                else: # Should not happen if load_dataset succeeded with data_files
                    logger.error(f"No usable splits found in DatasetDict '{dcm.raw_dataset_name}' even after specifying data_files.")
                    # 'full_hf_ds' remains None
        elif isinstance(loaded_object, (Dataset, IterableDataset)):
            logger.debug("Loaded object is a Dataset or IterableDataset directly.")
            full_hf_ds = loaded_object
        else:
            logger.error(f"load_dataset('{dcm.raw_dataset_name}') returned unexpected type: {type(loaded_object)}")
            # 'full_hf_ds' remains None

        # CRITICAL CHECK: Ensure full_hf_ds is not None and has column_names before proceeding
        if full_hf_ds is None or not hasattr(full_hf_ds, 'column_names'):
            logger.error(f"Failed to load or select a valid Dataset object from '{dcm.raw_dataset_name}'. Object is None or lacks 'column_names'.")
            dcm.set_cached_dataset(None, None) # Ensure cache is cleared
            return None

        # Validate essential columns (prompt and category for filtering)
        required_hf_cols = [dcm.raw_prompt_column, dcm.raw_category_column]
        optional_hf_cols = [dcm.raw_ground_truth_column, dcm.raw_prompt_id_column]
        if not _validate_columns(set(full_hf_ds.column_names), required_hf_cols, optional_hf_cols):
            logger.error(f"Essential columns missing in loaded Hugging Face dataset '{dcm.raw_dataset_name}'. Cannot proceed.")
            dcm.set_cached_dataset(None, None) # Ensure cache is cleared
            return None
        
        dcm.set_cached_dataset(full_hf_ds, dcm.raw_dataset_name) # Cache the full loaded dataset
        logger.info(f"Full dataset '{dcm.raw_dataset_name}' loaded from Hugging Face and cached.")

        # Save samples locally by category for future faster loads
        if isinstance(full_hf_ds, Dataset): # IterableDataset is harder to save this way
            _save_samples_by_category(full_hf_ds, dcm)
        
        # Now, filter for the current_category from the newly loaded full_hf_ds
        if dcm.raw_category_column and dcm.raw_category_column in full_hf_ds.column_names:
            logger.info(f"Filtering newly loaded HF dataset for category '{current_category}'.")
            category_specific_ds = full_hf_ds.filter(lambda ex: ex[dcm.raw_category_column] == current_category)
            if len(category_specific_ds) > 0:
                logger.info(f"Successfully filtered HF dataset for '{current_category}' ({len(category_specific_ds)} entries).")
                dcm._update_active_config(category_specific_ds, dcm.raw_dataset_name + f" [filtered for {current_category}]")
                return category_specific_ds
            else:
                logger.warning(f"No entries for category '{current_category}' in HF dataset '{dcm.raw_dataset_name}'.")
                # Return an empty dataset of the same type if appropriate, or None
                # For simplicity, returning None if category is not found.
                return None
        else:
            logger.error(f"Category column '{dcm.raw_category_column}' not found in HF dataset. Cannot filter.")
            return None

    except Exception as e:
        logger.exception(f"Critical error loading '{dcm.raw_dataset_name}' from Hugging Face: {e}")
        dcm.set_cached_dataset(None, None) # Clear cache on error
        return None

# --- Public Accessors for Active Configuration (used by other modules) ---
def get_active_prompt_column() -> Optional[str]:
    """Returns the active prompt column name from the config manager."""
    manager = _get_config_manager() # Relies on manager being initialized
    return manager.get_active_prompt_column() if manager else None

def get_active_ground_truth_column() -> Optional[str]:
    manager = _get_config_manager()
    return manager.get_active_ground_truth_column() if manager else None
    
def get_active_category_column() -> Optional[str]:
    manager = _get_config_manager()
    return manager.get_active_category_column() if manager else None

def get_active_prompt_id_column() -> Optional[str]:
    manager = _get_config_manager()
    return manager.get_active_prompt_id_column() if manager else None

def get_active_dataset_source_name() -> Optional[str]:
    manager = _get_config_manager()
    return manager.get_active_dataset_source_name() if manager else None

# --- Function to Extract Information from a Single Dataset Item ---
def extract_prompt_info_from_dataset_item(
    dataset_item: Dict[str, Any]
) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """
    Extracts prompt, ground truth (optional), and prompt ID (optional) from a dataset item,
    using the currently active column names from the DatasetConfigManager.
    """
    manager = _get_config_manager() # Relies on manager being initialized
    if not manager:
        logger.critical("DatasetConfigManager not initialized. Call load_prompt_dataset with app_config first.")
        # This state indicates a programming error if extract is called before load.
        return None, None, None
        
    prompt_col = manager.get_active_prompt_column()
    gt_col = manager.get_active_ground_truth_column()
    id_col = manager.get_active_prompt_id_column()

    if not prompt_col: # Should always be available if dataset loaded successfully
        logger.error("Active prompt column name is not set. Cannot extract info. Was dataset loaded successfully?")
        return None, None, None

    prompt_data = dataset_item.get(prompt_col)
    gt_data = dataset_item.get(gt_col) if gt_col else None
    prompt_id = dataset_item.get(id_col) if id_col else None
    
    if prompt_data is None:
        logger.warning(f"Prompt data not found in item using active column '{prompt_col}'. Item keys: {list(dataset_item.keys())}")
    # Similar warnings can be added for gt_data and prompt_id if they are expected but missing

    return prompt_data, gt_data, prompt_id
