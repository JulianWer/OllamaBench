import logging
import os
from typing import Any, Dict, Optional, Tuple, Union, List, Set, NamedTuple

import datasets
from datasets import load_dataset, Dataset, IterableDataset, DatasetDict, Features, Value

try:
    from .file_operations import save_dataset_to_json, load_json_file
except ImportError:
    from file_operations import save_dataset_to_json, load_json_file

logger = logging.getLogger(__name__)

# In-memory cache for full Hugging Face datasets to avoid re-downloading
_hf_dataset_cache: Dict[str, Dataset] = {}


class DatasetConfig(NamedTuple):
    """A simple, immutable container for all dataset-related configurations."""
    hf_dataset_name: Optional[str]
    data_files: Optional[Dict[str, str]]
    prompt_column: str
    ground_truth_column: Optional[str]
    category_column: Optional[str]
    prompt_id_column: Optional[str]
    num_save_entries: int
    category_base_dir: Optional[str]
    dataset_file_suffix: str
    dataset_save_lock_file: Optional[str]


def _parse_dataset_config(app_config: Dict[str, Any]) -> DatasetConfig:
    """Parses the main application config to create a dedicated DatasetConfig object."""
    dataset_cfg = app_config.get("dataset", {})
    columns_cfg = dataset_cfg.get("columns", {})
    paths_cfg = app_config.get("paths", {})

    return DatasetConfig(
        hf_dataset_name=dataset_cfg.get("name"),
        data_files=dataset_cfg.get("data_files"),
        prompt_column=columns_cfg.get("prompt"),
        ground_truth_column=columns_cfg.get("ground_truth"),
        category_column=columns_cfg.get("category"),
        prompt_id_column=columns_cfg.get("id"),
        num_save_entries=app_config.get("NUM_SAVE_DATASET_ENTRIES", 100),
        category_base_dir=paths_cfg.get("dataset_category_dir"),
        dataset_file_suffix=paths_cfg.get("dataset_file_suffix", "_prompts.json"),
        dataset_save_lock_file=paths_cfg.get("dataset_save_lock_file"),
    )


def _validate_columns(ds_columns: Set[str], config: DatasetConfig) -> bool:
    """Validates that essential columns from the config exist in the dataset."""
    required = [config.prompt_column, config.category_column]
    missing = [col for col in required if col and col not in ds_columns]
    if missing:
        logger.error(f"Required columns {missing} not found in dataset. Available: {list(ds_columns)}")
        return False
    return True


def _determine_features_from_data(data: List[Dict[str, Any]]) -> Optional[Features]:
    """Infers a Hugging Face Features schema from a list of dictionaries."""
    if not data:
        return None
    
    all_keys = {key for entry in data if isinstance(entry, dict) for key in entry.keys()}
    if not all_keys:
        return None

    feature_dict = {key: Value("string") for key in all_keys}
    if 'turns' in feature_dict:
        feature_dict['turns'] = datasets.Sequence(Value("string"))
        
    return Features(feature_dict)


def _load_from_local_json(config: DatasetConfig, category: str) -> Optional[Dataset]:
    """Loads a dataset for a specific category from a local JSON file."""
    if not config.category_base_dir:
        return None

    category_file_path = os.path.join(config.category_base_dir, f"{category.lower()}{config.dataset_file_suffix}")
    logger.info(f"Attempting to load local dataset for '{category}' from: '{category_file_path}'")

    if not os.path.isfile(category_file_path):
        logger.info(f"Local file not found: '{category_file_path}'.")
        return None

    data_list = load_json_file(category_file_path, lock_file_path=config.dataset_save_lock_file)
    if not isinstance(data_list, list) or not data_list:
        logger.warning(f"File '{category_file_path}' is empty or not a valid JSON list.")
        return None

    valid_entries = [entry for entry in data_list if isinstance(entry, dict)]
    if not valid_entries:
        return None

    try:
        features = _determine_features_from_data(valid_entries)
        dataset = Dataset.from_list(valid_entries, features=features)
        
        if config.prompt_column and config.prompt_column in dataset.column_names:
            logger.info(f"Successfully loaded and validated local dataset for '{category}'.")
            return dataset
        else:
            logger.error(f"Prompt column '{config.prompt_column}' not found in local file '{category_file_path}'.")
            return None
            
    except Exception as e:
        logger.exception(f"Failed to create Dataset object from '{category_file_path}': {e}")
        return None


def _save_dataset_by_category(full_ds: Dataset, config: DatasetConfig):
    """Filters a full dataset by category and saves each part to a separate JSON file."""
    if not all([config.category_column, config.category_base_dir, config.category_column in full_ds.column_names]):
        logger.warning("Category column or directory not configured, or column not in dataset. Skipping save by category.")
        return

    logger.info(f"Saving dataset samples by category '{config.category_column}' to '{config.category_base_dir}'...")
    os.makedirs(config.category_base_dir, exist_ok=True)
    
    try:
        for category_value in set(full_ds[config.category_column]):
            safe_category_name = "".join(c for c in str(category_value).lower() if c.isalnum() or c in ['_','-'])
            target_file_path = os.path.join(config.category_base_dir, f"{safe_category_name}{config.dataset_file_suffix}")

            category_ds = full_ds.filter(lambda ex: ex[config.category_column] == category_value)
            
            if len(category_ds) == 0:
                continue

            if config.num_save_entries > 0 and len(category_ds) > config.num_save_entries:
                dataset_to_save = category_ds.shuffle(seed=42).select(range(config.num_save_entries))
            else:
                dataset_to_save = category_ds
            
            save_dataset_to_json(dataset_to_save, target_file_path, config.dataset_save_lock_file)
            logger.info(f"Saved {len(dataset_to_save)} entries for category '{category_value}' to '{target_file_path}'.")

    except Exception as e:
        logger.exception(f"An error occurred while saving the dataset by category: {e}")


def load_prompt_dataset(app_config: Dict[str, Any], category: str) -> Optional[Dataset]:
    """
    Loads a dataset for a specific category using a tiered approach:
    1. Tries to load a pre-filtered local JSON file.
    2. If not found, loads the full dataset from Hugging Face Hub.
    3. Caches the full dataset and saves categorized samples locally for future runs.
    4. Filters the full dataset for the requested category.
    """
    config = _parse_dataset_config(app_config)
    if not config.prompt_column:
        logger.critical("The 'prompt' column is not defined in the configuration. Cannot load dataset.")
        return None

    local_ds = _load_from_local_json(config, category)
    if local_ds:
        return local_ds

    if not config.hf_dataset_name:
        logger.error("No local dataset found, and no Hugging Face dataset name is configured.")
        return None
        
    if config.hf_dataset_name in _hf_dataset_cache:
        logger.info(f"Using in-memory cached version of '{config.hf_dataset_name}'.")
        full_ds = _hf_dataset_cache[config.hf_dataset_name]
    else:
        logger.info(f"Loading full dataset '{config.hf_dataset_name}' from Hugging Face Hub.")
        try:
            load_params = {"path": config.hf_dataset_name, "trust_remote_code": True}
            if config.data_files:
                load_params["data_files"] = config.data_files
                
            loaded_object = load_dataset(**load_params)
            
            if isinstance(loaded_object, DatasetDict):
                full_ds = loaded_object.get("train") or loaded_object.get("test") or next(iter(loaded_object.values()), None)
            else:
                full_ds = loaded_object
            
            if not isinstance(full_ds, (Dataset, IterableDataset)):
                logger.error(f"Could not extract a valid Dataset object from loaded data (type: {type(full_ds)}).")
                return None
            
            if not _validate_columns(set(full_ds.column_names), config):
                return None
            
            _hf_dataset_cache[config.hf_dataset_name] = full_ds
            if isinstance(full_ds, Dataset):
                _save_dataset_by_category(full_ds, config)

        except Exception as e:
            logger.exception(f"Failed to load dataset from Hugging Face Hub: {e}")
            return None

    if config.category_column and config.category_column in full_ds.column_names:
        logger.info(f"Filtering dataset for category: '{category}'.")
        if isinstance(full_ds, IterableDataset):
             logger.warning("Filtering an IterableDataset is not directly supported.")
             return full_ds 
        
        category_ds = full_ds.filter(lambda ex: ex[config.category_column] == category)
        if len(category_ds) == 0:
            logger.warning(f"No entries found for category '{category}' in the dataset.")
            return None
        return category_ds
    else:
        logger.warning("Category column not configured or found. Returning full dataset.")
        return full_ds

def extract_prompt_info_from_dataset_item(
    app_config: Dict[str, Any],
    dataset_item: Dict[str, Any]
) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Extracts prompt, ground truth, and ID from a single dataset item based on config."""
    config = _parse_dataset_config(app_config)
    
    prompt_data = dataset_item.get(config.prompt_column)
    gt_data = dataset_item.get(config.ground_truth_column) if config.ground_truth_column else None
    prompt_id = dataset_item.get(config.prompt_id_column) if config.prompt_id_column else None

    if prompt_data is None:
        logger.warning(f"Prompt data not found in item using column name '{config.prompt_column}'.")

    return prompt_data, gt_data, prompt_id

def prepare_prompt_for_llm(raw_prompt_content: Any) -> Union[str, List[Dict[str, str]], None]:
    """
    Prepares the raw prompt content into the format expected by the LLM (string or message list).
    """
    if isinstance(raw_prompt_content, str):
        return raw_prompt_content
    elif isinstance(raw_prompt_content, list):
        if all(isinstance(item, dict) and "role" in item and "content" in item for item in raw_prompt_content):
            return raw_prompt_content
        elif all(isinstance(item, str) for item in raw_prompt_content):
            # Convert a list of strings into a multi-turn conversation format
            messages = []
            for i, text in enumerate(raw_prompt_content):
                role = "assistant" if i % 2 != 0 and i > 0 else "user"
                messages.append({"role": role, "content": text})
            return messages
    
    logger.warning(f"Unsupported prompt content type: {type(raw_prompt_content)}. Skipping.")
    return None

def extract_prompt_text(prompt_data: Any) -> Optional[str]:
    """
    Extracts a single string prompt from potentially complex prompt data, suitable for a judge.
    Prioritizes the last user turn, falls back to concatenating all turns' content.
    """
    if isinstance(prompt_data, str):
        return prompt_data
    elif isinstance(prompt_data, list):
        user_turns_content = []
        all_turns_content = []
        for turn in prompt_data:
            if isinstance(turn, dict) and isinstance(turn.get("content"), str):
                content = turn["content"]
                all_turns_content.append(content)
                if turn.get("role") == "user":
                    user_turns_content.append(content)
            elif isinstance(turn, str):
                 all_turns_content.append(turn)
        
        if user_turns_content:
            logger.debug("Extracted prompt text from the last 'user' turn for the judge.")
            return user_turns_content[-1]
        elif all_turns_content:
            logger.warning(f"No 'user' role found in prompt turns. Concatenating all {len(all_turns_content)} turns.")
            return "\n".join(all_turns_content)
        
    logger.warning(f"Unsupported or empty prompt data type: {type(prompt_data)}. Cannot extract text.")
    return None