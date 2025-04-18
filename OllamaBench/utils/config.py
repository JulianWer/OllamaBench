import yaml
import logging
from typing import Dict, Any

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        A dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the config file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
        ValueError: If the configuration file is empty or invalid.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if config is None:
                logger.error(f"Configuration file '{config_path}' is empty or invalid.")
                raise ValueError(f"Configuration file '{config_path}' is empty or invalid.")
            logger.info(f"Configuration loaded successfully from '{config_path}'.")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at '{config_path}'.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file '{config_path}': {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config '{config_path}': {e}")
        raise

