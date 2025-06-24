import yaml
import logging
from typing import Dict, Any, List, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = './config/config.yaml'

class ConfigService:
    """A service class to provide centralized and cached access to the application configuration."""
    
    _config: Dict[str, Any] = None
    _instance = None

    def __new__(cls, config_path: str = DEFAULT_CONFIG_PATH):
        # Singleton pattern to ensure only one instance of the config is loaded.
        if cls._instance is None:
            cls._instance = super(ConfigService, cls).__new__(cls)
            cls._instance.load_config(config_path)
        return cls._instance

    def load_config(self, config_path: str):
        """
        Loads the configuration from a YAML file.

        Args:
            config_path: Path to the configuration file.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            yaml.YAMLError: If an error occurs while parsing the YAML file.
            ValueError: If the configuration file is empty or invalid.
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if not isinstance(config, dict):
                    logger.error(f"Config file '{config_path}' is empty or has an invalid format.")
                    raise ValueError(f"Config file '{config_path}' is empty or invalid.")
                self._config = config
                logger.info(f"Configuration successfully loaded from '{config_path}'.")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: '{config_path}'.")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing the YAML file '{config_path}': {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration '{config_path}': {e}")
            raise

    @lru_cache(maxsize=None) # Cache results of path lookups
    def get_path(self, key: str) -> Optional[str]:
        """Gets a path from the 'paths' section of the config."""
        return self._config.get("paths", {}).get(key)

    @property
    @lru_cache(maxsize=1)
    def all_categories(self) -> List[str]:
        """Returns the list of all categories."""
        return self._config.get("categories", [])

    @property
    @lru_cache(maxsize=1)
    def comparison_llms_config(self) -> Dict[str, Any]:
        """Returns the configuration for comparison LLMs."""
        return self._config.get("comparison_llms", {})

    @property
    @lru_cache(maxsize=1)
    def judge_llm_config(self) -> Dict[str, Any]:
        """Returns the configuration for the judge LLM."""
        return self._config.get("judge_llm", {})

    @property
    @lru_cache(maxsize=1)
    def llm_runtime_config(self) -> Dict[str, Any]:
        """Returns the LLM runtime API configuration."""
        return self._config.get("LLM_runtime", {})

    @property
    @lru_cache(maxsize=1)
    def dataset_config(self) -> Dict[str, Any]:
        """Returns the dataset configuration."""
        return self._config.get("dataset", {})

    @property
    @lru_cache(maxsize=1)
    def mELO_config(self) -> Dict[str, Any]:
        """Returns the mELO configuration."""
        return self._config.get("mELO", {})
        
    def get_full_config(self) -> Dict[str, Any]:
        """Returns the entire configuration dictionary."""
        return self._config

# For compatibility with old `load_config` calls, though using the service is preferred.
def load_config(config_path: str) -> Dict[str, Any]:
    """Loads config and returns it as a dictionary."""
    return ConfigService(config_path).get_full_config()