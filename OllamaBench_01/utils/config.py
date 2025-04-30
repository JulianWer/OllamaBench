import yaml
import logging
from typing import Dict, Any

# Logging-Konfiguration kann zentral erfolgen, z.B. in main.py oder dashboard.py
logger = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = './config/config.yaml'


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Lädt die Konfiguration aus einer YAML-Datei.

    Args:
        config_path: Pfad zur Konfigurationsdatei.

    Returns:
        Ein Dictionary mit der Konfiguration.

    Raises:
        FileNotFoundError: Wenn die Konfigurationsdatei nicht gefunden wird.
        yaml.YAMLError: Wenn ein Fehler beim Parsen der YAML-Datei auftritt.
        ValueError: Wenn die Konfigurationsdatei leer oder ungültig ist.
        Exception: Bei unerwarteten Fehlern.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict): # Prüfen, ob es ein Dictionary ist
                logger.error(f"Konfigurationsdatei '{config_path}' ist leer oder hat ein ungültiges Format.")
                raise ValueError(f"Konfigurationsdatei '{config_path}' ist leer oder ungültig.")
            logger.info(f"Konfiguration erfolgreich aus '{config_path}' geladen.")
            return config
    except FileNotFoundError:
        logger.error(f"Konfigurationsdatei nicht gefunden: '{config_path}'.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Fehler beim Parsen der YAML-Datei '{config_path}': {e}")
        raise
    except Exception as e:
        logger.error(f"Unerwarteter Fehler beim Laden der Konfiguration '{config_path}': {e}")
        raise
