import json
import logging
from typing import Any, Dict, Optional
from datasets import load_dataset, Dataset, IterableDataset

logger = logging.getLogger(__name__)

def export_prompts_to_json(config: Dict[str, Any], output_json_path: str, max_entries: int = 100) -> bool:
    dataset_name = config.get("PROMPT_DATASET")
    prompt_column = config.get("PROMPT_DATASET_COLUMN")

    if not dataset_name or not prompt_column:
        logger.error("PROMPT_DATASET oder PROMPT_DATASET_COLUMN nicht in der Konfiguration angegeben.")
        return False

    logger.info(f"Lade Prompt-Datensatz '{dataset_name}'...")
    try:
        ds = load_dataset(dataset_name, split="train") 

        if prompt_column not in ds.column_names:
            logger.error(f"Prompt-Spalte '{prompt_column}' nicht im Datensatz '{dataset_name}' gefunden. Verfügbare Spalten: {ds.column_names}")
            return False

        num_entries_to_select = min(max_entries, len(ds))
        logger.info(f"Wähle die ersten {num_entries_to_select} Einträge aus der Spalte '{prompt_column}'.")

        selected_data = ds.select(range(num_entries_to_select))

        prompts = selected_data[prompt_column]

        logger.info(f"Speichere {len(prompts)} Prompts in '{output_json_path}'...")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=4)

        logger.info(f"Prompts erfolgreich in '{output_json_path}' gespeichert.")
        return True

    except FileNotFoundError:
        logger.error(f"Datensatz '{dataset_name}' nicht gefunden. Stelle sicher, dass der Name korrekt ist und du Zugriff darauf hast.")
        return False
    except Exception as e:
        logger.error(f"Fehler beim Verarbeiten des Datensatzes '{dataset_name}' oder beim Schreiben der JSON-Datei: {e}", exc_info=True)
        return False   
        