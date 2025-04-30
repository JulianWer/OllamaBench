import logging
import random
import os
import glob
from typing import Any, Dict, Optional, Tuple, Union, List, Set
import datasets
from datasets import load_dataset, Dataset, IterableDataset, DatasetDict, Features, Value, Sequence # Added Sequence
from datetime import datetime
# Stelle sicher, dass die Hilfsfunktionen korrekt importiert werden
from .file_operations import save_dataset_to_json, load_json_file
# Kein Import von load_config hier nötig, da es übergeben wird

logger = logging.getLogger(__name__)

# Globaler Cache
PROMPT_DATASET_CACHE: Optional[Union[Dataset, IterableDataset, DatasetDict]] = None
PROMPT_DATASET_NAME: Optional[str] = None
PROMPT_COLUMN: Optional[str] = None
GROUND_TRUTH_COLUMN: Optional[str] = None
CATEGORY_COLUMN: Optional[str] = None

# --- Hilfsfunktionen ---

def _get_dataset_config(config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], int, str, Optional[str], Optional[str]]:
    """Extrahiert Datensatz-bezogene Konfigurationen."""
    dataset_name = config.get("PROMPT_DATASET")
    prompt_column = config.get("PROMPT_DATASET_COLUMN")
    ground_truth_column = config.get("GROUND_TRUTH_DATASET_COLUMN")
    category_column_name = config.get("PROMPT_DATASET_CATEGORY_COLUMN")
    num_of_entries = config.get("NUM_SAVE_DATASET_ENTRIES", 100)
    # Korrektur: Pfad-Suffix aus paths-Sektion holen
    paths_config = config.get("paths", {})
    dataset_path_suffix = paths_config.get("dataset_file_suffix", "_prompts.json") # Fallback direkt hier
    category_base_dir = paths_config.get("dataset_category_dir")
    dataset_save_lock_path = paths_config.get("dataset_save_lock_file")
    return (dataset_name, prompt_column, ground_truth_column, category_column_name,
            num_of_entries, dataset_path_suffix, category_base_dir, dataset_save_lock_path)

def _validate_columns(ds_columns: Set[str], required_cols: List[str]) -> bool:
    """Prüft, ob die benötigten Spalten im Set der Spaltennamen vorhanden sind."""
    actual_required = [col for col in required_cols if col is not None]
    missing_columns = [col for col in actual_required if col not in ds_columns]
    if missing_columns:
        logger.error(f"Benötigte Spalten {missing_columns} nicht im Datensatz gefunden. Verfügbare Spalten: {list(ds_columns)}")
        return False
    logger.debug(f"Benötigte Spalten ({actual_required}) im Dataset gefunden.")
    return True

def _clean_data_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Bereinigt einen einzelnen Datensatz-Eintrag für konsistente Typen (vereinfacht für Debugging)."""
    # ACHTUNG: Diese Bereinigung ist sehr einfach gehalten und behandelt 'turns' nicht optimal!
    cleaned_entry = entry.copy()
    for key, value in cleaned_entry.items():
        if key == "year":
            try: cleaned_entry[key] = int(value) if value is not None and value != '' else None
            except (ValueError, TypeError): cleaned_entry[key] = None
        elif key == "hardness":
            try: cleaned_entry[key] = float(value) if value is not None and value != '' else None
            except (ValueError, TypeError): cleaned_entry[key] = None
        elif key == "turns":
            # Konvertiert jedes Element in der turns-Liste zu einem String!
            # Das zerstört die [{"role": ..., "content": ...}] Struktur.
            if not isinstance(value, list):
                 cleaned_entry[key] = [str(value)] if value is not None else []
            else:
                 logger.warning("Cleaning 'turns' by converting all items to string. Original structure might be lost.")
                 cleaned_entry[key] = [str(item) if item is not None else "" for item in value]
        # Fallback für andere Typen (optional, kann zu Problemen führen)
        # elif value is not None and not isinstance(value, (str, int, float, bool, list, dict)):
        #     logger.debug(f"Converting complex type {type(value)} to string for key '{key}'")
        #     cleaned_entry[key] = str(value)

    return cleaned_entry

def _determine_features_simplified(data: List[Dict[str, Any]]) -> Optional[Features]:
    """Bestimmt ein vereinfachtes Schema (meist Strings) für Debugging."""
    if not data: return None
    all_keys = set()
    for entry in data:
        if isinstance(entry, dict): all_keys.update(entry.keys())
    if not all_keys: return None

    feature_dict = {}
    for key in all_keys:
        # Annahme: turns ist eine Sequenz von Strings (wegen _clean_data_entry)
        if key == "turns":
            feature_type = datasets.Sequence(Value("string"))
        elif key == "year": feature_type = Value("int64")
        elif key == "hardness": feature_type = Value("float64")
        else: feature_type = Value("string") # Default zu String
        feature_dict[key] = feature_type

    logger.debug(f"Verwende vereinfachtes Schema für lokales Laden: {feature_dict}")
    return Features(feature_dict)

# --- Laden aus lokalen JSON-Dateien ---
def _load_from_local_json(category_base_dir: str, dataset_save_lock_path: Optional[str], required_cols_for_op: List[str], category) -> Optional[Dataset]:
    """Lädt und bereinigt (vereinfacht) Daten aus lokalen JSON-Dateien."""
    if not category_base_dir or not os.path.isdir(category_base_dir):
         logger.warning(f"Lokales Kategorie-Verzeichnis '{category_base_dir}' nicht gefunden oder kein Verzeichnis.")
         return None

    logger.info(f"Beginne Ladeversuch aus lokalen JSON-Dateien in '{category_base_dir}'...")

    # Annahme: Suffix ist direkt '.json' - sollte aus config kommen!
    json_files = glob.glob(os.path.join(category_base_dir, f"*{category}_prompts.json"))
    logger.info(f"{len(json_files)} JSON-Dateien gefunden: {[os.path.basename(f) for f in json_files]}")
    if not json_files: return None

    all_data_from_json: List[Dict] = []
    load_errors = False
    processed_files = 0
    for file_path in json_files:
        basename = os.path.basename(file_path)
        # Exclude results.json and lock file
        if (dataset_save_lock_path and os.path.abspath(file_path) == os.path.abspath(dataset_save_lock_path)) or \
           basename == "results.json": # Simple check for results.json
            logger.debug(f"Überspringe Datei: {basename}")
            continue
        if not os.path.isfile(file_path): continue

        logger.debug(f"Versuche Datei zu laden: {file_path}")
        try:
            # Verwende die locking Funktion zum Laden
            loaded_data = load_json_file(file_path, lock_file_path=dataset_save_lock_path) # Use correct lock file

            if isinstance(loaded_data, list):
                if loaded_data:
                    logger.debug(f"'{basename}' enthält {len(loaded_data)} Einträge. Bereinige (vereinfacht)...")
                    # Wende die einfache Bereinigung an
                    cleaned_data = [_clean_data_entry(entry) for entry in loaded_data if isinstance(entry, dict)]
                    # Filtere Einträge, bei denen die Bereinigung evtl. None zurückgibt (sollte nicht passieren)
                    valid_cleaned_data = [d for d in cleaned_data if d is not None]
                    if len(valid_cleaned_data) != len(loaded_data):
                         logger.warning(f"Einige Einträge in '{basename}' wurden beim Bereinigen entfernt.")

                    all_data_from_json.extend(valid_cleaned_data)
                    processed_files += 1
                    logger.debug(f"Daten aus '{basename}' bereinigt und hinzugefügt ({len(valid_cleaned_data)} Einträge).")
                else:
                     logger.warning(f"Datei '{basename}' ist leer.")
            elif loaded_data is None:
                logger.warning(f"Laden von '{basename}' gab None zurück (möglicherweise Fehler oder leer).")
                load_errors = True
            else:
                logger.error(f"Unerwarteter Datentyp aus '{basename}': {type(loaded_data)}. Überspringe.")
                load_errors = True
        except Exception as e:
             logger.exception(f"Schwerer Fehler beim Verarbeiten der Datei '{basename}': {e}")
             load_errors = True

    if not all_data_from_json:
        logger.error("Konnte keine gültigen Daten aus lokalen JSON-Dateien extrahieren.")
        return None
    if load_errors:
        logger.warning("Es gab Fehler beim Laden/Verarbeiten einiger lokaler JSON-Dateien.")

    logger.info(f"{len(all_data_from_json)} Einträge aus {processed_files} Dateien gesammelt. Erstelle Dataset...")

    try:
        # Verwende vereinfachtes Schema basierend auf bereinigten Daten
        features = _determine_features_simplified(all_data_from_json)
        if features is None:
            logger.error("Konnte kein vereinfachtes Schema aus den geladenen JSON-Daten bestimmen.")
            return None

        logger.info(f"Versuche Dataset.from_list mit Schema: {features}")
        ds_from_json = Dataset.from_list(all_data_from_json, features=features)
        logger.info(f"Dataset erfolgreich erstellt mit {len(ds_from_json)} Zeilen und Spalten: {ds_from_json.column_names}")

        if not _validate_columns(set(ds_from_json.column_names), required_cols_for_op):
            logger.error("Validierung der benötigten Spalten für das lokale Dataset fehlgeschlagen.")
            return None

        logger.info(f"Lokales Dataset erfolgreich erstellt und validiert.")
        return ds_from_json
    except Exception as e:
        logger.exception(f"Fehler beim Erstellen des Datasets aus lokalen JSON: {e}.")
        return None

# --- Speichern von kategorisierten Samples ---
def _save_categorized_samples(ds: Dataset, category_column_name: str, category_base_dir: str,
                              num_of_entries: int, dataset_path_suffix: str,
                              config: Dict[str, Any], # config wird benötigt
                              dataset_save_lock_path: Optional[str]):
    """Speichert ausgewählte Spalten von Samples des Datasets in JSON-Dateien."""
    # ACHTUNG: Diese Funktion speichert die Daten, wie sie im übergebenen Dataset sind.
    # Wenn das Dataset nicht bereinigt wurde, werden rohe Daten gespeichert.
    if not isinstance(ds, Dataset):
        logger.warning("Kategorisiertes Speichern wird nur für 'Dataset'-Objekte unterstützt.")
        return

    # Spalten zum Speichern definieren (alle Spalten des Datasets)
    columns_to_save = ds.column_names
    logger.info(f"Speichere folgende Spalten: {columns_to_save}")

    # Präfix aus Config holen (nicht mehr verwendet im Dateinamen in dieser Version)
    # default_category_prefix = config.get("DEFAULT_CATEGORY", "unknown")

    logger.info(f"Beginne mit dem Speichern von Dataset-Samples nach Kategorie '{category_column_name}' in '{category_base_dir}'...")
    try:
        os.makedirs(category_base_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Konnte Verzeichnis '{category_base_dir}' nicht erstellen: {e}. Überspringe Speichern.")
        return

    try:
        if category_column_name not in ds.column_names:
             logger.error(f"Kategorie-Spalte '{category_column_name}' nicht im Dataset gefunden.")
             return

        unique_categories = set(ds[category_column_name])
        logger.info(f"{len(unique_categories)} eindeutige Kategorien gefunden: {list(unique_categories)[:10]}...")

        for category in unique_categories:
            # Dateiname nur aus Kategorie und Suffix
            safe_category_name = "".join(c if c.isalnum() else "_" for c in str(category).lower())[:50]
            if not safe_category_name: safe_category_name = "unknown_category"
            base_filename = f"{safe_category_name}{dataset_path_suffix}" # Nur Kategorie + Suffix
            target_file_path = os.path.join(category_base_dir, base_filename)

            # Timestamp-Logik (optional, kann entfernt werden für einfaches Überschreiben)
            final_save_path = target_file_path
            if os.path.exists(target_file_path):
                 logger.warning(f"Datei '{target_file_path}' existiert bereits und wird überschrieben.")
                 # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                 # name_part, ext_part = os.path.splitext(base_filename)
                 # final_save_path = os.path.join(category_base_dir, f"{name_part}_{timestamp}{ext_part}")
                 # logger.info(f"Datei '{target_file_path}' existiert. Speichere in '{final_save_path}'.")
            # else:
            #     logger.info(f"Datei '{target_file_path}' existiert nicht. Wird neu erstellt.")

            logger.info(f"Filtere Daten für Kategorie '{category}' zum Speichern in '{final_save_path}'...")
            try:
                # Filtern basierend auf Rohdaten
                filtered_ds = ds.filter(lambda ex: ex[category_column_name] == category, num_proc=1)
                num_filtered = len(filtered_ds)
                logger.debug(f"{num_filtered} Einträge für Kategorie '{category}' gefunden.")

                if num_filtered > 0:
                    dataset_to_process = filtered_ds
                    if num_filtered > num_of_entries:
                        logger.info(f"Sample {num_of_entries} Einträge für Kategorie '{category}'.")
                        dataset_to_process = filtered_ds.shuffle(seed=42).select(range(num_of_entries))

                    # Wähle *alle* Spalten aus (da columns_to_save = ds.column_names)
                    dataset_to_save = dataset_to_process # Keine Spaltenauswahl nötig

                    logger.info(f"Speichere {len(dataset_to_save)} Einträge nach '{final_save_path}'...")
                    # save_dataset_to_json wird verwendet (konvertiert zu Liste)
                    if save_dataset_to_json(dataset_to_save, final_save_path, dataset_save_lock_path):
                        logger.info(f"Kategorie '{category}' erfolgreich in '{final_save_path}' gespeichert.")
                    else:
                        logger.error(f"Fehler beim Speichern der Kategorie '{category}' in '{final_save_path}'.")
                else:
                     logger.warning(f"Keine Einträge für Kategorie '{category}' nach dem Filtern gefunden.")
            except Exception as filter_err:
                logger.exception(f"Fehler beim Verarbeiten/Speichern für Kategorie '{category}': {filter_err}")

    except Exception as cat_err:
         logger.exception(f"Allgemeiner Fehler während des kategorisierten Speicherns: {cat_err}")


# --- Haupt-Ladefunktion ---
def _load_prompt_dataset(config: Dict[str, Any]) -> Optional[Union[Dataset, IterableDataset]]:
    """
    Lädt das Prompt-Datensatz (roh), priorisiert lokale JSON, dann Cache, dann Hugging Face.
    """
    global PROMPT_DATASET_CACHE, PROMPT_DATASET_NAME, PROMPT_COLUMN, GROUND_TRUTH_COLUMN, CATEGORY_COLUMN

    (dataset_name, prompt_column, ground_truth_column, category_column_name,
     num_of_entries, dataset_path_suffix, category_base_dir, dataset_save_lock_path) = _get_dataset_config(config)

    if not prompt_column: # Prompt column ist essentiell
        logger.critical("PROMPT_DATASET_COLUMN fehlt in der Konfiguration. Laden nicht möglich.")
        return None

    required_cols_for_op = [prompt_column]
    if ground_truth_column: required_cols_for_op.append(ground_truth_column)
    if category_column_name: required_cols_for_op.append(category_column_name)

    # --- 1. Versuch: Aus lokalen JSON laden (gibt bereinigtes Dataset zurück) ---
    if category_base_dir:
        logger.info("--- Versuch 1: Lade Dataset aus lokalen JSON-Dateien ---")
        category = config.get("CURRENT_CATEGORY")

        # Diese Funktion führt die vereinfachte Bereinigung durch
        ds_from_json = _load_from_local_json(category_base_dir, dataset_save_lock_path, required_cols_for_op,category)
        if ds_from_json:
            logger.info("Dataset erfolgreich aus lokalen Dateien geladen und wird verwendet.")
            # Update cache vars
            PROMPT_DATASET_CACHE = ds_from_json
            PROMPT_DATASET_NAME = f"local_json:{category_base_dir}"
            PROMPT_COLUMN = prompt_column
            GROUND_TRUTH_COLUMN = ground_truth_column
            CATEGORY_COLUMN = category_column_name if category_column_name in ds_from_json.column_names else None
            return ds_from_json
        else:
            logger.info("Kein gültiges Dataset aus lokalen Dateien erstellt.")

    # --- 2. Versuch: Aus In-Memory-Cache (nur für Hugging Face Dataset) ---
    # Cache wird nur verwendet, wenn lokale Dateien nicht geladen wurden.
    logger.info("--- Versuch 2: Prüfe Cache für Hugging Face Dataset ---")
    # Prüfe, ob der Cache für das *konfigurierte* HF Dataset gültig ist
    is_cache_valid = (
        PROMPT_DATASET_CACHE is not None and
        PROMPT_DATASET_NAME == dataset_name and # Name muss übereinstimmen
        PROMPT_COLUMN == prompt_column # Prüfe auch Spaltennamen im Cache
        # Weitere Prüfungen (GT, Cat Column) könnten hinzugefügt werden
    )
    if is_cache_valid:
        # Prüfe Spalten im Cache-Objekt selbst
        if _validate_columns(set(PROMPT_DATASET_CACHE.column_names), required_cols_for_op):
             logger.info(f"Verwende gecachten Datensatz '{dataset_name}' von Hugging Face.")
             return PROMPT_DATASET_CACHE
        else:
             logger.warning("Cache gefunden, aber Spaltenvalidierung fehlgeschlagen. Lade neu.")
             PROMPT_DATASET_CACHE = None # Cache invalidieren
    else:
        logger.info("Kein gültiger Cache für Hugging Face Dataset gefunden oder Cache invalidiert.")


    # --- 3. Versuch: Von Hugging Face laden (roh) ---
    if not dataset_name:
         logger.error("Kein lokales Dataset gefunden und PROMPT_DATASET (für Hugging Face) nicht konfiguriert.")
         return None

    logger.info(f"--- Versuch 3: Lade Prompt-Datensatz '{dataset_name}' von Hugging Face (roh) ---")
    try:
        # Lade das Dataset ohne Bereinigung hier
        loaded_object = load_dataset(dataset_name, trust_remote_code=True)
        ds: Optional[Union[Dataset, IterableDataset]] = None

        # Split-Auswahl
        if isinstance(loaded_object, DatasetDict):
            split_preference = ["test", "validation", "train"]
            found_split = None
            for split_name in split_preference:
                if split_name in loaded_object: ds = loaded_object[split_name]; found_split = split_name; break
            if not ds:
                 first_split_key = next(iter(loaded_object.keys()), None)
                 if first_split_key: ds = loaded_object[first_split_key]; logger.warning(f"Verwende ersten Split: '{first_split_key}'.")
                 else: logger.error(f"Kein Split in DatasetDict '{dataset_name}'."); return None
            logger.info(f"Verwende '{found_split or first_split_key}'-Split aus DatasetDict '{dataset_name}'.")
        elif isinstance(loaded_object, (Dataset, IterableDataset)):
             ds = loaded_object
        else: logger.error(f"load_dataset gab unerwarteten Typ: {type(loaded_object)}"); return None

        # Validierung der Spalten für das rohe HF Dataset
        if not _validate_columns(set(ds.column_names), required_cols_for_op):
             logger.error("Benötigte Spalten im Hugging Face Dataset nicht gefunden.")
             return None

        # Cache aktualisieren (mit rohem Dataset)
        PROMPT_DATASET_CACHE = ds
        PROMPT_DATASET_NAME = dataset_name
        PROMPT_COLUMN = prompt_column
        GROUND_TRUTH_COLUMN = ground_truth_column
        CATEGORY_COLUMN = category_column_name if category_column_name in ds.column_names else None
        logger.info(f"Roher Datensatz '{dataset_name}' erfolgreich von Hugging Face geladen und gecacht.")

        # Kategorisiert speichern (rohe Daten)
        can_save_categorized = (category_base_dir and category_column_name and
                                isinstance(ds, Dataset) and category_column_name in ds.column_names)
        if can_save_categorized:
             logger.info("Speichere Samples des rohen Hugging Face Datasets lokal nach Kategorien...")
             # Übergebe das rohe Dataset zum Speichern
             _save_categorized_samples(ds, category_column_name, category_base_dir,
                                       num_of_entries, dataset_path_suffix, config, dataset_save_lock_path)

        return ds # Gib das rohe Dataset zurück

    except Exception as e:
        logger.exception(f"Schwerwiegender Fehler beim Laden von '{dataset_name}' von HF: {e}")
        PROMPT_DATASET_CACHE = None; PROMPT_DATASET_NAME = None; PROMPT_COLUMN = None
        GROUND_TRUTH_COLUMN = None; CATEGORY_COLUMN = None
        return None

# --- Funktion zum Abrufen eines zufälligen Prompts ---
def get_random_prompt_and_ground_truth(config: Dict[str, Any]) -> Optional[Tuple[str, Optional[str]]]:
    """
    Holt einen zufälligen Prompt und optional Ground Truth aus dem Dataset.
    ACHTUNG: Gibt Daten zurück, wie sie geladen wurden (potenziell unbereinigt,
    insbesondere wenn von Hugging Face geladen). Die aufrufende Funktion muss
    die Datenstruktur prüfen und verarbeiten können!
    """
    dataset_source = _load_prompt_dataset(config) # Ruft die Haupt-Ladelogik auf
    prompt_column = PROMPT_COLUMN
    ground_truth_column = GROUND_TRUTH_COLUMN
    # category_column = CATEGORY_COLUMN # Kategorie wird hier nicht direkt benötigt
    current_dataset_name = PROMPT_DATASET_NAME

    if dataset_source is None or prompt_column is None:
        logger.error("Kein Dataset verfügbar oder Prompt-Spalte nicht definiert für Zufallsauswahl.")
        return None

    # Umgang mit DatasetDict (sollte durch _load_prompt_dataset eigentlich nicht mehr vorkommen)
    if isinstance(dataset_source, DatasetDict):
         logger.error("Unerwarteter DatasetDict in get_random_prompt. Versuche Split-Extraktion.")
         split_preference = ["test", "validation", "train"]
         dataset = None
         for split_name in split_preference:
              if split_name in dataset_source: dataset = dataset_source[split_name]; break
         if not dataset: dataset = next(iter(dataset_source.values()), None)
         if not dataset: logger.error("Konnte keinen Split aus DatasetDict extrahieren."); return None
    elif isinstance(dataset_source, (Dataset, IterableDataset)):
         dataset = dataset_source
    else: logger.error(f"Unerwarteter Datensatztyp: {type(dataset_source)}"); return None

    logger.debug(f"Wähle zufälligen Prompt aus Dataset '{current_dataset_name}' (Typ: {type(dataset)}).")

    try:
        selected_item_raw = None
        if isinstance(dataset, Dataset):
            if len(dataset) == 0: logger.error(f"Dataset '{current_dataset_name}' ist leer."); return None
            random_index = random.randint(0, len(dataset) - 1)
            selected_item_raw = dataset[random_index] # Rohes Item
            logger.debug(f"Zufälliger Eintrag (Index {random_index}) aus Dataset '{current_dataset_name}'.")
        elif isinstance(dataset, IterableDataset):
             logger.debug(f"Versuche Zufalls-Prompt aus IterableDataset '{current_dataset_name}'.")
             try:
                  shuffled_ds = dataset.shuffle(seed=random.randint(0, 10000), buffer_size=500)
                  selected_item_raw = next(iter(shuffled_ds), None) # Rohes Item
                  if selected_item_raw is None: logger.warning(f"IterableDataset '{current_dataset_name}' scheint leer/erschöpft."); return None
             except Exception as iterable_err: logger.error(f"Fehler bei IterableDataset '{current_dataset_name}': {iterable_err}"); return None
             logger.debug(f"Zufälliger Eintrag (gesampelt) aus IterableDataset.")

        if selected_item_raw is None: logger.error(f"Konnte keinen Eintrag aus '{current_dataset_name}' wählen."); return None

        # --- Rohe Daten extrahieren ---
        # Konvertiere zu dict, falls es LazyRow etc. ist
        if hasattr(selected_item_raw, 'items'):
             selected_item_dict = dict(selected_item_raw.items())
        elif isinstance(selected_item_raw, dict):
             selected_item_dict = selected_item_raw
        else:
             logger.error(f"Ausgewähltes Item ist kein Dictionary oder dict-like: {type(selected_item_raw)}")
             return None

        prompt_data_raw = selected_item_dict.get(prompt_column)
        gt_data_raw = selected_item_dict.get(ground_truth_column) if ground_truth_column else None

        # --- Primitive Prompt-Extraktion (ohne Bereinigung!) ---
        # Der Aufrufer muss prüfen, ob prompt_data_raw eine Liste oder String ist!
        prompt_for_return = None
        if isinstance(prompt_data_raw, list):
             # Versuche, den letzten User-Turn zu finden (Best Effort)
             user_turns = [turn["content"] for turn in prompt_data_raw if isinstance(turn, dict) and turn.get("role") == "user" and isinstance(turn.get("content"), str)]
             if user_turns:
                 prompt_for_return = user_turns[-1]
             else:
                 # Fallback: Nimm den ersten String in der Liste? Oder ganze Liste?
                 # Hier nehmen wir an, der Aufrufer will einen String.
                 first_string = next((str(item) for item in prompt_data_raw if isinstance(item, str)), None)
                 if first_string:
                      prompt_for_return = first_string
                 else: # Wenn Liste leer oder nur non-strings/non-user-dicts
                      logger.warning(f"Konnte keinen geeigneten User-Prompt-String aus der Liste extrahieren: {prompt_data_raw}")
                      # Rückgabe None, da kein Prompt extrahiert werden konnte
                      return None
        elif isinstance(prompt_data_raw, str):
            prompt_for_return = prompt_data_raw
        elif prompt_data_raw is not None: # Wenn es weder Liste noch String ist
             logger.warning(f"Prompt-Spalte hat unerwarteten Typ {type(prompt_data_raw)}. Konvertiere zu String.")
             prompt_for_return = str(prompt_data_raw)
        else: # Wenn prompt_data_raw None ist
             logger.error(f"Prompt-Spalte '{prompt_column}' fehlt oder ist None im ausgewählten Eintrag.")
             return None

        # --- Ground Truth Extraktion (einfach) ---
        ground_truth_for_return = str(gt_data_raw) if gt_data_raw is not None else None

        logger.debug(f"Zufälliger Prompt/GT ausgewählt (roh). Prompt: '{prompt_for_return[:50]}...', GT: {'Ja' if ground_truth_for_return else 'Nein'}")
        # Gib nur Prompt und GT zurück, wie von der Funktion erwartet
        return prompt_for_return, ground_truth_for_return

    except Exception as e:
         logger.exception(f"Fehler bei Zufallsauswahl aus '{current_dataset_name}': {e}")
         return None
