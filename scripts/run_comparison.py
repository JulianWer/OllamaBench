from itertools import combinations
import logging
import os
import random
from typing import Dict, Any, Optional

from utils.get_data import _load_prompt_dataset, extract_information_from_dataset_prompt

# Stellen Sie sicher, dass diese Importe für Ihre Projektstruktur korrekt sind
# Gegebenenfalls müssen Sie die Pfade anpassen, z.B. wenn evaluation und models
# Pakete auf der gleichen Ebene wie Ihr Hauptskript sind.
# from evaluation.ranking import update_elo, ModelRatingsType
# from models.judge_llm import JudgeLLM
# from utils.get_data import get_random_prompt_and_ground_truth
# from utils.file_operations import save_elo_results, load_json_file

# Platzhalter für die tatsächlichen Importe, falls sie nicht direkt gefunden werden
# Dies dient dazu, dass das Snippet an sich lauffähig ist, auch wenn die Module fehlen.
# In einer realen Umgebung sollten diese durch die korrekten Importe ersetzt werden.
try:
    from evaluation.ranking import update_elo, ModelRatingsType
    from models.judge_llm import JudgeLLM
    from utils.get_data import get_random_prompt_and_ground_truth
    from utils.file_operations import save_elo_results, load_json_file
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import all necessary modules. Using placeholders. Ensure your project structure and PYTHONPATH are correct.")
    # Definieren Sie Platzhalter-Typen/Funktionen, um Syntaxfehler zu vermeiden
    ModelRatingsType = Dict[str, Any]
    def update_elo(*args, **kwargs): return None, None
    class JudgeLLM:
        def __init__(self, *args, **kwargs): pass
        def evaluate(self, *args, **kwargs): return None
    def get_random_prompt_and_ground_truth(*args, **kwargs): return None, None, None
    def save_elo_results(*args, **kwargs): return False
    def load_json_file(*args, **kwargs): return None


logger = logging.getLogger(__name__)

def _extract_prompt_text(prompt_data: Any) -> Optional[str]:
    """
    Extracts a single string prompt from potentially complex prompt data.
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
            logger.debug("Extracted prompt text from the last 'user' turn.")
            return user_turns_content[-1]
        elif all_turns_content:
            logger.warning(f"No 'user' role found in prompt turns. Falling back to concatenating content from all {len(all_turns_content)} turns/strings.")
            return "\n".join(all_turns_content)
        else:
            logger.warning(f"Prompt data is a list but contains no extractable string content: {prompt_data}")
            return None
    else:
        logger.warning(f"Unsupported prompt data type: {type(prompt_data)}. Cannot extract text.")
        return None


def run_judge_comparison(config: Dict[str, Any], category: str) -> Optional[ModelRatingsType]:
    """
    Executes comparison cycles for a given category.
    Selects random prompts and then random pairs of existing model answers for those prompts.
    """
    updated_ratings_snapshot: Optional[ModelRatingsType] = None
    
    # Anzahl der zu vergleichenden Prompts pro Kategorie aus der Konfiguration holen
    num_prompt_comparisons_per_category = config.get("comparison", {}).get("comparisons_per_run", 1)
    # Anzahl der zu vergleichenden Antwortpaare pro Prompt aus der Konfiguration holen
    num_pair_comparisons_per_prompt = config.get("comparison", {}).get("comparisons_per_prompt", 30) # Standardwert 30

    paths_config = config.get("paths", {})
    base_output_dir = paths_config.get("output_dir", "output/model_responses") 
    
    k_factor = config.get("elo", {}).get("k_factor", 32)
    initial_rating = config.get("elo", {}).get("initial_rating", 1000.0)
    
    ollama_api_url = config.get("ollama", {}).get("api_base_url")
    judge_model_name = config.get("JUDGE_MODEL")
    judge_temp = config.get("generation_options_judge", {}).get("temperature", 0.0) # Standardwert 0.0
    judge_has_reasoning = config.get("HAS_REASONING", False) # Standardwert False
    judge_options = config.get("generation_options_judge", {})


    if not ollama_api_url:
        logger.error("Ollama API base URL ('ollama.api_base_url') not configured.")
        return None
    if not judge_model_name:
        logger.error("Judge model ('JUDGE_MODEL') not configured.")
        return None

    judge = JudgeLLM(
        api_url=ollama_api_url,
        model_name=judge_model_name,
        temperature=judge_temp,
        has_reasoning=judge_has_reasoning,
        options=judge_options
    )
    
    overall_run_success = True
    
    dataset = _load_prompt_dataset(config, category)
    
    # Äußere Schleife: Iteriert über die Anzahl der zu vergleichenden Prompts
    for prompt in dataset:
        cycle_success_flag_for_logging = False # Für die finale Log-Nachricht dieses Zyklus
        prompt_processing_details = {"prompt_id": "N/A", "status": "Skipped", "pairs_processed_attempts": 0, "pairs_succeeded": 0}
        
        logger.info(f"Starting comparison cycle {0 + 1}/{num_prompt_comparisons_per_category} for category '{category}'.")

        try:
            logger.info("Fetching a random prompt and its ground truth...")
            prompt_data_tuple = extract_information_from_dataset_prompt(dataset_item=prompt)

            if prompt_data_tuple is None:
                logger.error("Failed to retrieve a valid prompt/ground truth tuple. Skipping this cycle.")
                prompt_processing_details["status"] = "Failed (No Prompt)"
                overall_run_success = False
                continue 
            
            prompt_content, ground_truth, prompt_id = prompt_data_tuple
            prompt_processing_details["prompt_id"] = prompt_id
            logger.info(f"Retrieved prompt ID: {prompt_id}")

            # Dateipfad für die Antworten zu diesem Prompt erstellen
            category_specific_output_dir = os.path.join(base_output_dir, str(category)) # Sicherstellen, dass category ein String ist
            # Erstellt einen sicheren Dateinamen für den Prompt
            safe_prompt_id_filename = "".join(c if c.isalnum() or c in ['_','-'] else '_' for c in str(prompt_id)) + ".json"
            output_file_path = os.path.join(category_specific_output_dir, safe_prompt_id_filename)

            logger.info(f"Loading existing answers for prompt ID {prompt_id} from: {output_file_path}")
            existing_answers = load_json_file(output_file_path) 

            if not isinstance(existing_answers, list) or len(existing_answers) < 2:
                logger.warning(f"Not enough existing answers (found {len(existing_answers) if isinstance(existing_answers, list) else 0}, need at least 2) for prompt ID {prompt_id}. Skipping comparisons for this prompt.")
                prompt_processing_details["status"] = "Skipped (Not enough answers)"
                continue

            indices = list(range(len(existing_answers)))
            # Erstellt eine Liste aller möglichen einzigartigen Paare von Indizes
            pair_indices_iterator = list(combinations(indices, 2)) 
            
            if not pair_indices_iterator:
                logger.warning(f"No model pairs could be formed from the {len(existing_answers)} answers for prompt ID {prompt_id}. Skipping.")
                prompt_processing_details["status"] = "Skipped (No pairs formed)"
                continue

            logger.info(f"Found {len(existing_answers)} answers, forming {len(pair_indices_iterator)} possible unique pairs for prompt ID {prompt_id}.")
            
            num_successful_pairs_for_prompt = 0
            # Innere Schleife: Wählt zufällig Paare aus und bewertet sie
            # Versucht, 'num_pair_comparisons_per_prompt' Vergleiche durchzuführen,
            # aber nicht mehr als tatsächlich verfügbare Paare.
            actual_comparisons_to_run = min(num_pair_comparisons_per_prompt, len(pair_indices_iterator))
            prompt_processing_details["pairs_processed_attempts"] = actual_comparisons_to_run

            logger.info(f"Attempting to process up to {actual_comparisons_to_run} random pairs for this prompt.")

            for pair_selection_attempt in range(actual_comparisons_to_run):
                if not pair_indices_iterator: # Keine Paare mehr übrig
                    logger.info("No more unique pairs to compare for this prompt.")
                    break
                
                tuple_index = random.randrange(len(pair_indices_iterator))
                index1, index2 = pair_indices_iterator.pop(tuple_index) 

                selected_model_response_object_1 = existing_answers[index1]
                selected_model_response_object_2 = existing_answers[index2]

                model_a = selected_model_response_object_1.get("model")
                model_b = selected_model_response_object_2.get("model")
                response_a = selected_model_response_object_1.get("answer")
                response_b = selected_model_response_object_2.get("answer")
                
                if not all([model_a, model_b, response_a is not None, response_b is not None]):
                    logger.warning(f"Skipping pair for prompt {prompt_id} due to missing model name or answer: "
                                   f"M1({model_a}, ans_exists={response_a is not None}), M2({model_b}, ans_exists={response_b is not None})")
                    continue

                if model_a == model_b: 
                    logger.info(f"Skipping pair for prompt {prompt_id}: model_a and model_b are the same ('{model_a}').")
                    continue
                
                # Prompt-Text für den Judge extrahieren
                prompt_text_for_judge = _extract_prompt_text(prompt_content)
                if prompt_text_for_judge is None:
                    logger.error(f"Could not extract usable prompt text from data for prompt ID {prompt_id}: {prompt_content}")
                    prompt_processing_details["status"] = "Failed (Prompt Extraction Error)"
                    overall_run_success = False
                    # Breche die innere Schleife für diesen Prompt ab, da der Prompt-Text fehlt
                    break 

                logger.info(f"Comparing models for prompt ID {prompt_id}: '{model_a}' vs '{model_b}'. Attempt {pair_selection_attempt + 1}/{actual_comparisons_to_run}.")

                # Antworten bewerten (mit Positionsbias-Berücksichtigung)
                verdict_1 = judge.evaluate(
                    user_prompt=prompt_text_for_judge, 
                    response_a=response_a,
                    response_b=response_b,
                    ground_truth=ground_truth
                )
                verdict_2 = judge.evaluate(
                    user_prompt=prompt_text_for_judge, 
                    response_a=response_b, # Antworten getauscht
                    response_b=response_a,
                    ground_truth=ground_truth
                )
                
                # Score basierend auf den beiden Urteilen bestimmen
                # verdict_1/2 gibt 1.0 für A (erste Antwort), 0.0 für B (zweite Antwort), 0.5 für Tie
                final_score_model_a = 0.5 # Standard ist unentschieden
                if verdict_1 is None or verdict_2 is None:
                    logger.warning(f"Judging failed for one or both positions for prompt {prompt_id}, pair ({model_a} vs {model_b}). Skipping ELO update for this pair.")
                    continue # Nächstes Paar versuchen

                # Logik zur Konsolidierung der beiden Urteile
                # Wenn beide Urteile übereinstimmen und nicht unentschieden sind:
                if verdict_1 == verdict_2 and verdict_1 != 0.5:
                    final_score_model_a = verdict_1 # A gewinnt (1.0), B gewinnt (0.0)
                # Wenn Urteile widersprüchlich sind (A vs B und B vs A), dann unentschieden
                elif (verdict_1 == 1.0 and verdict_2 == 0.0) or \
                     (verdict_1 == 0.0 and verdict_2 == 1.0):
                    logger.info(f"Contradictory verdicts for prompt {prompt_id}, pair ({model_a} vs {model_b}). Treating as a tie.")
                    final_score_model_a = 0.5
                # Wenn ein Urteil unentschieden ist, das andere aber nicht, nimm das nicht-unentschiedene Urteil
                # (Hier muss man aufpassen, dass verdict_2 auf die ursprüngliche Reihenfolge A vs B zurückgerechnet wird)
                elif verdict_1 == 0.5 and verdict_2 != 0.5: # Erstes Urteil unentschieden
                    final_score_model_a = 1.0 - verdict_2 # Wenn verdict_2 = 1 (B gewinnt in getauschter Runde), dann A verliert (0.0)
                                                        # Wenn verdict_2 = 0 (A gewinnt in getauschter Runde), dann A gewinnt (1.0)
                elif verdict_2 == 0.5 and verdict_1 != 0.5: # Zweites Urteil unentschieden
                    final_score_model_a = verdict_1
                # Wenn beide unentschieden sind, bleibt es 0.5 (bereits Standard)

                logger.info(f"Judgement for prompt {prompt_id}, pair ({model_a} vs {model_b}): Verdict1 (A vs B)={verdict_1}, Verdict2 (B vs A)={verdict_2}. Final score for A = {final_score_model_a}")


                # ELO-Bewertungen aktualisieren
                logger.debug(f"Loading current ratings before ELO update for category '{category}'...")
                current_results_data = load_json_file(config["paths"]["results_file"], config["paths"]["lock_file"])
                ratings_obj_for_update = {}
                if isinstance(current_results_data, dict) and isinstance(current_results_data.get("models"), dict):
                    ratings_obj_for_update = current_results_data.get("models", {})
                else:
                    logger.warning("Could not load valid 'models' data from results file. ELO calculation will use initial ratings for missing models/categories.")

                new_rating_a, new_rating_b = update_elo(
                    model_ratings=ratings_obj_for_update, # Übergibt das gesamte Rating-Objekt
                    model_a=model_a,
                    model_b=model_b,
                    category=category, 
                    score_a=final_score_model_a, # Der Score für Model A (1.0 für Sieg A, 0.0 für Sieg B, 0.5 für Unentschieden)
                    k_factor=k_factor,
                    initial_rating=initial_rating
                )

                if new_rating_a is None or new_rating_b is None: 
                    logger.error(f"ELO update function indicated an error for prompt {prompt_id}, pair ({model_a}, {model_b}). Skipping save for this pair.")
                    overall_run_success = False
                    continue 

                # ratings_obj_for_update wurde von update_elo direkt modifiziert
                updated_ratings_snapshot = ratings_obj_for_update 
                logger.info(f"ELO ratings updated in memory for category '{category}': {model_a} -> {new_rating_a:.1f}, {model_b} -> {new_rating_b:.1f}")

                # Ergebnisse speichern
                if not save_elo_results(updated_ratings_snapshot, config): # Speichert das gesamte aktualisierte Objekt
                    logger.error("Failed to save updated ELO results to file. Ratings may be inconsistent.")
                    overall_run_success = False
                else:
                    logger.debug("ELO results saved successfully after pair comparison.")
                    num_successful_pairs_for_prompt += 1
            
            # Status für diesen Prompt aktualisieren
            prompt_processing_details["pairs_succeeded"] = num_successful_pairs_for_prompt
            if prompt_processing_details["pairs_processed_attempts"] > 0:
                if num_successful_pairs_for_prompt == prompt_processing_details["pairs_processed_attempts"]:
                    logger.info(f"Successfully processed all {prompt_processing_details['pairs_processed_attempts']} attempted model pairs for prompt {prompt_id}.")
                    prompt_processing_details["status"] = "Success"
                    cycle_success_flag_for_logging = True
                elif num_successful_pairs_for_prompt > 0:
                    logger.warning(f"Processed {num_successful_pairs_for_prompt}/{prompt_processing_details['pairs_processed_attempts']} model pairs for prompt {prompt_id}. Some pairs may have failed or were skipped.")
                    prompt_processing_details["status"] = f"Partial Success ({num_successful_pairs_for_prompt}/{prompt_processing_details['pairs_processed_attempts']})"
                    cycle_success_flag_for_logging = True # Auch Teilerfolg ist ein "Finished"
                else:
                    logger.error(f"Failed to process any model pairs for prompt {prompt_id} out of {prompt_processing_details['pairs_processed_attempts']} attempts.")
                    prompt_processing_details["status"] = "Failed (All pairs failed)"
                    overall_run_success = False
            else: 
                logger.info(f"No pair processing was attempted for prompt {prompt_id} (e.g. not enough answers or pairs).")
                # Status bleibt wie zuvor gesetzt (z.B. "Skipped (Not enough answers)")


        except Exception as e:
            logger.exception(f"An unexpected error terminated the comparison cycle for prompt ID {prompt_processing_details.get('prompt_id', 'N/A')}: {e}")
            prompt_processing_details["status"] = "Failed (Exception)"
            overall_run_success = False
            # Hier nicht `return None`, damit die äußere Schleife ggf. weiterläuft
        finally:
            log_status = "Finished" if cycle_success_flag_for_logging or prompt_processing_details["status"].startswith("Skipped") else "Failed"
            logger.info(f"Details for prompt ID '{prompt_processing_details['prompt_id']}': Status: {prompt_processing_details['status']}, Attempted Pairs: {prompt_processing_details['pairs_processed_attempts']}, Succeeded Pairs: {prompt_processing_details['pairs_succeeded']}.")
            logger.info("=" * 50)

    if not overall_run_success:
        logger.error(f"One or more errors occurred during the run_judge_comparison for category '{category}'. Check logs for details.")
        return None # Gibt None zurück, wenn es schwerwiegende Fehler gab
    else:
        logger.info(f"Successfully completed all {num_prompt_comparisons_per_category} planned comparison cycles for category '{category}'.")
        # Lade die finalen Ratings, um den aktuellsten Stand zurückzugeben
        final_results_data = load_json_file(config["paths"]["results_file"], config["paths"]["lock_file"])
        if isinstance(final_results_data, dict) and isinstance(final_results_data.get("models"), dict):
            return final_results_data.get("models")
        return updated_ratings_snapshot # Fallback auf den letzten Snapshot im Speicher

