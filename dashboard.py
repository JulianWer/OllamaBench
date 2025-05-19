import logging
import threading
import time
import os
import json
import datetime as dt
from typing import Any, Dict, Optional, Iterator, List, Set
from flask import Flask, jsonify, render_template, render_template_string, request, Response, stream_with_context
from werkzeug.exceptions import NotFound
import filelock 
from scripts.generate_model_answers_for_category import generate_and_save_model_answers_for_category
from utils.file_operations import load_json_file 
from utils.config import load_config, DEFAULT_CONFIG_PATH
from scripts.run_comparison import run_judge_comparison

# --- Konfigurationsladung ---
CONFIG = None
try:
    CONFIG = load_config(DEFAULT_CONFIG_PATH)
    RESULTS_FILE_PATH = CONFIG["paths"]["results_file"]
    LOCK_FILE_PATH = CONFIG["paths"]["lock_file"] 
    DASHBOARD_HOST = CONFIG.get("dashboard", {}).get("host", "127.0.0.1")
    DASHBOARD_PORT = CONFIG.get("dashboard", {}).get("port", 5001)
    DASHBOARD_DEBUG = CONFIG.get("dashboard", {}).get("debug", False)
    COMPARISONS_PER_RUN = CONFIG.get("comparison", {}).get("comparisons_per_run", 1)
    DEFAULT_CATEGORY = CONFIG.get("CURRENT_CATEGORY")
    ALL_CATEGORIES = CONFIG.get("ALL_CATEGORIES")
    RUN_ALL_CATEGORIES_IS_ENABLED = CONFIG.get("RUN_ALL_CATEGORIES_IS_ENABLED")

except (FileNotFoundError, KeyError, ValueError, Exception) as e:
    logging.basicConfig(level=logging.ERROR)
    logging.critical(f"FATAL: Konnte Konfiguration nicht laden oder validieren von '{DEFAULT_CONFIG_PATH}': {e}", exc_info=True)
    exit(1)

# --- Logging Setup ---
log_level_str = CONFIG.get("logging", {}).get("level", "INFO").upper()
log_format = CONFIG.get("logging", {}).get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

log_handlers = [logging.StreamHandler()]

logging.basicConfig(level=getattr(logging, log_level_str, logging.INFO),
                    format=log_format,
                    handlers=log_handlers)

logger = logging.getLogger(__name__)
logger.info("Dashboard Konfiguration erfolgreich geladen.")
logger.info(f"Ergebnisdatei: {RESULTS_FILE_PATH}")
logger.info(f"Sperrdatei: {LOCK_FILE_PATH}")

# --- Flask App Initialisierung ---
app = Flask(__name__)

# --- Globaler Status (Threadsicherheit) ---
state_lock = threading.Lock()
evaluation_running = False
evaluation_thread: Optional[threading.Thread] = None
last_evaluation_status: Dict[str, Any] = {"running": False, "message": "Noch keine Evaluation gestartet.", "status": "idle"}

results_lock = filelock.FileLock(LOCK_FILE_PATH, timeout=10)

def get_results_data() -> Optional[Dict[str, Any]]:
    try:
        with results_lock:
            if not os.path.exists(RESULTS_FILE_PATH) or os.path.getsize(RESULTS_FILE_PATH) == 0:
                logger.warning(f"Ergebnisdatei '{RESULTS_FILE_PATH}' nicht gefunden oder leer.")
                return {"timestamp": dt.datetime.utcnow().isoformat(), "models": {}}
            with open(RESULTS_FILE_PATH, "r", encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and "timestamp" in data and "models" in data:
                 if isinstance(data.get("timestamp"), (int, float)):
                      data["timestamp"] = dt.datetime.fromtimestamp(data["timestamp"], tz=dt.timezone.utc).isoformat()
                 elif not isinstance(data.get("timestamp"), str): 
                      data["timestamp"] = dt.datetime.utcnow().isoformat()

                 if not isinstance(data.get("models"), dict):
                     logger.warning("Ungültige 'models'-Struktur in results.json. Setze auf leer.")
                     data["models"] = {}

                 return data
            else:
                logger.error(f"Ungültige Struktur in Ergebnisdatei: {RESULTS_FILE_PATH}. Inhalt: {str(data)[:200]}")
                return {"timestamp": dt.datetime.utcnow().isoformat(), "models": {}} 
    except filelock.Timeout:
        logger.error(f"Timeout beim Warten auf Sperre für Ergebnisdatei: '{LOCK_FILE_PATH}'")
        return None # Signalisiert einen Fehler
    except json.JSONDecodeError:
        logger.error(f"Fehler beim Dekodieren von JSON aus: '{RESULTS_FILE_PATH}'. Datei könnte korrupt sein.", exc_info=True)
        return {"timestamp": dt.datetime.utcnow().isoformat(), "models": {}} 
    except Exception as e:
        logger.error(f"Fehler beim Lesen der Ergebnisdatei '{RESULTS_FILE_PATH}': {e}", exc_info=True)
        return None 

# --- SSE Endpunkt ---
@app.route('/stream')
def stream():
    def event_stream():
        last_mod_time = None
        while True:
            try:
                current_mod_time = os.path.getmtime(RESULTS_FILE_PATH)
                if last_mod_time is None or current_mod_time > last_mod_time:
                    logger.info(f"Änderung in '{RESULTS_FILE_PATH}' erkannt. Sende Update.")
                    data = get_results_data()
                    if data: 
                        json_data = json.dumps(data)
                        yield f"data: {json_data}\n\n"
                        last_mod_time = current_mod_time
                    else:
                         logger.warning("Konnte keine gültigen Daten für SSE Stream laden.")
                time.sleep(2) 
            except FileNotFoundError:
                 logger.warning(f"Ergebnisdatei '{RESULTS_FILE_PATH}' nicht gefunden während SSE Stream. Warte...")
                 time.sleep(5) 
            except Exception as e:
                logger.error(f"Fehler im SSE Stream: {e}", exc_info=True)
                time.sleep(5) 

    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')


# --- API Endpunkte ---
@app.route('/api/results')
def api_results():
    logger.debug("API-Anfrage für /api/results erhalten")
    data = get_results_data()
    if data is None:
         return jsonify({"error": "Konnte Ergebnisdaten nicht laden."}), 500
    return jsonify(data)


@app.route('/api/evaluation_status')
def api_evaluation_status():
    with state_lock: 
        status_copy = last_evaluation_status.copy()
        status_copy["running"] = evaluation_running and evaluation_thread is not None and evaluation_thread.is_alive()
        if not status_copy["running"] and evaluation_running:
             status_copy["message"] = "Evaluation abgeschlossen (Thread beendet)."
             status_copy["status"] = "idle" 
        elif not status_copy["running"] and not evaluation_running:
              pass 
        return jsonify(status_copy)

def background_evaluation_task(num_comparisons: int):
    global evaluation_running, last_evaluation_status
    thread_id = threading.get_ident()
    logger.info(f"[Thread-{thread_id}] Starte Hintergrund-Evaluation für {num_comparisons} Vergleiche.")

    with state_lock:
        last_evaluation_status = {"running": True, "message": f"Evaluation läuft (0/{num_comparisons})...", "status": "loading"}

    success_count = 0
    error_count = 0
    start_time = time.time()

    try:
        
        if RUN_ALL_CATEGORIES_IS_ENABLED:
            for category in ALL_CATEGORIES:
                try:
                    # firstly gerneate all answers from all models and then judge
                    generate_and_save_model_answers_for_category(config=CONFIG, category=category)
                    run_judge_comparison(CONFIG,category) 
                except Exception as e:
                    logger.exception(f"Critical error during comparison cycle: {e}")
                    break 
        else:
            try:
                generate_and_save_model_answers_for_category(config=CONFIG, category=DEFAULT_CATEGORY)
                run_judge_comparison(CONFIG,category)                     

            except Exception as e:
                logger.exception(f"Critical error: {e}")
    

    except Exception as task_exc:
        logger.error(f"[Thread-{thread_id}] Unerwarteter Fehler in Hintergrund-Evaluationstask: {task_exc}", exc_info=True)
        with state_lock:
            last_evaluation_status = {"running": False, "message": f"Evaluation mit Fehler abgebrochen: {task_exc}", "status": "error"}
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"[Thread-{thread_id}] Hintergrund-Evaluation beendet nach {duration:.2f}s. Erfolgreich: {success_count}, Fehler: {error_count}.")
        with state_lock:
            evaluation_running = False 
            if error_count > 0:
                 last_evaluation_status = {"running": False, "message": f"Evaluation beendet mit {error_count} Fehlern ({success_count} erfolgreich). Dauer: {duration:.1f}s", "status": "warning"}
            else:
                 last_evaluation_status = {"running": False, "message": f"Evaluation erfolgreich abgeschlossen ({success_count} Vergleiche). Dauer: {duration:.1f}s", "status": "success"}


@app.route('/api/run_evaluation', methods=['POST'])
def api_run_evaluation():
    global evaluation_running, evaluation_thread, last_evaluation_status
    logger.info("API-Anfrage für /api/run_evaluation (POST) erhalten")

    with state_lock:
        if evaluation_running and evaluation_thread is not None and evaluation_thread.is_alive():
            logger.warning("Anfrage zum Starten der Evaluation erhalten, aber sie läuft bereits.")
            return jsonify({"error": "Evaluation läuft bereits."}), 409

        logger.info("Starte neuen Hintergrund-Evaluationsthread...")
        evaluation_running = True
        last_evaluation_status = {"running": True, "message": "Starte Evaluation...", "status": "loading"}
        evaluation_thread = threading.Thread(target=background_evaluation_task, args=(COMPARISONS_PER_RUN,))
        evaluation_thread.daemon = True 
        evaluation_thread.start()
        logger.info(f"Hintergrund-Evaluationsthread gestartet (ID: {evaluation_thread.ident}).")
        return jsonify({"message": "Evaluation gestartet."}), 202 

# --- Hauptroute ---
@app.route('/')
def index():
    logger.info("Anfrage für / erhalten")
    # Pass COMPARISONS_PER_RUN to the template
    return render_template('index.html', COMPARISONS_PER_RUN=COMPARISONS_PER_RUN)

# --- Start ---
if __name__ == '__main__':
    logger.info(f"Starte OllamaBench Dashboard Server auf http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    logger.info(f"Debug Modus: {'ON' if DASHBOARD_DEBUG else 'OFF'}")
    try:
        from waitress import serve
        serve(app, host=DASHBOARD_HOST, port=DASHBOARD_PORT, threads=8) 
    except ImportError:
        logger.warning("Waitress nicht gefunden. Fallback auf Flask Development Server (NICHT für Produktion empfohlen).")
        app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DASHBOARD_DEBUG, threaded=True) 

