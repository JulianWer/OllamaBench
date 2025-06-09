import logging
import threading
import time
import os
import json
import datetime as dt
from typing import Any, Dict, Optional, List, Set
from flask import Flask, jsonify, render_template, request, Response, stream_with_context
import filelock
from scripts.engine import run_benchmark
from utils.config import load_config, DEFAULT_CONFIG_PATH

# --- Configuration Loading ---
CONFIG = None
try:
    CONFIG = load_config(DEFAULT_CONFIG_PATH)
    RESULTS_FILE_PATH = CONFIG["paths"]["results_file"]
    LOCK_FILE_PATH = CONFIG["paths"]["lock_file"]
    DASHBOARD_HOST = CONFIG.get("dashboard", {}).get("host", "127.0.0.1")
    DASHBOARD_PORT = CONFIG.get("dashboard", {}).get("port", 5001)
    DASHBOARD_DEBUG = CONFIG.get("dashboard", {}).get("debug", False)
    ALL_CATEGORIES = CONFIG.get("ALL_CATEGORIES")


except (FileNotFoundError, KeyError, ValueError, Exception) as e:
    logging.basicConfig(level=logging.ERROR)
    logging.critical(f"FATAL: Failed to load or validate configuration from '{DEFAULT_CONFIG_PATH}': {e}", exc_info=True)
    exit(1)

# --- Logging Setup ---
log_level_str = CONFIG.get("logging", {}).get("level", "INFO").upper()
log_format = CONFIG.get("logging", {}).get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

log_handlers = [logging.StreamHandler()]

logging.basicConfig(level=getattr(logging, log_level_str, logging.INFO),
                    format=log_format,
                    handlers=log_handlers)

logger = logging.getLogger(__name__)
logger.info("Dashboard configuration loaded successfully.")
logger.info(f"Results file: {RESULTS_FILE_PATH}")
logger.info(f"Lock file: {LOCK_FILE_PATH}")

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Global State (Thread Safety) ---
state_lock = threading.Lock()
evaluation_running = False
evaluation_thread: Optional[threading.Thread] = None
last_evaluation_status: Dict[str, Any] = {"running": False, "message": "No evaluation has been started yet.", "status": "idle"}

results_lock = filelock.FileLock(LOCK_FILE_PATH, timeout=10)

def get_results_data() -> Optional[Dict[str, Any]]:
    """
    Safely reads and returns the contents of the results JSON file.
    Handles file locking, missing files, and corrupt JSON data.
    """
    try:
        with results_lock:
            if not os.path.exists(RESULTS_FILE_PATH) or os.path.getsize(RESULTS_FILE_PATH) == 0:
                logger.warning(f"Results file '{RESULTS_FILE_PATH}' not found or empty.")
                return {"timestamp": dt.datetime.utcnow().isoformat(), "models": {}}
            with open(RESULTS_FILE_PATH, "r", encoding='utf-8') as f:
                data = json.load(f)
            
            # Basic data validation and normalization
            if isinstance(data, dict) and "timestamp" in data and "models" in data:
                 if isinstance(data.get("timestamp"), (int, float)):
                      data["timestamp"] = dt.datetime.fromtimestamp(data["timestamp"], tz=dt.timezone.utc).isoformat()
                 elif not isinstance(data.get("timestamp"), str):
                      data["timestamp"] = dt.datetime.utcnow().isoformat()

                 if not isinstance(data.get("models"), dict):
                     logger.warning("Invalid 'models' structure in results.json. Resetting to empty.")
                     data["models"] = {}

                 return data
            else:
                logger.error(f"Invalid structure in results file: {RESULTS_FILE_PATH}. Content: {str(data)[:200]}")
                return {"timestamp": dt.datetime.utcnow().isoformat(), "models": {}}
    except filelock.Timeout:
        logger.error(f"Timeout while waiting for lock on results file: '{LOCK_FILE_PATH}'")
        return None # Indicates a failure to load data
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from: '{RESULTS_FILE_PATH}'. The file might be corrupt.", exc_info=True)
        return {"timestamp": dt.datetime.utcnow().isoformat(), "models": {}} # Return empty structure on corruption
    except Exception as e:
        logger.error(f"Error reading the results file '{RESULTS_FILE_PATH}': {e}", exc_info=True)
        return None

# --- SSE Endpoint ---
@app.route('/stream')
def stream():
    """Streams results file updates to the client using Server-Sent Events."""
    def event_stream():
        last_mod_time = None
        while True:
            try:
                # Check for file modification to avoid unnecessary reads
                current_mod_time = os.path.getmtime(RESULTS_FILE_PATH)
                if last_mod_time is None or current_mod_time > last_mod_time:
                    logger.info(f"Change detected in '{RESULTS_FILE_PATH}'. Sending update.")
                    data = get_results_data()
                    if data:
                        json_data = json.dumps(data)
                        yield f"data: {json_data}\n\n"
                        last_mod_time = current_mod_time
                    else:
                         logger.warning("Could not load valid data for SSE stream.")
                time.sleep(2) # Poll interval
            except FileNotFoundError:
                 logger.warning(f"Results file '{RESULTS_FILE_PATH}' not found during SSE stream. Waiting...")
                 time.sleep(5)
            except Exception as e:
                logger.error(f"Error in SSE stream: {e}", exc_info=True)
                # Avoid busy-looping on persistent errors
                time.sleep(5)

    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

# --- API Endpoints ---
@app.route('/api/results')
def api_results():
    """API endpoint to get the latest results."""
    logger.debug("API request for /api/results received")
    data = get_results_data()
    if data is None:
         return jsonify({"error": "Could not load result data."}), 500
    return jsonify(data)

@app.route('/api/evaluation_status')
def api_evaluation_status():
    """API endpoint to get the current status of the evaluation process."""
    with state_lock:
        status_copy = last_evaluation_status.copy()
        # Ensure 'running' status is always up-to-date
        is_alive = evaluation_thread is not None and evaluation_thread.is_alive()
        status_copy["running"] = evaluation_running and is_alive
        
        # If the master switch is off but the thread just died, update the message
        if not status_copy["running"] and evaluation_running:
             status_copy["message"] = "Evaluation finished (thread ended)."
             status_copy["status"] = "idle"
        
        return jsonify(status_copy)

def background_evaluation_task(dynamic_config):
    """The actual evaluation logic that runs in a background thread."""
    global evaluation_running, last_evaluation_status
    thread_id = threading.get_ident()
    logger.info(f"[Thread-{thread_id}] Starting background evaluation.")

    with state_lock:
        last_evaluation_status = {"running": True, "message": f"Evaluation running...", "status": "loading"}

    success_count = 0
    error_count = 0
    start_time = time.time()
    categories_to_run: List[str]
    
    try:
        if dynamic_config['category']:
            categories_to_run = [dynamic_config['category']]
            logger.info(f"Preparing to run for a single specified category: {dynamic_config['category']}")
        else:
            if not dynamic_config['all_categories']:
                logger.info("No category specified, defaulting to --all-categories.")
            
            categories_to_run = ALL_CATEGORIES # Fallback

        complete_run = not dynamic_config['generate_only'] and not dynamic_config['judge_only']
        generate_answers = not dynamic_config['judge_only'] and dynamic_config['generate_only'] or complete_run
        run_judgement =  not dynamic_config['generate_only'] and dynamic_config['judge_only'] or complete_run
        start_time, error_count = run_benchmark(config=CONFIG, categories_to_run=categories_to_run, generate_answers= generate_answers, run_judgement=run_judgement)

    except Exception as task_exc:
        logger.error(f"[Thread-{thread_id}] Unexpected error in background evaluation task: {task_exc}", exc_info=True)
        with state_lock:
            last_evaluation_status = {"running": False, "message": f"Evaluation aborted with error: {task_exc}", "status": "error"}
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"[Thread-{thread_id}] Background evaluation finished after {duration:.2f}s. Successful: {success_count}, Errors: {error_count}.")
        with state_lock:
            evaluation_running = False
            if error_count > 0:
                 last_evaluation_status = {"running": False, "message": f"Evaluation finished with {error_count} errors ({success_count} successful). Duration: {duration:.1f}s", "status": "warning"}
            else:
                 last_evaluation_status = {"running": False, "message": f"Evaluation completed successfully ({success_count} successful). Duration: {duration:.1f}s", "status": "success"}

@app.route('/api/run_evaluation', methods=['POST'])
def api_run_evaluation():
    """API endpoint to trigger a new evaluation run."""
    global evaluation_running, evaluation_thread, last_evaluation_status
    logger.info("API request for /api/run_evaluation (POST) received")

    payload = request.get_json()
    if not payload:
        return jsonify({"error": "No JSON-Payload found."}), 400

    logger.info(f"Payload: {payload}")

    with state_lock:
        if evaluation_running and evaluation_thread is not None and evaluation_thread.is_alive():
            logger.warning("Request to start evaluation received, but it is already running.")
            return jsonify({"error": "Evaluation is already running."}), 409 # 409 Conflict

        logger.info("Starting new background evaluation thread...")
        evaluation_running = True
        last_evaluation_status = {"running": True, "message": "Starting evaluation...", "status": "loading"}
        evaluation_thread = threading.Thread(target=background_evaluation_task,args=(payload,))
        evaluation_thread.daemon = True # Allows main thread to exit even if this thread is running
        evaluation_thread.start()
        logger.info(f"Background evaluation thread started (ID: {evaluation_thread.ident}).")
        return jsonify({"message": "Evaluation started."}), 202 

# --- Main Route ---
@app.route('/')
def index():
    """Serves the main dashboard page."""
    logger.info("Request for / received")
    return render_template('index.html')

# --- Start Server ---
if __name__ == '__main__':
    logger.info(f"Starting LocalBench Dashboard Server on http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    logger.info(f"Debug Mode: {'ON' if DASHBOARD_DEBUG else 'OFF'}")
    try:
        from waitress import serve
        serve(app, host=DASHBOARD_HOST, port=DASHBOARD_PORT, threads=8)
    except ImportError:
        logger.warning("Waitress not found. Falling back to Flask Development Server (NOT recommended for production).")
        app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DASHBOARD_DEBUG, threaded=True)