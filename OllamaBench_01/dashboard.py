import logging
import threading
import time
import os
import json
import datetime as dt
from typing import Any, Dict, Optional, Iterator, List, Set
from flask import Flask, jsonify, render_template_string, request, Response, stream_with_context
from werkzeug.exceptions import NotFound
import filelock 

# Projekt-spezifische Importe
from utils.file_operations import load_json_file 
from utils.config import load_config, DEFAULT_CONFIG_PATH
from scripts.run_comparison import run_comparison_cycle

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

except (FileNotFoundError, KeyError, ValueError, Exception) as e:
    logging.basicConfig(level=logging.ERROR)
    logging.critical(f"FATAL: Konnte Konfiguration nicht laden oder validieren von '{DEFAULT_CONFIG_PATH}': {e}", exc_info=True)
    exit(1)

# --- Logging Setup ---
log_level_str = CONFIG.get("logging", {}).get("level", "INFO").upper()
log_format = CONFIG.get("logging", {}).get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_file = CONFIG.get("paths", {}).get("log_file")

log_handlers = [logging.StreamHandler()]
if log_file:
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        log_handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    except Exception as e:
        logging.error(f"Konnte FileHandler für Logdatei '{log_file}' nicht erstellen: {e}. Logge nur zur Konsole.")

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

HTML_TEMPLATE = f"""
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OllamaBench Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; background-color: #f3f4f6; }}
        .chart-container {{ position: relative; height: 40vh; width: 100%; max-width: 600px; margin: auto; }}
        canvas {{ display: block; box-sizing: border-box; height: 100% !important; width: 100% !important; }}
        .table-container {{ overflow-x: auto; max-height: 400px; /* Erhöhte max Höhe */ }}
        .table-container::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        .table-container::-webkit-scrollbar-track {{ background: #f1f1f1; border-radius: 10px; }}
        .table-container::-webkit-scrollbar-thumb {{ background: #888; border-radius: 10px; }}
        .table-container::-webkit-scrollbar-thumb:hover {{ background: #555; }}
        .loading-spinner {{
            border: 4px solid rgba(0, 0, 0, 0.1); width: 36px; height: 36px;
            border-radius: 50%; border-left-color: #3b82f6; /* Blau */
            animation: spin 1s ease infinite; margin: 5px auto;
            display: inline-block; vertical-align: middle;
        }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
        .action-button {{
            background-color: #3b82f6; color: white; padding: 10px 20px; border: none;
            border-radius: 8px; font-weight: 500; cursor: pointer; transition: background-color 0.3s ease;
            display: inline-flex; align-items: center; gap: 8px; /* Icon und Text ausrichten */
        }}
        .action-button:hover {{ background-color: #2563eb; }}
        .action-button:disabled {{ background-color: #9ca3af; cursor: not-allowed; }}
        .action-button .loading-spinner {{ width: 20px; height: 20px; border-width: 3px; margin: 0; }} /* Spinner im Button */
        .status-message {{
            margin-top: 15px; padding: 10px 15px; border-radius: 6px; font-size: 0.9em;
            text-align: center; transition: opacity 0.5s ease-in-out;
        }}
        .status-success {{ background-color: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }}
        .status-error {{ background-color: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }}
        .status-warning {{ background-color: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }}
        .status-info {{ background-color: #e0e7ff; color: #3730a3; border: 1px solid #a5b4fc; }}
        .status-loading {{
             background-color: #e0e7ff; color: #3730a3; border: 1px solid #a5b4fc;
             display: flex; align-items: center; justify-content: center; gap: 8px;
        }}
        .status-loading .loading-spinner {{ width: 20px; height: 20px; border-width: 3px; margin: 0; }}
        .category-select {{
            padding: 8px 12px; border-radius: 6px; border: 1px solid #d1d5db; /* Gray-300 */
            background-color: white; margin-left: 10px; font-size: 0.9em; cursor: pointer;
        }}
        .hidden {{ display: none; }}
        .sticky-header th {{ position: sticky; top: 0; background-color: #f9fafb; /* Hellgrauer Hintergrund */ z-index: 10; white-space: nowrap; }} /* Wichtig für Spaltenüberschriften */
        .detailed-ratings-table td {{ white-space: nowrap; }} /* Verhindert Umbruch in Zellen */
    </style>
</head>
<body class="p-4 md:p-8">
    <div class="max-w-7xl mx-auto">
        <h1 class="text-2xl md:text-3xl font-bold mb-6 text-center text-gray-800">OllamaBench Dashboard</h1>

        <div class="mb-6 p-4 bg-white rounded-lg shadow-md text-center">
            <button id="run-eval-button" class="action-button">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
                  <path stroke-linecap="round" stroke-linejoin="round" d="M15.91 11.672a.375.375 0 0 1 0 .656l-5.603 3.113a.375.375 0 0 1-.557-.328V8.887c0-.286.307-.466.557-.327l5.603 3.112Z" />
                </svg>
                <span>Starte {COMPARISONS_PER_RUN} Vergleiche</span>
                <div class="loading-spinner hidden"></div>
            </button>
            <div id="eval-status" class="status-message status-info" style="opacity: 1;">Prüfe Status...</div>
        </div>

        <div id="loading-indicator" class="text-center my-8">
            <div class="loading-spinner"></div>
            <p class="text-gray-600 mt-2">Lade Dashboard-Daten...</p>
        </div>

        <div id="dashboard-content" class="hidden">
            <div class="bg-white p-4 md:p-6 rounded-lg shadow-md mb-6">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-xl font-semibold text-gray-700">Gesamtranking der Modelle</h2>
                    <span class="text-xs text-gray-500">Letzte Aktualisierung: <span id="data-timestamp">N/A</span></span>
                </div>
                <div class="table-container">
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="sticky-header">
                            <tr>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rang</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Modell</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Beste Kategorie</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Höchstes ELO</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Alle Kategorien (Durchschnitts-ELO)</th>
                            </tr>
                        </thead>
                        <tbody id="overall-rankings-table-body" class="bg-white divide-y divide-gray-200">
                            <tr><td colspan="5" class="text-center py-4 text-gray-500">Lade Rankings...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                 <div class="bg-white p-4 md:p-6 rounded-lg shadow-md flex flex-col items-center">
                    <div class="flex items-center justify-center w-full mb-4">
                         <h2 class="text-xl font-semibold text-gray-700">ELO-Ratings nach Kategorie</h2>
                         <select id="category-select-bar" class="category-select ml-auto">
                             <option value="">Lade...</option>
                         </select>
                    </div>
                    <div class="chart-container">
                        <canvas id="barChart"></canvas>
                         <p id="bar-chart-no-data" class="text-center text-gray-500 mt-4 hidden">Keine Daten für diese Kategorie verfügbar.</p>
                    </div>
                </div>
                 <div class="bg-white p-4 md:p-6 rounded-lg shadow-md flex flex-col items-center">
                     <div class="flex items-center justify-center w-full mb-4">
                         <h2 class="text-xl font-semibold text-gray-700">Modellvergleich nach Kategorie</h2>
                          <select id="category-select-radar" class="category-select ml-auto">
                             <option value="">Lade...</option>
                          </select>
                     </div>
                     <div class="chart-container">
                        <canvas id="radarChart"></canvas>
                        <p id="radar-chart-no-data" class="text-center text-gray-500 mt-4 hidden">Nicht genügend Daten (>= 3 Modelle benötigt) für Radar-Diagramm.</p>
                    </div>
                </div>
            </div>

             <div class="bg-white p-4 md:p-6 rounded-lg shadow-md mt-6">
                <h2 class="text-xl font-semibold mb-4 text-gray-700">Detaillierte Ratings pro Kategorie</h2>
                <div class="table-container">
                    <table class="min-w-full divide-y divide-gray-200 detailed-ratings-table">
                        <thead id="detailed-ratings-thead" class="sticky-header">
                            </thead>
                        <tbody id="detailed-ratings-tbody" class="bg-white divide-y divide-gray-200">
                             <tr><td colspan="1" class="text-center py-4 text-gray-500">Lade detaillierte Ratings...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        // API Endpunkte
        const API_INITIAL_RESULTS_URL = '/api/results'; // Für den initialen Datenabruf
        const API_RESULTS_STREAM_URL = '/stream';      // Für Server-Sent Events
        const API_RUN_EVAL_URL = '/api/run_evaluation';
        const API_EVAL_STATUS_URL = '/api/evaluation_status';

        // --- Chart Instanzen & Daten ---
        let barChartInstance = null;
        let radarChartInstance = null;
        let currentModelsData = {{}}; // Globale Speicherung der Modelldaten
        let availableCategories = []; // Globale Speicherung verfügbarer Kategorien
        let currentTimestamp = null;
        let eventSource = null; // Für die SSE Verbindung

        // --- Hilfsfunktionen ---
        function generateColors(count) {{
            const colors = [];
            const baseHue = 210; // Start-Farbton (Blau)
            const saturation = 70;
            const lightness = 60;
            for (let i = 0; i < count; i++) {{
                const hue = (baseHue + (i * (360 / (count + 1)))) % 360; // Sicherstellen, dass Farben unterschiedlich sind
                colors.push({{
                    fill: `hsla(${{hue}}, ${{saturation}}%, ${{lightness}}%, 0.5)`, // Mehr Transparenz
                    stroke: `hsla(${{hue}}, ${{saturation}}%, ${{lightness - 10}}%, 1)` // Etwas dunklerer Rand
                }});
            }}
            return colors;
        }}

        function formatTimestamp(isoTimestamp) {{
            if (!isoTimestamp) return 'N/A';
            try {{
                const date = new Date(isoTimestamp);
                return new Intl.DateTimeFormat('de-DE', {{
                    year: 'numeric', month: '2-digit', day: '2-digit',
                    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
                }}).format(date);
            }} catch (e) {{
                console.error("Fehler beim Formatieren des Zeitstempels:", isoTimestamp, e);
                return isoTimestamp; // Fallback auf ISO-String
            }}
        }}

        function capitalizeFirstLetter(string) {{
            if (!string) return '';
            return string.charAt(0).toUpperCase() + string.slice(1);
        }}

        // --- Status Update Funktionen ---
         function showStatusMessage(message, type = 'info', sticky = false) {{
            const statusDiv = document.getElementById('eval-status');
            if (!statusDiv) return;

            statusDiv.innerHTML = ''; // Löscht vorherigen Inhalt (auch Spinner)
            statusDiv.className = 'status-message'; // Klassen zurücksetzen
            statusDiv.style.opacity = 1; // Sichtbar machen

            let statusClass = '';
            switch (type) {{
                case 'success': statusClass = 'status-success'; break;
                case 'error': statusClass = 'status-error'; break;
                case 'warning': statusClass = 'status-warning'; break;
                case 'loading': statusClass = 'status-loading'; break;
                default: statusClass = 'status-info'; break;
            }}
            statusDiv.classList.add(statusClass);

            if (type === 'loading') {{
                // Füge Spinner hinzu, wenn 'loading'
                const spinner = document.createElement('div');
                spinner.className = 'loading-spinner';
                statusDiv.appendChild(spinner);
                statusDiv.appendChild(document.createTextNode(' ' + message)); // Text nach Spinner
            }} else {{
                statusDiv.textContent = message;
            }}

            // Nachricht nach einiger Zeit ausblenden, außer wenn sticky
            if (!sticky && type !== 'loading') {{ // Lade-Nachrichten bleiben bestehen
                setTimeout(() => {{
                    if (statusDiv.classList.contains(statusClass)) {{ // Nur ausblenden, wenn es noch die gleiche Nachricht ist
                         statusDiv.style.opacity = 0;
                         // Optional: Nachricht nach Ausblenden entfernen
                         // setTimeout(() => {{ if (statusDiv.style.opacity == 0) statusDiv.textContent = ''; }}, 500);
                    }}
                }}, 5000); // 5 Sekunden
            }}
        }}

        function updateEvaluationButton(isLoading, message = null) {{
            const runButton = document.getElementById('run-eval-button');
            const buttonText = runButton?.querySelector('span');
            const spinner = runButton?.querySelector('.loading-spinner');
            if (!runButton || !buttonText || !spinner) return;

            runButton.disabled = isLoading;
            if (isLoading) {{
                spinner.classList.remove('hidden');
                buttonText.textContent = message || 'Läuft...';
            }} else {{
                spinner.classList.add('hidden');
                buttonText.textContent = `Starte {COMPARISONS_PER_RUN} Vergleiche`;
            }}
        }}

        // --- Chart Update Funktionen ---
        function updateBarChart(selectedCategory) {{
            console.debug(`Aktualisiere Balkendiagramm für Kategorie: ${{selectedCategory}}`);
            const barCtx = document.getElementById('barChart')?.getContext('2d');
            const noDataElement = document.getElementById('bar-chart-no-data');
            if (!barCtx || !noDataElement) {{ console.error("Balkendiagramm Canvas oder no-data Element nicht gefunden."); return; }}
            if (barChartInstance) barChartInstance.destroy();
            noDataElement.classList.add('hidden');

            const labels = [];
            const ratings = [];
            Object.entries(currentModelsData).forEach(([modelName, modelDetails]) => {{
                const categoryRating = modelDetails.categorie?.[selectedCategory];
                if (categoryRating !== undefined && typeof categoryRating === 'number') {{
                    labels.push(modelName);
                    ratings.push(categoryRating);
                }}
            }});

            const sortedIndices = ratings.map((_, i) => i).sort((a, b) => ratings[b] - ratings[a]);
            const sortedLabels = sortedIndices.map(i => labels[i]);
            const sortedRatings = sortedIndices.map(i => ratings[i]);

            if (sortedLabels.length === 0) {{
                console.warn(`Keine Daten für Kategorie ${{selectedCategory}} im Balkendiagramm.`);
                noDataElement.classList.remove('hidden');
                return;
            }}

            const chartColors = generateColors(sortedLabels.length);
            const barBgColors = sortedIndices.map(i => chartColors[i % chartColors.length].fill);
            const barBorderColors = sortedIndices.map(i => chartColors[i % chartColors.length].stroke);

            barChartInstance = new Chart(barCtx, {{
                type: 'bar',
                data: {{
                    labels: sortedLabels,
                    datasets: [{{
                        label: `ELO Rating (${{capitalizeFirstLetter(selectedCategory)}})`,
                        data: sortedRatings,
                        backgroundColor: barBgColors,
                        borderColor: barBorderColors,
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false, indexAxis: 'y',
                    scales: {{
                        y: {{ title: {{ display: true, text: 'Modell' }} }},
                        x: {{ beginAtZero: false, title: {{ display: true, text: 'ELO Rating' }} }}
                    }},
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{ callbacks: {{ label: (ctx) => `${{ctx.dataset.label || ''}}: ${{ctx.parsed.x !== null ? ctx.parsed.x.toFixed(1) : 'N/A'}}` }} }}
                    }}
                }}
            }});
        }}

        function updateRadarChart(selectedCategory) {{
            console.debug(`Aktualisiere Radardiagramm für Kategorie: ${{selectedCategory}}`);
            const radarCtx = document.getElementById('radarChart')?.getContext('2d');
            const noDataElement = document.getElementById('radar-chart-no-data');
             if (!radarCtx || !noDataElement) {{ console.error("Radardiagramm Canvas oder no-data Element nicht gefunden."); return; }}
            if (radarChartInstance) radarChartInstance.destroy();
            noDataElement.classList.add('hidden');

            const labels = [];
            const ratings = [];
             Object.entries(currentModelsData).forEach(([modelName, modelDetails]) => {{
                const categoryRating = modelDetails.categorie?.[selectedCategory];
                if (categoryRating !== undefined && typeof categoryRating === 'number') {{
                    labels.push(modelName);
                    ratings.push(categoryRating);
                }}
            }});

            if (labels.length < 3) {{
                console.warn(`Nicht genug Daten (gefunden ${{labels.length}}, >= 3 benötigt) für Kategorie ${{selectedCategory}} im Radardiagramm.`);
                noDataElement.classList.remove('hidden');
                return;
            }}

            const radarColors = generateColors(1)[0];
            const minRating = Math.min(...ratings);
            const maxRating = Math.max(...ratings);
            const scalePadding = Math.max(20, (maxRating - minRating) * 0.1);

            radarChartInstance = new Chart(radarCtx, {{
                type: 'radar',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: `ELO Rating (${{capitalizeFirstLetter(selectedCategory)}})`,
                        data: ratings,
                        backgroundColor: radarColors.fill, borderColor: radarColors.stroke,
                        pointBackgroundColor: radarColors.stroke, pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff', pointHoverBorderColor: radarColors.stroke,
                        borderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    scales: {{
                        r: {{
                            beginAtZero: false, angleLines: {{ display: true }},
                            suggestedMin: Math.floor(minRating - scalePadding / 2),
                            suggestedMax: Math.ceil(maxRating + scalePadding),
                            pointLabels: {{ font: {{ size: 11 }} }},
                            ticks: {{ backdropColor: 'rgba(255, 255, 255, 0.75)', stepSize: 50 }}
                        }}
                    }},
                    plugins: {{
                        legend: {{ position: 'top' }},
                        tooltip: {{ callbacks: {{ label: (ctx) => `${{ctx.label}}: ${{ctx.parsed.r !== null ? ctx.parsed.r.toFixed(1) : 'N/A'}}` }} }}
                    }},
                    elements: {{ line: {{ borderWidth: 3 }} }}
                }}
            }});
        }}

        // --- Daten-Darstellungsfunktionen ---
        function populateOverallRankings(modelsData) {{
            const tableBody = document.getElementById('overall-rankings-table-body');
            if (!tableBody) return;
            tableBody.innerHTML = ''; // Vorherige Daten löschen
            const modelStats = [];

            Object.entries(modelsData).forEach(([modelName, modelDetails]) => {{
                const categories = modelDetails.categorie || {{}};
                if (Object.keys(categories).length > 0) {{
                    let highestRating = -Infinity;
                    let highestCategory = 'N/A';
                    let totalRating = 0;
                    let categoryCount = 0;
                    Object.entries(categories).forEach(([cat, rating]) => {{
                        if (typeof rating === 'number') {{
                            if (rating > highestRating) {{ highestRating = rating; highestCategory = cat; }}
                            totalRating += rating;
                            categoryCount++;
                        }}
                    }});
                    const avgRating = categoryCount > 0 ? totalRating / categoryCount : 0;
                    modelStats.push({{
                        name: modelName, highestCategory: highestCategory,
                        highestRating: highestRating, avgRating: avgRating, categoryCount: categoryCount
                    }});
                }}
            }});

            modelStats.sort((a, b) => b.highestRating - a.highestRating); // Nach höchstem ELO sortieren

            if (modelStats.length === 0) {{
                tableBody.innerHTML = '<tr><td colspan="5" class="px-6 py-4 text-sm text-gray-500 text-center">Keine Modelle mit Ratings gefunden. Starte Vergleiche.</td></tr>';
                return;
            }}

            modelStats.forEach((stats, index) => {{
                const row = `
                <tr>
                    <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${{index + 1}}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${{stats.name}}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${{capitalizeFirstLetter(stats.highestCategory)}}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${{stats.highestRating.toFixed(1)}}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${{stats.avgRating.toFixed(1)}} (${{stats.categoryCount}} Kategorien)</td>
                </tr>`;
                tableBody.innerHTML += row;
            }});
        }}

        function populateDetailedTable(modelsData, categories) {{
            const tableHead = document.getElementById('detailed-ratings-thead');
            const tableBody = document.getElementById('detailed-ratings-tbody');
            if (!tableHead || !tableBody) return;

            // --- Header generieren ---
            tableHead.innerHTML = ''; // Alten Header löschen
            let headerHtml = '<tr><th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Modell</th>';
            if (categories.length === 0) {{
                 headerHtml += '<th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Keine Kategorien</th>';
            }} else {{
                categories.forEach(cat => {{
                    headerHtml += `<th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">${{capitalizeFirstLetter(cat)}}</th>`;
                }});
            }}
            headerHtml += '</tr>';
            tableHead.innerHTML = headerHtml;

            // --- Body generieren ---
            tableBody.innerHTML = ''; // Alten Body löschen
            const modelNames = Object.keys(modelsData).sort(); // Modelle alphabetisch sortieren

            if (modelNames.length === 0) {{
                tableBody.innerHTML = `<tr><td colspan="${{categories.length + 1}}" class="text-center py-4 text-gray-500">Keine Modelldaten verfügbar.</td></tr>`;
                return;
            }}

            modelNames.forEach(modelName => {{
                const modelDetails = modelsData[modelName];
                const categoryRatings = modelDetails.categorie || {{}};
                let rowHtml = `<tr><td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${{modelName}}</td>`;
                if (categories.length === 0) {{
                     rowHtml += '<td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">N/A</td>';
                }} else {{
                    categories.forEach(cat => {{
                        const rating = categoryRatings[cat];
                        const displayRating = (typeof rating === 'number') ? rating.toFixed(1) : 'N/A';
                        rowHtml += `<td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${{displayRating}}</td>`;
                    }});
                }}
                rowHtml += '</tr>';
                tableBody.innerHTML += rowHtml;
            }});
             if (tableBody.innerHTML === '') {{ // Fallback, falls keine Modelle Ratings hatten
                 tableBody.innerHTML = `<tr><td colspan="${{categories.length + 1}}" class="text-center py-4 text-gray-500">Keine detaillierten Ratings verfügbar.</td></tr>`;
            }}
        }}


        function populateCategoryDropdowns(categories) {{
            const selects = [document.getElementById('category-select-bar'), document.getElementById('category-select-radar')];
            selects.forEach(select => {{
                if (!select) return;
                const currentValue = select.value; // Auswahl merken
                select.innerHTML = ''; // Optionen löschen
                if (categories.length === 0) {{
                    select.innerHTML = '<option value="">Keine Kategorien</option>';
                    select.disabled = true;
                }} else {{
                    select.disabled = false;
                    categories.forEach(category => {{
                        const option = document.createElement('option');
                        option.value = category;
                        option.textContent = capitalizeFirstLetter(category);
                        select.appendChild(option);
                    }});
                    // Auswahl wiederherstellen oder erste Kategorie wählen
                    if (categories.includes(currentValue)) {{
                        select.value = currentValue;
                    }} else {{
                        select.value = categories[0];
                    }}
                }}
            }});
        }}

        // --- Dashboard Update Logik ---
        function updateDashboard(data) {{
            const dashboardContent = document.getElementById('dashboard-content');
            const loadingIndicator = document.getElementById('loading-indicator');
             if (!data || !data.models) {{
                 console.error("Ungültige Datenstruktur empfangen:", data);
                 loadingIndicator.innerHTML = '<p class="text-red-600 font-semibold">Fehler beim Empfangen der Daten.</p>';
                 dashboardContent.classList.add('hidden');
                 return;
             }}

            currentModelsData = data.models;
            currentTimestamp = data.timestamp;
            document.getElementById('data-timestamp').textContent = formatTimestamp(currentTimestamp);

            // Kategorien extrahieren und UI aktualisieren
            const categories = new Set();
            Object.values(currentModelsData).forEach(details => {{
                Object.keys(details.categorie || {{}}).forEach(cat => categories.add(cat));
            }});
            availableCategories = Array.from(categories).sort();

            if (Object.keys(currentModelsData).length === 0) {{
                console.warn("Keine Modelldaten in den Ergebnissen.");
                document.getElementById('overall-rankings-table-body').innerHTML = '<tr><td colspan="5" class="px-6 py-4 text-sm text-gray-500 text-center">Keine Modelldaten verfügbar. Starte Vergleiche.</td></tr>';
                populateDetailedTable({{}}, []); // Leere Tabelle und Header
                populateCategoryDropdowns([]);
                if (barChartInstance) barChartInstance.destroy();
                if (radarChartInstance) radarChartInstance.destroy();
                document.getElementById('bar-chart-no-data').classList.remove('hidden');
                document.getElementById('radar-chart-no-data').classList.remove('hidden');
            }} else {{
                populateOverallRankings(currentModelsData);
                populateDetailedTable(currentModelsData, availableCategories); // Mit Kategorien für Header
                populateCategoryDropdowns(availableCategories);

                const selectedCategoryBar = document.getElementById('category-select-bar').value;
                const selectedCategoryRadar = document.getElementById('category-select-radar').value;

                if (availableCategories.length > 0) {{
                    updateBarChart(selectedCategoryBar || availableCategories[0]);
                    updateRadarChart(selectedCategoryRadar || availableCategories[0]);
                }} else {{
                    console.warn("Keine Kategorien zum Aktualisieren der Diagramme gefunden.");
                    if (barChartInstance) barChartInstance.destroy();
                    if (radarChartInstance) radarChartInstance.destroy();
                    document.getElementById('bar-chart-no-data').classList.remove('hidden');
                    document.getElementById('radar-chart-no-data').classList.remove('hidden');
                }}
            }}

            // Ladeanzeige ausblenden und Inhalt anzeigen
            loadingIndicator.classList.add('hidden');
            dashboardContent.classList.remove('hidden');
        }}

        // --- Event Listener und Initialisierung ---
        document.addEventListener('DOMContentLoaded', () => {{
            const loadingIndicator = document.getElementById('loading-indicator');
            const dashboardContent = document.getElementById('dashboard-content');
            const runButton = document.getElementById('run-eval-button');
            const categorySelectBar = document.getElementById('category-select-bar');
            const categorySelectRadar = document.getElementById('category-select-radar');

            // Initialen Status prüfen
            fetchEvaluationStatus();

             // Event Listener für Buttons und Dropdowns
            if (runButton) {{
                runButton.addEventListener('click', runEvaluation);
            }}
            if (categorySelectBar) {{
                categorySelectBar.addEventListener('change', (event) => {{
                    if (availableCategories.length > 0) {{
                        updateBarChart(event.target.value);
                    }}
                }});
            }}
             if (categorySelectRadar) {{
                categorySelectRadar.addEventListener('change', (event) => {{
                    if (availableCategories.length > 0) {{
                        updateRadarChart(event.target.value);
                    }}
                }});
            }}

            // --- Initialer Datenabruf ---
            fetch(API_INITIAL_RESULTS_URL)
                .then(response => {{
                    if (!response.ok) {{
                        throw new Error(`HTTP Fehler! Status: ${{response.status}}`);
                    }}
                    return response.json();
                }})
                .then(data => {{
                    console.log("Initiale Daten geladen:", data);
                    updateDashboard(data); // Dashboard mit initialen Daten füllen
                }})
                .catch(error => {{
                    console.error('Fehler beim initialen Laden der Dashboard-Daten:', error);
                    loadingIndicator.innerHTML = '<p class="text-red-600 font-semibold">Fehler beim Laden der initialen Daten.</p>';
                    dashboardContent.classList.add('hidden');
                }});

            // --- SSE Verbindung aufbauen ---
            function connectEventSource() {{
                if (eventSource) {{
                    eventSource.close(); // Schließe alte Verbindung, falls vorhanden
                }}
                console.log("Verbinde mit SSE Stream:", API_RESULTS_STREAM_URL);
                eventSource = new EventSource(API_RESULTS_STREAM_URL);

                eventSource.onmessage = function(event) {{
                    console.log("SSE Nachricht empfangen:", event.data);
                    try {{
                        const data = JSON.parse(event.data);
                        updateDashboard(data); // Dashboard mit neuen Daten aktualisieren
                    }} catch (e) {{
                        console.error("Fehler beim Parsen der SSE Daten:", e, "Daten:", event.data);
                    }}
                }};

                eventSource.onerror = function(err) {{
                    console.error("EventSource Fehler:", err);
                    showStatusMessage("Verbindung zum Server verloren. Versuche erneut...", "warning", true);
                    eventSource.close();
                    // Optional: Nach kurzer Pause erneut verbinden
                    setTimeout(connectEventSource, 5000); // Versuche nach 5s erneut
                }};

                eventSource.onopen = function() {{
                     console.log("SSE Verbindung geöffnet.");
                     // Statusmeldung ggf. zurücksetzen, wenn sie eine Warnung war
                     const statusDiv = document.getElementById('eval-status');
                     if (statusDiv && statusDiv.classList.contains('status-warning')) {{
                         fetchEvaluationStatus(); // Aktuellen Eval-Status holen
                     }}
                }};
            }}

            connectEventSource(); // Erste Verbindung herstellen

        }}); // Ende DOMContentLoaded

        // --- API Aufrufe ---
        async function fetchEvaluationStatus() {{
            try {{
                const response = await fetch(API_EVAL_STATUS_URL);
                if (!response.ok) {{
                    throw new Error(`HTTP Fehler! Status: ${{response.status}}`);
                }}
                const statusData = await response.json();
                console.log("Evaluationsstatus:", statusData);
                 last_evaluation_status = statusData; // Globalen Status aktualisieren
                updateEvaluationButton(statusData.running, statusData.message);
                showStatusMessage(statusData.message, statusData.status || 'info', statusData.running); // Status anzeigen
            }} catch (error) {{
                console.error('Fehler beim Abrufen des Evaluationsstatus:', error);
                showStatusMessage("Fehler beim Abrufen des Evaluationsstatus.", "error");
                updateEvaluationButton(false); // Button sicherheitshalber freigeben
            }}
        }}

        async function runEvaluation() {{
            updateEvaluationButton(true, "Starte Evaluation...");
            showStatusMessage("Sende Anfrage zum Starten der Evaluation...", "loading", true);
            try {{
                const response = await fetch(API_RUN_EVAL_URL, {{ method: 'POST' }});
                if (!response.ok) {{
                     // Versuche, Fehlerdetails aus der Antwort zu lesen
                     let errorMsg = `HTTP Fehler! Status: ${{response.status}}`;
                     try {{
                         const errorData = await response.json();
                         errorMsg = errorData.error || errorMsg;
                     }} catch (jsonError) {{ /* Ignoriere JSON Parse Fehler hier */ }}
                    throw new Error(errorMsg);
                }}
                const result = await response.json();
                console.log("Evaluationsstart Antwort:", result);
                // Status sofort aktualisieren (Backend sollte den Status ändern)
                fetchEvaluationStatus();
                // Die SSE Verbindung wird die Daten aktualisieren, wenn die Evaluation abgeschlossen ist
            }} catch (error) {{
                console.error('Fehler beim Starten der Evaluation:', error);
                showStatusMessage(`Fehler beim Starten: ${{error.message}}`, "error");
                updateEvaluationButton(false); // Button wieder freigeben
            }}
        }}

    </script>
</body>
</html>
"""

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
        for i in range(num_comparisons):
            logger.info(f"[Thread-{thread_id}] Führe Vergleich {i+1}/{num_comparisons} aus...")
            with state_lock:
                 last_evaluation_status["message"] = f"Evaluation läuft ({i+1}/{num_comparisons})..."

            try:
                updated_ratings = run_comparison_cycle(CONFIG)
                if updated_ratings is not None:
                    success_count += 1
                else:
                    error_count += 1
                    logger.warning(f"[Thread-{thread_id}] Vergleichszyklus {i+1} nicht erfolgreich abgeschlossen oder gab None zurück.")
                    time.sleep(1)

            except Exception as cycle_exc:
                error_count += 1
                logger.error(f"[Thread-{thread_id}] Schwerer Fehler in Vergleichszyklus {i+1}: {cycle_exc}", exc_info=True)
                time.sleep(2)

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
    return render_template_string(HTML_TEMPLATE, COMPARISONS_PER_RUN=COMPARISONS_PER_RUN)

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

