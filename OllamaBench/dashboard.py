import logging
import threading
from flask import Flask, jsonify, render_template_string
from utils.file_operations import load_current_json
from utils.config import load_config
from scripts.run_comparison import run_comparison_cycle

# --- Konfiguration ---
CONFIG_PATH = './config/config.yaml'
try:
    config = load_config(CONFIG_PATH)
    RESULTS_FILE_PATH = config.get("paths", {}).get("results_file", "data/results.json")
    LOCK_FILE_PATH = config.get("paths", {}).get("lock_file", "data/results.lock")
except FileNotFoundError:
    logging.error(f"Konfigurationsdatei '{CONFIG_PATH}' nicht gefunden.")
    exit(1)
except Exception as e:
    logging.error(f"Fehler beim Laden der Konfiguration: {e}")
    exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

evaluation_running = False
evaluation_thread = None

# --- HTML Vorlage mit Dropdown und aktualisiertem JavaScript ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Bewertungs-Dashboard (Dynamisch)</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #f3f4f6; }
        .chart-container { position: relative; height: 40vh; width: 100%; max-width: 600px; margin: auto; }
        canvas { display: block; box-sizing: border-box; height: 100% !important; width: 100% !important; }
        .table-container::-webkit-scrollbar { width: 8px; height: 8px; }
        .table-container::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 10px; }
        .table-container::-webkit-scrollbar-thumb { background: #888; border-radius: 10px; }
        .table-container::-webkit-scrollbar-thumb:hover { background: #555; }
        .loading-spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            border-left-color: #09f;
            animation: spin 1s ease infinite;
            margin: 5px auto; /* Adjusted margin */
            display: inline-block; /* Make it inline */
            vertical-align: middle; /* Align vertically */
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        /* Button Styling */
        .action-button {
            background-color: #3b82f6; /* Blue */
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .action-button:hover {
            background-color: #2563eb; /* Darker Blue */
        }
        .action-button:disabled {
            background-color: #9ca3af; /* Gray */
            cursor: not-allowed;
        }
        /* Status Message Styling */
        .status-message {
            margin-top: 10px;
            padding: 8px;
            border-radius: 4px;
            font-size: 0.9em;
            text-align: center;
        }
        .status-success {
            background-color: #d1fae5; /* Green tint */
            color: #065f46; /* Dark Green */
        }
        .status-error {
            background-color: #fee2e2; /* Red tint */
            color: #991b1b; /* Dark Red */
        }
        .status-loading {
             background-color: #e0e7ff; /* Indigo tint */
             color: #3730a3; /* Dark Indigo */
             display: flex;
             align-items: center;
             justify-content: center;
        }
         .status-loading .loading-spinner { /* Smaller spinner for status */
             width: 20px;
             height: 20px;
             border-width: 3px;
             margin-right: 8px;
             margin-top: 0;
             margin-bottom: 0;
         }
         /* Dropdown Styling */
         .category-select {
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid #d1d5db; /* Gray-300 */
            background-color: white;
            margin-left: 10px;
            font-size: 0.9em;
            cursor: pointer;
         }
    </style>
</head>
<body class="p-4 md:p-8">
    <h1 class="text-2xl md:text-3xl font-bold mb-6 text-center text-gray-800">LLM Bewertungs-Dashboard (Dynamisch)</h1>

    <div class="mb-6 p-4 bg-white rounded-lg shadow-md text-center">
        <button id="run-eval-button" class="action-button">Neuen Durchlauf starten</button>
        <div id="eval-status" class="status-message" style="display: none;"></div>
    </div>

    <div id="loading-indicator" class="text-center my-8">
        <div class="loading-spinner"></div>
        <p>Lade Daten...</p>
    </div>

    <div id="dashboard-content" class="hidden grid grid-cols-1 gap-6">
        <div class="bg-white p-4 md:p-6 rounded-lg shadow-md">
            <h2 class="text-xl font-semibold mb-4 text-gray-700">Bewertungstabelle (Alle Kategorien)</h2>
            <div class="table-container overflow-x-auto max-h-96">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50 sticky top-0">
                        <tr>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Modell</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Kategorie</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ELO Bewertung</th>
                        </tr>
                    </thead>
                    <tbody id="ratings-table-body" class="bg-white divide-y divide-gray-200">
                        </tbody>
                </table>
            </div>
             <p class="text-xs text-gray-500 mt-2 text-right">Stand: <span id="data-timestamp">N/A</span></p>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
             <div class="bg-white p-4 md:p-6 rounded-lg shadow-md flex flex-col items-center">
                <div class="flex items-center justify-center w-full mb-4">
                     <h2 class="text-xl font-semibold text-gray-700">ELO Bewertungen</h2>
                     <select id="category-select-bar" class="category-select">
                         </select>
                </div>
                <div class="chart-container">
                    <canvas id="barChart"></canvas>
                </div>
            </div>
             <div class="bg-white p-4 md:p-6 rounded-lg shadow-md flex flex-col items-center">
                 <div class="flex items-center justify-center w-full mb-4">
                     <h2 class="text-xl font-semibold text-gray-700">Bewertungsvergleich</h2>
                      <select id="category-select-radar" class="category-select">
                         </select>
                 </div>
                 <div class="chart-container">
                    <canvas id="radarChart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Hilfsfunktionen
        function generateColors(count) {
            const colors = []; const baseHue = 200;
            for (let i = 0; i < count; i++) {
                const hue = (baseHue + (i * 360 / count)) % 360;
                colors.push({ fill: `hsla(${hue}, 70%, 60%, 0.4)`, stroke: `hsla(${hue}, 70%, 60%, 1)` });
            } return colors;
        }
        function formatTimestamp(isoTimestamp) {
            if (!isoTimestamp) return 'N/A'; try { const date = new Date(isoTimestamp); const options = { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }; return date.toLocaleString('de-DE', options); } catch (e) { console.error("Error formatting timestamp:", e); return isoTimestamp; }
        }

        // Globale Variablen für Chart-Instanzen und Daten
        let barChartInstance = null;
        let radarChartInstance = null;
        let currentModelsData = {}; // Store fetched data globally
        let availableCategories = []; // Store available categories

        // Funktion zum Anzeigen von Statusmeldungen (wie vorher)
        function showStatusMessage(message, type = 'info') {
            const statusDiv = document.getElementById('eval-status');
            if (!statusDiv) return;
            statusDiv.textContent = ''; statusDiv.style.display = 'block'; statusDiv.className = 'status-message';

            if (type === 'loading') {
                statusDiv.classList.add('status-loading');
                const spinner = document.createElement('div'); spinner.className = 'loading-spinner'; statusDiv.appendChild(spinner);
                const textNode = document.createTextNode(message); statusDiv.appendChild(textNode);
            } else {
                 statusDiv.textContent = message;
                 if (type === 'success') statusDiv.classList.add('status-success');
                 else if (type === 'error') statusDiv.classList.add('status-error');
                 setTimeout(() => { statusDiv.style.display = 'none'; }, 5000);
            }
        }

        // Funktion zum Erstellen/Aktualisieren des Balkendiagramms für eine Kategorie
        function updateBarChart(selectedCategory) {
            console.log(`Updating Bar Chart for category: ${selectedCategory}`);
            const barCtx = document.getElementById('barChart')?.getContext('2d');
            if (!barCtx) { console.error("Canvas für Balkendiagramm nicht gefunden."); return; }

            // Destroy existing chart
            if (barChartInstance) barChartInstance.destroy();

            const labels = [];
            const ratings = [];

            // Prepare data for the selected category
            Object.entries(currentModelsData).forEach(([modelName, modelDetails]) => {
                const categoryRating = modelDetails.categorie?.[selectedCategory]; // Access rating for the selected category
                if (categoryRating !== undefined && typeof categoryRating === 'number') {
                    labels.push(modelName);
                    ratings.push(categoryRating);
                } else {
                    // Optionally include models with 0 or default rating if they exist but lack this category
                    // labels.push(modelName);
                    // ratings.push(0); // Or some default value
                    console.log(`Model ${modelName} has no rating for category ${selectedCategory}`);
                }
            });

            if (labels.length === 0) {
                console.log(`Keine Daten für Kategorie ${selectedCategory} im Balkendiagramm verfügbar.`);
                 // Optional: Display a message on the canvas
                 barCtx.clearRect(0, 0, barCtx.canvas.width, barCtx.canvas.height);
                 barCtx.textAlign = 'center';
                 barCtx.fillText(`Keine Daten für Kategorie '${selectedCategory}'`, barCtx.canvas.width / 2, barCtx.canvas.height / 2);
                return;
            }

            const barBgColors = labels.map((_, i) => generateColors(labels.length)[i].fill);
            const barBorderColors = labels.map((_, i) => generateColors(labels.length)[i].stroke);

            barChartInstance = new Chart(barCtx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: `ELO Bewertung (${selectedCategory})`,
                        data: ratings,
                        backgroundColor: barBgColors,
                        borderColor: barBorderColors,
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: { y: { beginAtZero: false, title: { display: true, text: 'ELO Bewertung' } }, x: { title: { display: true, text: 'Modell' } } },
                    plugins: { legend: { display: false }, tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label || ''}: ${ctx.parsed.y !== null ? ctx.parsed.y.toFixed(2) : ''}` } } }
                }
             });
        }

        // Funktion zum Erstellen/Aktualisieren des Radar-Diagramms für eine Kategorie
        function updateRadarChart(selectedCategory) {
             console.log(`Updating Radar Chart for category: ${selectedCategory}`);
             const radarCtx = document.getElementById('radarChart')?.getContext('2d');
             if (!radarCtx) { console.error("Canvas für Radar-Diagramm nicht gefunden."); return; }

             // Destroy existing chart
             if (radarChartInstance) radarChartInstance.destroy();

             const labels = [];
             const ratings = [];

             // Prepare data for the selected category
             Object.entries(currentModelsData).forEach(([modelName, modelDetails]) => {
                 const categoryRating = modelDetails.categorie?.[selectedCategory];
                 if (categoryRating !== undefined && typeof categoryRating === 'number') {
                     labels.push(modelName);
                     ratings.push(categoryRating);
                 } else {
                      console.log(`Model ${modelName} has no rating for category ${selectedCategory}`);
                 }
             });

             if (labels.length < 3) { // Radar chart needs at least 3 points
                 console.log(`Nicht genügend Daten (braucht mind. 3) für Kategorie ${selectedCategory} im Radar-Diagramm.`);
                 // Optional: Display a message on the canvas
                 radarCtx.clearRect(0, 0, radarCtx.canvas.width, radarCtx.canvas.height);
                 radarCtx.textAlign = 'center';
                 radarCtx.fillText(`Nicht genügend Daten für '${selectedCategory}'`, radarCtx.canvas.width / 2, radarCtx.canvas.height / 2);
                 return;
             }

             const radarColors = generateColors(1)[0]; // One color for the dataset

             radarChartInstance = new Chart(radarCtx, {
                 type: 'radar',
                 data: {
                     labels: labels,
                     datasets: [{
                         label: `ELO Bewertung (${selectedCategory})`,
                         data: ratings,
                         backgroundColor: radarColors.fill,
                         borderColor: radarColors.stroke,
                         pointBackgroundColor: radarColors.stroke,
                         pointBorderColor: '#fff',
                         pointHoverBackgroundColor: '#fff',
                         pointHoverBorderColor: radarColors.stroke,
                         borderWidth: 2
                     }]
                 },
                 options: {
                     responsive: true, maintainAspectRatio: false,
                     scales: { r: { beginAtZero: false, angleLines: { display: true }, suggestedMin: Math.min(...ratings) - 20, suggestedMax: Math.max(...ratings) + 20, pointLabels: { font: { size: 10 } }, ticks: { backdropColor: 'rgba(255, 255, 255, 0.75)', stepSize: 50 } } },
                     plugins: { legend: { position: 'top' }, tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label || ''}: ${ctx.parsed.r !== null ? ctx.parsed.r.toFixed(2) : ''}` } } },
                     elements: { line: { borderWidth: 3 } }
                 }
             });
         }


        // Funktion zum Abrufen der Daten und Aktualisieren des Dashboards
        async function fetchDataAndUpdateDashboard() {
            console.log("Fetching data from /api/results...");
            const dashboardContent = document.getElementById('dashboard-content');
            const loadingIndicator = document.getElementById('loading-indicator');
            if (dashboardContent.classList.contains('hidden')) {
                 loadingIndicator.style.display = 'block';
            }

            try {
                const response = await fetch('/api/results');
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                const jsonData = await response.json();
                console.log("Data received:", jsonData);

                loadingIndicator.style.display = 'none';
                dashboardContent.classList.remove('hidden');

                currentModelsData = jsonData.models || {}; // Store globally
                const timestamp = jsonData.timestamp;

                const timestampElement = document.getElementById('data-timestamp');
                if (timestampElement) timestampElement.textContent = formatTimestamp(timestamp);

                if (Object.keys(currentModelsData).length === 0) {
                    console.warn("Keine Modelldaten gefunden.");
                    document.getElementById('ratings-table-body').innerHTML = '<tr><td colspan="3" class="px-6 py-4 text-sm text-gray-500 text-center">Keine Daten verfügbar.</td></tr>';
                    if (barChartInstance) barChartInstance.destroy();
                    if (radarChartInstance) radarChartInstance.destroy();
                    // Clear dropdowns as well
                    document.getElementById('category-select-bar').innerHTML = '';
                    document.getElementById('category-select-radar').innerHTML = '';
                    return;
                }

                // --- Populate Table ---
                const tableBody = document.getElementById('ratings-table-body');
                tableBody.innerHTML = '';
                availableCategories = new Set(); // Use a Set to store unique categories

                Object.entries(currentModelsData).forEach(([modelName, modelDetails]) => {
                    const categories = modelDetails.categorie || {};
                    if (Object.keys(categories).length === 0) {
                        tableBody.innerHTML += `<tr><td class="px-6 py-4 text-sm font-medium text-gray-900">${modelName}</td><td class="px-6 py-4 text-sm text-gray-500">-</td><td class="px-6 py-4 text-sm text-gray-500">-</td></tr>`;
                    } else {
                        Object.entries(categories).forEach(([categoryName, rating]) => {
                             availableCategories.add(categoryName); // Add category to the set
                             const ratingFormatted = typeof rating === 'number' ? rating.toFixed(2) : 'N/A';
                             tableBody.innerHTML += `<tr><td class="px-6 py-4 text-sm font-medium text-gray-900">${modelName}</td><td class="px-6 py-4 text-sm text-gray-500">${categoryName}</td><td class="px-6 py-4 text-sm text-gray-500">${ratingFormatted}</td></tr>`;
                        });
                    }
                });

                availableCategories = Array.from(availableCategories).sort(); // Convert set to sorted array

                // --- Populate Dropdowns ---
                const categorySelectBar = document.getElementById('category-select-bar');
                const categorySelectRadar = document.getElementById('category-select-radar');
                categorySelectBar.innerHTML = ''; // Clear existing options
                categorySelectRadar.innerHTML = '';

                if (availableCategories.length > 0) {
                    availableCategories.forEach(category => {
                        const optionBar = document.createElement('option');
                        optionBar.value = category;
                        optionBar.textContent = category.charAt(0).toUpperCase() + category.slice(1); // Capitalize
                        categorySelectBar.appendChild(optionBar);

                        const optionRadar = document.createElement('option');
                        optionRadar.value = category;
                        optionRadar.textContent = category.charAt(0).toUpperCase() + category.slice(1); // Capitalize
                        categorySelectRadar.appendChild(optionRadar);
                    });

                    // Initial chart rendering with the first category
                    updateBarChart(availableCategories[0]);
                    updateRadarChart(availableCategories[0]);

                } else {
                    // Handle case with no categories found
                     categorySelectBar.innerHTML = '<option value="">Keine Kategorien</option>';
                     categorySelectRadar.innerHTML = '<option value="">Keine Kategorien</option>';
                     if (barChartInstance) barChartInstance.destroy();
                     if (radarChartInstance) radarChartInstance.destroy();
                     // Optionally display message on canvas
                     const barCtx = document.getElementById('barChart')?.getContext('2d');
                     if(barCtx) {
                         barCtx.clearRect(0, 0, barCtx.canvas.width, barCtx.canvas.height);
                         barCtx.textAlign = 'center';
                         barCtx.fillText(`Keine Kategoriedaten verfügbar`, barCtx.canvas.width / 2, barCtx.canvas.height / 2);
                     }
                     const radarCtx = document.getElementById('radarChart')?.getContext('2d');
                      if(radarCtx) {
                         radarCtx.clearRect(0, 0, radarCtx.canvas.width, radarCtx.canvas.height);
                         radarCtx.textAlign = 'center';
                         radarCtx.fillText(`Keine Kategoriedaten verfügbar`, radarCtx.canvas.width / 2, radarCtx.canvas.height / 2);
                     }
                }


            } catch (error) {
                console.error('Fehler beim Laden oder Verarbeiten der Dashboard-Daten:', error);
                 loadingIndicator.style.display = 'block';
                 loadingIndicator.innerHTML = '<p class="text-red-500">Fehler beim Laden der Dashboard-Daten.</p>';
                 dashboardContent.classList.add('hidden');
                if (barChartInstance) barChartInstance.destroy();
                if (radarChartInstance) radarChartInstance.destroy();
                // Clear dropdowns on error too
                document.getElementById('category-select-bar').innerHTML = '';
                document.getElementById('category-select-radar').innerHTML = '';
            }
        }

        // Event Listener für Dropdown-Änderungen
        const categorySelectBar = document.getElementById('category-select-bar');
        if (categorySelectBar) {
            categorySelectBar.addEventListener('change', (event) => {
                updateBarChart(event.target.value);
                // Optional: Sync radar chart dropdown or update radar chart too
                 const categorySelectRadar = document.getElementById('category-select-radar');
                 categorySelectRadar.value = event.target.value; // Sync selection
                 updateRadarChart(event.target.value);
            });
        }
         const categorySelectRadar = document.getElementById('category-select-radar');
         if (categorySelectRadar) {
             categorySelectRadar.addEventListener('change', (event) => {
                 updateRadarChart(event.target.value);
                 // Optional: Sync bar chart dropdown or update bar chart too
                 const categorySelectBar = document.getElementById('category-select-bar');
                 categorySelectBar.value = event.target.value; // Sync selection
                 updateBarChart(event.target.value);
             });
         }

        // Event Listener für den Button "Neuen Durchlauf starten" (wie vorher)
        const runEvalButton = document.getElementById('run-eval-button');
        if (runEvalButton) {
            runEvalButton.addEventListener('click', async () => {
                runEvalButton.disabled = true;
                showStatusMessage('Starte neuen Bewertungsdurchlauf...', 'loading');
                try {
                    const response = await fetch('/api/run_evaluation', { method: 'POST', headers: {'Content-Type': 'application/json'} });
                    const result = await response.json();
                    if (response.ok) {
                        showStatusMessage(result.message || 'Durchlauf erfolgreich gestartet/beendet.', 'success');
                        setTimeout(fetchDataAndUpdateDashboard, 1000); // Refresh data after a delay
                    } else {
                        throw new Error(result.error || `Serverfehler: ${response.status}`);
                    }
                } catch (error) {
                    console.error('Fehler beim Starten des Durchlaufs:', error);
                    showStatusMessage(`Fehler: ${error.message}`, 'error');
                } finally {
                     runEvalButton.disabled = false;
                }
            });
        } else {
            console.error("Button 'run-eval-button' nicht gefunden.");
        }


        // Initiales Laden der Daten
        document.addEventListener('DOMContentLoaded', fetchDataAndUpdateDashboard);

    </script>
</body>
</html>
"""

@app.route('/')
@app.route('/dashboard')
def dashboard():
    logger.info("Anfrage für /dashboard erhalten.")
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/results')
def api_results():
    logger.info(f"Anfrage für /api/results erhalten. Lese Datei: {RESULTS_FILE_PATH}")
    data = load_current_json(RESULTS_FILE_PATH, LOCK_FILE_PATH)
    if data:
        # Make sure the structure is consistent, especially the 'categorie' key
        # Optional: Add validation/migration logic here if needed
        return jsonify(data)
    else:
        logger.warning(f"Konnte Daten aus {RESULTS_FILE_PATH} nicht laden oder Datei ist leer/ungültig.")
        # Return a default structure even on error to prevent JS errors
        return jsonify({"models": {}, "timestamp": None})

# --- Background Task Logic (run_evaluation_background) ---
# (Unchanged from previous version - keep the existing logic)
def run_evaluation_background():
    global evaluation_running
    logger.info("Starte Bewertungsdurchlauf im Hintergrund...")
    try:
        config = load_config(CONFIG_PATH)
        num_comparisons = config.get("comparison", {}).get("comparisons_per_run", 1)
        logger.info(f"Starting synchronous evaluation for {num_comparisons} comparisons.")

        errors = []
        success_count = 0

        for i in range(num_comparisons):
            logger.info(f"Running comparison {i+1}/{num_comparisons}...")
            try:
                updated_ratings = run_comparison_cycle(config)
                if updated_ratings is None:
                    logger.warning(f"Comparison cycle {i+1} did not complete successfully or returned None.")
                    errors.append(f"Comparison cycle {i+1} failed.")
                else:
                    success_count += 1
                    logger.info(f"Comparison cycle {i+1} completed.")

            except Exception as e:
                logger.error(f"Unhandled exception in comparison cycle {i+1}: {e}", exc_info=True)
                errors.append(f"Error during comparison {i+1}: {e}")
                # Decide if you want to stop on first error or continue
                # break # Uncomment to stop on first error

        logger.info(f"Synchronous evaluation finished. Successful cycles: {success_count}/{num_comparisons}.")
        if errors:
             logger.error(f"Errors occurred during evaluation: {errors}")

    except Exception as e:
        logger.error(f"Fehler während des Bewertungsdurchlaufs im Hintergrund: {e}", exc_info=True)
    finally:
        evaluation_running = False
        logger.info("Hintergrund-Thread für Bewertung beendet.")


# --- API Endpoint to Trigger Evaluation (trigger_evaluation) ---
# (Unchanged from previous version - keep the existing logic)
@app.route('/api/run_evaluation', methods=['POST'])
def trigger_evaluation():
    global evaluation_running, evaluation_thread
    logger.info("Anfrage für /api/run_evaluation (POST) erhalten.")

    if evaluation_running and evaluation_thread and evaluation_thread.is_alive():
         logger.warning("Anfrage zum Starten der Bewertung erhalten, aber ein Durchlauf läuft bereits.")
         return jsonify({"error": "Ein Bewertungsdurchlauf läuft bereits."}), 409 # 409 Conflict

    logger.info("Starte neuen Bewertungsdurchlauf...")
    evaluation_running = True

    # Start the background task in a separate thread
    evaluation_thread = threading.Thread(target=run_evaluation_background, daemon=True)
    evaluation_thread.start()
    logger.info(f"Bewertungsdurchlauf in Thread {evaluation_thread.ident} gestartet.")

    # Return immediately, indicating the task has been accepted
    return jsonify({"message": "Bewertungsdurchlauf gestartet."}), 202 # 202 Accepted


# --- Server starten ---
if __name__ == '__main__':
    logger.info("Starte Flask Development Server für das Dashboard...")
    # Use threaded=True for development if background tasks need context
    # For production, use a proper WSGI server like Gunicorn or Waitress
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
