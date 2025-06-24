// static/js/dashboard.js

console.log("Dashboard script loaded successfully.");

// --- API Endpoints & Global State ---
const API_INITIAL_RESULTS_URL = '/api/results';
const API_RESULTS_STREAM_URL = '/stream';
const API_RUN_EVAL_URL = '/api/run_evaluation';
const API_EVAL_STATUS_URL = '/api/evaluation_status';

let barChartInstance, radarChartInstance, comparisonChartInstance = null;
let currentModelsData = {}, availableCategories = [], eventSource = null;
const ALL_CATEGORIES_KEY = "__all__";

// --- Utility Functions ---
function generateColors(count) {
    const colors = [];
    const baseHues = [210, 30, 160, 260, 340, 190, 50, 280, 110, 300];
    for (let i = 0; i < count; i++) {
        const hue = baseHues[i % baseHues.length] + Math.floor(i / baseHues.length) * 35;
        colors.push({
            fill: `hsla(${hue % 360}, 80%, 65%, 0.7)`,
            stroke: `hsla(${hue % 360}, 80%, 55%, 1)`
        });
    }
    return colors;
}

function capitalizeFirstLetter(s) {
    return s ? s.charAt(0).toUpperCase() + s.slice(1) : '';
}

// --- UI Update Functions ---
function showStatusMessage(message) {
    document.getElementById('eval-status').textContent = message;
}

function updateEvaluationButton(isLoading, message = null) {
    const runButton = document.getElementById('run-eval-button');
    if (!runButton) return;
    const buttonTextEl = document.getElementById('run-eval-button-text');
    const spinner = runButton.querySelector('.loading-spinner');
    const dynamicTextEl = runButton.querySelector('.dynamic-text');
    runButton.disabled = isLoading;
    spinner.classList.toggle('hidden', !isLoading);
    buttonTextEl.classList.toggle('hidden', isLoading);
    if (isLoading) {
        dynamicTextEl.textContent = message || 'Wird ausgeführt...';
        dynamicTextEl.classList.remove('hidden');
    } else {
        dynamicTextEl.classList.add('hidden');
    }
}

function updateBarChart(selectedCategory) {
    const barCtx = document.getElementById('barChart')?.getContext('2d');
    if (!barCtx) return;
    if (barChartInstance) barChartInstance.destroy();

    const data = { labels: [], ratings: [] };
    if (selectedCategory === ALL_CATEGORIES_KEY) {
        Object.entries(currentModelsData).forEach(([model, details]) => {
            const cats = details.elo_rating_by_category || {};
            const ratingValues = Object.values(cats).filter(r => typeof r === 'number');
            if (ratingValues.length > 0) {
                data.labels.push(model);
                data.ratings.push(ratingValues.reduce((s, r) => s + r, 0) / ratingValues.length);
            }
        });
    } else {
        Object.entries(currentModelsData).forEach(([model, details]) => {
            const rating = details.elo_rating_by_category?.[selectedCategory];
            if (typeof rating === 'number') { data.labels.push(model); data.ratings.push(rating); }
        });
    }
    const sortedIndices = data.ratings.map((_, i) => i).sort((a, b) => data.ratings[b] - data.ratings[a]);
    const gradient = barCtx.createLinearGradient(0, 0, 0, 380);
    gradient.addColorStop(0, 'hsla(210, 90%, 55%, 0.8)');
    gradient.addColorStop(1, 'hsla(210, 90%, 75%, 0.3)');

    barChartInstance = new Chart(barCtx, {
        type: 'bar',
        data: {
            labels: sortedIndices.map(i => data.labels[i]),
            datasets: [{ label: `ELO Rating`, data: sortedIndices.map(i => data.ratings[i]), backgroundColor: gradient, borderColor: 'hsl(210, 90%, 55%)', borderWidth: 2, borderRadius: 6, barThickness: 'flex', maxBarThickness: 45 }]
        },
        options: { responsive: true, maintainAspectRatio: false, scales: { x: { grid: { display: false }, ticks: { color: 'hsl(220, 10%, 50%)' } }, y: { grid: { color: 'hsl(220, 20%, 92%)' }, ticks: { color: 'hsl(220, 10%, 50%)' } } }, plugins: { legend: { display: false }, tooltip: { backgroundColor: 'hsl(225, 15%, 12%)', titleFont: { weight: 'bold'}, bodyFont: { size: 13 }, padding: 12, cornerRadius: 8, titleColor: 'white', bodyColor: 'white' } } }
    });
}

function updateRadarChart() {
    const radarCtx = document.getElementById('radarChart')?.getContext('2d');
    if (!radarCtx) return; if (radarChartInstance) radarChartInstance.destroy();
    const modelNames = Object.keys(currentModelsData).sort();
    const radarLabels = availableCategories.map(cat => capitalizeFirstLetter(cat));
    if (modelNames.length === 0 || radarLabels.length < 3) return;
    const radarChartColors = generateColors(modelNames.length);
    const datasets = modelNames.map((modelName, index) => ({
        label: modelName.length > 20 ? modelName.substring(0,17) + '...' : modelName,
        data: availableCategories.map(cat => currentModelsData[modelName]?.elo_rating_by_category?.[cat] ?? null),
        backgroundColor: radarChartColors[index % radarChartColors.length].fill,
        borderColor: radarChartColors[index % radarChartColors.length].stroke,
        borderWidth: 2, pointRadius: 4, pointBackgroundColor: radarChartColors[index % radarChartColors.length].stroke
    })).filter(ds => ds.data.some(d => d !== null));
    if (datasets.length === 0) return;
    radarChartInstance = new Chart(radarCtx, {
        type: 'radar',
        data: { labels: radarLabels, datasets: datasets },
        options: { responsive: true, maintainAspectRatio: false, scales: { r: { angleLines: { color: 'hsl(220, 20%, 92%)' }, pointLabels: { font: { size: 12, weight: '500' }, color: 'hsl(220, 10%, 40%)' }, ticks: { backdropColor: 'hsla(0, 0%, 100%, 0.75)', font: {size: 10}, color: 'hsl(220, 10%, 50%)' }, grid: { color: 'hsl(220, 20%, 92%)' } } }, plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, padding: 15, font: {size: 11}, color: 'hsl(220, 10%, 50%)' } }, tooltip: { backgroundColor: 'hsl(225, 15%, 12%)', titleFont: { weight: 'bold'}, bodyFont: { size: 13 }, padding: 12, cornerRadius: 8, titleColor: 'white', bodyColor: 'white' } } }
    });
}

function populateOverallRankings(modelsData) {
    const tableBody = document.getElementById('overall-rankings-table-body');
    if (!tableBody) return; tableBody.innerHTML = '';
    const modelStats = [];
    Object.entries(modelsData).forEach(([modelName, details]) => {
        const cats = details.elo_rating_by_category || {};
        let highestRating = -Infinity, highestCategory = 'N/A', totalRating = 0, catCount = 0;
        Object.entries(cats).forEach(([cat, rating]) => {
            if(typeof rating === 'number') {
                if (rating > highestRating) { highestRating = rating; highestCategory = cat; }
                totalRating += rating; catCount++;
            }
        });
        modelStats.push({ name: modelName, highestCategory, highestRating: highestRating === -Infinity ? 0 : highestRating, avgRating: catCount > 0 ? totalRating / catCount : 0, categoryCount: catCount, ...details });
    });
    modelStats.sort((a, b) => b.avgRating - a.avgRating);
    if (modelStats.length === 0) { tableBody.innerHTML = '<tr><td colspan="9" class="text-center py-12 text-[var(--text-secondary)]">Keine Daten vorhanden. Führen Sie eine Evaluierung durch.</td></tr>'; return; }
    modelStats.forEach((stats, index) => {
        const rank = index + 1; let rankClass = 'rank-default';
        if (rank === 1) rankClass = 'rank-1'; else if (rank === 2) rankClass = 'rank-2'; else if (rank === 3) rankClass = 'rank-3';
        tableBody.innerHTML += `<tr class="hover:bg-slate-50 transition-colors duration-150"><td class="px-6 py-4 text-center"><span class="rank-badge ${rankClass}">${rank}</span></td><td class="px-6 py-4 text-sm font-semibold text-slate-800">${stats.name}</td><td class="px-6 py-4 text-sm text-slate-600">${capitalizeFirstLetter(stats.highestCategory)}</td><td class="px-6 py-4 text-sm font-medium text-right">${stats.highestRating.toFixed(1)}</td><td class="px-6 py-4 text-sm text-[var(--accent-primary-text)] font-bold text-right">${stats.avgRating.toFixed(1)}</td><td class="px-6 py-4 text-sm text-right">${stats.num_comparisons || 0}</td><td class="px-6 py-4 text-sm text-green-600 font-medium text-right">${stats.wins || 0}</td><td class="px-6 py-4 text-sm text-red-600 font-medium text-right">${stats.losses || 0}</td><td class="px-6 py-4 text-sm text-right">${stats.draws || 0}</td></tr>`;
    });
}

function populateDetailedTable(modelsData, categories) {
    const tableHead = document.getElementById('detailed-ratings-thead'), tableBody = document.getElementById('detailed-ratings-tbody');
    if (!tableHead || !tableBody) return;
    tableHead.innerHTML = `<tr><th class="px-6 py-4 text-left text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider sticky left-0 bg-slate-50 z-10 border-b border-[var(--border-color)]">Modell</th>${categories.map(c => `<th class="px-6 py-4 text-right text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider border-b border-[var(--border-color)]">${capitalizeFirstLetter(c)}</th>`).join('')}</tr>`;
    const modelNames = Object.keys(modelsData).sort();
    if(modelNames.length === 0) { tableBody.innerHTML = ''; return; }
    const bestInCategory = {};
    categories.forEach((cat, index) => {
        let maxRating = -Infinity;
        modelNames.forEach(modelName => {
            const rating = modelsData[modelName]?.elo_rating_by_category?.[cat];
            if (typeof rating === 'number' && rating > maxRating) maxRating = rating;
        });
        bestInCategory[cat] = maxRating;
    });
    tableBody.innerHTML = modelNames.map(modelName => {
        const details = modelsData[modelName];
        let rowHtml = `<tr class="group hover:bg-slate-50 transition-colors duration-150"><td class="px-6 py-4 text-sm font-semibold text-slate-800 sticky left-0 bg-white group-hover:bg-slate-50 z-10">${modelName}</td>`;
        rowHtml += categories.map(cat => {
            const rating = details.elo_rating_by_category?.[cat];
            const isBest = (typeof rating === 'number' && rating === bestInCategory[cat] && rating > 0);
            const display = typeof rating === 'number' ? rating.toFixed(1) : '<span class="text-slate-400">N/A</span>';
            const counts = details.comparison_counts_by_category?.[cat];
            const tip = counts ? `S:${counts.wins||0} | N:${counts.losses||0} | U:${counts.draws||0}` : 'Keine Daten';
            return `<td class="px-6 py-4 text-sm text-right ${isBest ? 'best-in-category' : ''}" title="${tip}">${display}</td>`;
        }).join('');
        return rowHtml + '</tr>';
    }).join('');
}

function updateMetricCards(modelsData) {
    const models = Object.values(modelsData);
    document.getElementById('metric-total-comparisons').textContent = new Intl.NumberFormat('de-DE').format(models.reduce((s, m) => s + (m.num_comparisons || 0), 0));
    document.getElementById('metric-model-count').textContent = models.length;
    let topModel = "N/A";
    if (models.length > 0) {
        const sortedByElo = Object.entries(modelsData).map(([name, m]) => {
            const ratingValues = Object.values(m.elo_rating_by_category || {}).filter(r => typeof r === 'number');
            return { name, avgElo: ratingValues.length > 0 ? ratingValues.reduce((s, r) => s + r, 0) / ratingValues.length : 0 };
        }).sort((a,b) => b.avgElo - a.avgElo);
        if(sortedByElo.length > 0 && sortedByElo[0].avgElo > 0) topModel = sortedByElo[0].name;
    }
    document.getElementById('metric-top-model').textContent = topModel.length > 20 ? topModel.substring(0,17) + '...' : topModel;
}

function populateSelect(selectEl, options, currentValue, defaultValue, defaultText) {
    if (!selectEl) return;
    const currentVal = selectEl.value;
    selectEl.innerHTML = `<option value="${defaultValue}">${defaultText}</option>` + options.map(o => `<option value="${o.value}">${o.text}</option>`).join('');
    selectEl.value = Array.from(selectEl.options).some(opt => opt.value === currentVal) ? currentVal : defaultValue;
}

function updateComparisonChart() {
    const model1 = document.getElementById('model-select-1').value;
    const model2 = document.getElementById('model-select-2').value;
    const container = document.getElementById('comparison-chart-container');
    const noDataEl = document.getElementById('comparison-no-data');
    const ctx = document.getElementById('comparisonChart')?.getContext('2d');
    if (!ctx) return;
    if (comparisonChartInstance) comparisonChartInstance.destroy();

    if (!model1 || !model2 || model1 === model2) {
        container.classList.add('hidden');
        noDataEl.classList.remove('hidden');
        noDataEl.textContent = !model1 || !model2 ? "Bitte zwei Modelle auswählen." : "Bitte zwei verschiedene Modelle auswählen.";
        return;
    }
    container.classList.remove('hidden');
    noDataEl.classList.add('hidden');
    const data1 = currentModelsData[model1], data2 = currentModelsData[model2];
    const commonCategories = availableCategories.filter(c => data1.elo_rating_by_category?.[c] && data2.elo_rating_by_category?.[c]);
    if (commonCategories.length === 0) {
         container.classList.add('hidden'); noDataEl.classList.remove('hidden');
         noDataEl.textContent = 'Keine gemeinsamen Kategorien für einen Vergleich gefunden.';
         return;
    }
    const colors = generateColors(2);
    comparisonChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: commonCategories.map(c => capitalizeFirstLetter(c)),
            datasets: [
                { label: model1, data: commonCategories.map(c => data1.elo_rating_by_category[c]), backgroundColor: colors[0].fill, borderColor: colors[0].stroke, borderWidth: 1.5, borderRadius: 5 },
                { label: model2, data: commonCategories.map(c => data2.elo_rating_by_category[c]), backgroundColor: colors[1].fill, borderColor: colors[1].stroke, borderWidth: 1.5, borderRadius: 5 }
            ]
        },
        options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: false, grid: { color: 'hsl(220, 20%, 92%)' }, ticks: { color: 'hsl(220, 10%, 50%)' } }, x: { grid: { display: false }, ticks: { color: 'hsl(220, 10%, 50%)' } } }, plugins: { legend: { position: 'top', labels: { color: 'hsl(220, 10%, 50%)' } }, tooltip: { backgroundColor: 'hsl(225, 15%, 12%)', titleFont: { weight: 'bold'}, bodyFont: { size: 13 }, padding: 12, cornerRadius: 8, titleColor: 'white', bodyColor: 'white' } } }
    });
}

function exportTableToCSV(tableId, filename) {
    let csv = [];
    const rows = document.querySelectorAll(`#${tableId} tr`);
    for (const row of rows) {
        let row_data = [];
        const cols = row.querySelectorAll('td, th');
        for (const col of cols) {
            let data = '"' + col.innerText.replace(/"/g, '""') + '"';
            row_data.push(data);
        }
        csv.push(row_data.join(','));
    }
    const csv_string = csv.join('\n');
    const data_uri = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv_string);
    const link = document.createElement('a');
    link.setAttribute('href', data_uri);
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function updateAllUI(data) {
    currentModelsData = data.models || {};
    const categories = new Set();
    Object.values(currentModelsData).forEach(d => Object.keys(d.elo_rating_by_category || {}).forEach(c => categories.add(c)));
    availableCategories = Array.from(categories).sort();

    updateMetricCards(currentModelsData);
    populateOverallRankings(currentModelsData);
    populateDetailedTable(currentModelsData, availableCategories);

    const categoryOptions = availableCategories.map(c => ({value: c, text: capitalizeFirstLetter(c)}));
    populateSelect(document.getElementById('eval-category-select'), categoryOptions, '', ALL_CATEGORIES_KEY, 'Alle Kategorien');
    populateSelect(document.getElementById('category-select-bar'), categoryOptions, '', ALL_CATEGORIES_KEY, 'Alle Kategorien (Ø)');

    const modelOptions = Object.keys(currentModelsData).sort().map(m => ({value: m, text: m}));
    populateSelect(document.getElementById('model-select-1'), modelOptions, '', '', 'Modell 1 auswählen...');
    populateSelect(document.getElementById('model-select-2'), modelOptions, '', '', 'Modell 2 auswählen...');

    updateBarChart(document.getElementById('category-select-bar').value || ALL_CATEGORIES_KEY);
    updateRadarChart();
    updateComparisonChart();

    document.getElementById('loading-indicator').classList.add('hidden');
    document.getElementById('dashboard-content').classList.remove('hidden');
}

// --- Main Execution Logic & Event Handlers ---
document.addEventListener('DOMContentLoaded', () => {
    fetch(API_INITIAL_RESULTS_URL)
        .then(res => { if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`); return res.json(); })
        .then(updateAllUI)
        .catch(err => {
            console.error('Initial data load failed:', err);
            document.getElementById('loading-indicator').innerHTML = `<div class="text-center p-6 bg-red-50 text-red-800 rounded-lg border border-red-200"><p class="font-semibold text-lg">Fehler beim Laden der Daten</p><p class="text-sm mt-2">Konnte keine Verbindung zum Backend herstellen. Bitte stellen Sie sicher, dass der Server läuft und erreichbar ist.</p></div>`;
        }).finally(() => {
            fetchEvaluationStatus();
            connectEventSource();
        });

    const navLinks = document.querySelectorAll('.nav-link');
    const tabContents = document.querySelectorAll('.tab-content');
    const mainHeaderTitle = document.getElementById('main-header-title');

    const handleTabClick = (e) => {
        e.preventDefault();
        const tabName = e.currentTarget.dataset.tab;

        navLinks.forEach(l => l.classList.remove('nav-link-active'));
        document.querySelectorAll(`.nav-link[data-tab="${tabName}"]`).forEach(l => l.classList.add('nav-link-active'));

        const headerText = e.currentTarget.textContent;
        mainHeaderTitle.textContent = headerText;
        const headerDescriptions = {
            'Übersicht': 'Ihr zentrales Cockpit für den Vergleich von LLM-Leistungsmetriken.',
            'Detail-Analyse': 'Eine detaillierte Aufschlüsselung der ELO-Werte für jedes Modell und jede Kategorie.',
            'Modellvergleich': 'Vergleichen Sie die Leistung von zwei Modellen direkt nebeneinander.'
        };
        mainHeaderTitle.nextElementSibling.textContent = headerDescriptions[headerText] || '';

        tabContents.forEach(c => c.classList.remove('active'));
        document.getElementById(`tab-content-${tabName}`).classList.add('active');

        const mobileMenu = document.getElementById('mobile-menu');
        if (!mobileMenu.classList.contains('hidden')) {
             mobileMenu.classList.add('hidden');
             document.getElementById('mobile-menu-open-icon').classList.remove('hidden');
             document.getElementById('mobile-menu-close-icon').classList.add('hidden');
        }
    };

    navLinks.forEach(link => link.addEventListener('click', handleTabClick));

    document.querySelector('.nav-link[data-tab="overview"]').click();

    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');
    const openIcon = document.getElementById('mobile-menu-open-icon');
    const closeIcon = document.getElementById('mobile-menu-close-icon');

    mobileMenuButton.addEventListener('click', () => {
        mobileMenu.classList.toggle('hidden');
        openIcon.classList.toggle('hidden');
        closeIcon.classList.toggle('hidden');
    });

    // Event Listeners for controls
    const generateOnlyCheckbox = document.getElementById('generate-only-checkbox');
    const judgeOnlyCheckbox = document.getElementById('judge-only-checkbox');
    generateOnlyCheckbox.addEventListener('change', () => { if (generateOnlyCheckbox.checked) judgeOnlyCheckbox.checked = false; });
    judgeOnlyCheckbox.addEventListener('change', () => { if (judgeOnlyCheckbox.checked) generateOnlyCheckbox.checked = false; });
    document.getElementById('run-eval-button').addEventListener('click', runEvaluation);
    document.getElementById('category-select-bar').addEventListener('change', e => updateBarChart(e.target.value));
    document.getElementById('model-select-1').addEventListener('change', updateComparisonChart);
    document.getElementById('model-select-2').addEventListener('change', updateComparisonChart);
    document.getElementById('export-overall').addEventListener('click', () => exportTableToCSV('overall-rankings-table', 'ollamabench-overall.csv'));
    document.getElementById('export-details').addEventListener('click', () => exportTableToCSV('detailed-ratings-table', 'ollamabench-details.csv'));
});

// --- Core Functions ---
function connectEventSource() {
    if (eventSource && eventSource.readyState !== EventSource.CLOSED) eventSource.close();
    eventSource = new EventSource(API_RESULTS_STREAM_URL);
    eventSource.onopen = () => console.log("SSE connected.");
    eventSource.onmessage = (event) => { try { updateAllUI(JSON.parse(event.data)); } catch(e) { console.error("SSE parse error", e); }};
    eventSource.onerror = () => { console.error("SSE connection failed. Retrying in 5s."); eventSource.close(); setTimeout(connectEventSource, 5000); };
}

async function fetchEvaluationStatus() {
    try {
        const res = await fetch(API_EVAL_STATUS_URL);
        if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
        const status = await res.json();
        updateEvaluationButton(status.running, status.message);
        showStatusMessage(status.message);
    } catch (e) {
        console.error("Fetching status failed:", e);
        showStatusMessage("Statusabruf fehlgeschlagen.");
        updateEvaluationButton(false);
    }
}

async function runEvaluation() {
    updateEvaluationButton(true, "Wird gestartet...");
    showStatusMessage("Sende Anfrage an den Server...");
    const selectedCategory = document.getElementById('eval-category-select').value;
    const runAllCategories = selectedCategory === ALL_CATEGORIES_KEY;
    const payload = {
        all_categories: runAllCategories,
        category: runAllCategories ? null : selectedCategory,
        generate_only: document.getElementById('generate-only-checkbox').checked,
        judge_only: document.getElementById('judge-only-checkbox').checked
    };
    try {
        const res = await fetch(API_RUN_EVAL_URL, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        const resText = await res.text();
        if (!res.ok) { throw new Error(`Serverantwort: ${res.status} ${resText}`); }
        const result = JSON.parse(resText);
        updateEvaluationButton(result.running, result.message);
        showStatusMessage(result.message);
    } catch (e) {
        console.error("Starting evaluation failed:", e);
        showStatusMessage(`Start der Evaluierung fehlgeschlagen.`);
        updateEvaluationButton(false);
    }
}