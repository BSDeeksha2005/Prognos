const VIEW_NAMES = ['overview', 'engines', 'comparison', 'models', 'quality'];
const SENSOR_NAMES = Array.from({ length: 21 }, (_, index) => `sensor_${index + 1}`);
const formatMetric = (value, digits = 1) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const formatRul = (value) => Number.isFinite(value) ? `${Math.round(value)} cycles` : '—';

let dashboard = null;
let engineCycles = new Map();
let activeWindowLength = 30;
let registrySort = { key: 'predicted_rul', direction: 'asc' };

function switchView(name, updateHistory = true) {
  if (!VIEW_NAMES.includes(name)) name = 'overview';
  document.querySelectorAll('.rail-link[role="tab"]').forEach((tab) => {
    const selected = tab.dataset.view === name;
    tab.classList.toggle('selected', selected);
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  document.querySelectorAll('.workspace-view').forEach((view) => {
    const selected = view.dataset.page === name;
    view.classList.toggle('active', selected);
    view.hidden = !selected;
  });
  if (updateHistory && location.hash !== `#${name}`) history.pushState(null, '', `#${name}`);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function setServiceStatus(state, label) {
  const status = document.querySelector('#apiState');
  status.classList.toggle('ready', state === 'ready');
  status.classList.toggle('offline', state === 'offline');
  status.querySelector('span').textContent = label;
}

function renderReadiness(data) {
  const files = data.required_files ?? [];
  const availableNames = new Set(files.filter((file) => file.available).map((file) => file.name));
  let availableCount = 0;
  document.querySelectorAll('.file-check').forEach((row) => {
    const available = availableNames.has(row.dataset.file);
    availableCount += Number(available);
    row.classList.toggle('available', available);
    row.querySelector('.file-state').textContent = available ? '✓' : '○';
    row.querySelector('small').textContent = available ? 'READY' : 'MISSING';
  });
  document.querySelectorAll('.quality-file').forEach((row) => {
    const available = availableNames.has(row.dataset.file);
    row.classList.toggle('available', available);
    row.querySelector('.quality-file-icon').textContent = available ? '✓' : '○';
    row.querySelector('small').textContent = available ? 'READY' : 'MISSING';
  });
  document.querySelector('#fileProgress').textContent = `${availableCount} / 3`;
  const banner = document.querySelector('#readinessBanner');
  banner.dataset.state = data.state;
  const title = document.querySelector('#readinessTitle');
  const message = document.querySelector('#readinessMessage');
  const detail = document.querySelector('#readinessDetail');
  const symbol = document.querySelector('#readinessSymbol');
  const datasetState = document.querySelector('#datasetState');

  const readiness = {
    evaluated: ['EVALUATION COMPLETE', 'FD001 test predictions are available for review.', 'Results are from a simulated research benchmark, not operational guidance.', 'EVALUATED', '✓'],
    model_ready: ['MODEL READY', 'The exported ONNX model is ready to score an engine window.', 'No held-out test report is available yet.', 'MODEL READY', '✓'],
    files_ready: ['DATASET READY', 'FD001 files are present. Run the pipeline to evaluate the models.', 'From the project root, run make train.', 'READY TO TRAIN', '✓'],
    awaiting_data: ['DATASET NOT LOADED', 'Add the three FD001 source files to begin evaluation.', 'Place them in data/raw/ at the Prognos project root.', 'DATA MISSING', '!'],
  };
  const [heading, summary, explanation, stateLabel, mark] = readiness[data.state] ?? readiness.awaiting_data;
  title.textContent = heading;
  message.textContent = summary;
  detail.textContent = explanation;
  symbol.querySelector('span').textContent = mark;
  symbol.classList.toggle('ready', data.state === 'evaluated');
  datasetState.classList.toggle('ready', data.state === 'evaluated' || data.state === 'files_ready');
  datasetState.querySelector('span').textContent = stateLabel;
  document.querySelector('#qualityFileState').textContent = data.files_present ? '3 / 3 READY' : `${availableCount} / 3 PRESENT`;
  document.querySelector('#qualityFileState').classList.toggle('complete', data.files_present);
}

function createCell(value, className = '') {
  const cell = document.createElement('td');
  cell.textContent = value;
  if (className) cell.className = className;
  return cell;
}

function appendEmptyRow(body, columns, message) {
  body.replaceChildren();
  const row = document.createElement('tr');
  const cell = createCell(message, 'table-empty');
  cell.colSpan = columns;
  row.append(cell);
  body.append(row);
}

function createEngineButton(engine) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'engine-link';
  button.textContent = `ENG-${String(engine.unit_id).padStart(3, '0')}`;
  button.addEventListener('click', () => showEngineDetail(engine));
  return button;
}

function appendEngineRow(body, engine, rank, includeError = false) {
  const row = document.createElement('tr');
  const unit = document.createElement('td');
  unit.append(createEngineButton(engine));
  row.append(unit);
  row.append(createCell(String(engine.last_cycle)));
  row.append(createCell(formatRul(engine.predicted_rul), 'rul-number'));
  row.append(createCell(formatRul(engine.actual_rul)));
  if (includeError) {
    const error = Number.isFinite(engine.actual_rul) ? engine.predicted_rul - engine.actual_rul : null;
    row.append(createCell(formatMetric(error), error > 0 ? 'error-positive' : error < 0 ? 'error-negative' : ''));
  }
  row.append(createCell(String(rank).padStart(2, '0'), 'rank-cell'));
  body.append(row);
}

function renderWatchlist(engines) {
  const body = document.querySelector('#watchRows');
  const priority = [...engines].sort((a, b) => a.predicted_rul - b.predicted_rul).slice(0, 5);
  if (!priority.length) return appendEmptyRow(body, 4, 'No test predictions available yet.');
  body.replaceChildren();
  priority.forEach((engine) => {
    const row = document.createElement('tr');
    const unit = document.createElement('td');
    unit.append(createEngineButton(engine));
    row.append(unit, createCell(String(engine.last_cycle)), createCell(formatRul(engine.predicted_rul), 'rul-number'), createCell(formatRul(engine.actual_rul)));
    body.append(row);
  });
}

function renderRegistry() {
  const body = document.querySelector('#registryRows');
  const all = dashboard?.engines ?? [];
  const search = document.querySelector('#engineSearch').value.trim().toLowerCase().replace(/^eng-?/, '');
  const filtered = all.filter((engine) => String(engine.unit_id).padStart(3, '0').includes(search) || String(engine.unit_id).includes(search));
  const { key, direction } = registrySort;
  filtered.sort((a, b) => {
    const first = a[key];
    const second = b[key];
    if (!Number.isFinite(first)) return 1;
    if (!Number.isFinite(second)) return -1;
    return (first - second) * direction;
  });
  document.querySelector('#registryCount').textContent = all.length ? String(all.length) : '—';
  document.querySelector('#registrySummary').textContent = all.length
    ? `${filtered.length} of ${all.length} engine records`
    : 'Waiting for evaluation data';
  if (!filtered.length) {
    return appendEmptyRow(body, 6, all.length ? 'No engine IDs match this search.' : 'Engine records appear after a completed test evaluation.');
  }
  body.replaceChildren();
  filtered.forEach((engine, index) => appendEngineRow(body, engine, index + 1, true));
}

function renderScatter(svgId, emptyId, yId, engines) {
  const svg = document.querySelector(`#${svgId}`);
  const empty = document.querySelector(`#${emptyId}`);
  const axis = document.querySelector(`#${yId}`);
  const comparable = (engines ?? []).filter((engine) => Number.isFinite(engine.actual_rul) && Number.isFinite(engine.predicted_rul));
  svg.replaceChildren();
  if (!comparable.length) {
    empty.hidden = false;
    return;
  }
  const maxRul = Math.ceil(Math.max(125, ...comparable.flatMap((engine) => [engine.actual_rul, engine.predicted_rul])) / 25) * 25;
  const xFor = (rul) => 20 + Math.max(0, Math.min(maxRul, rul)) / maxRul * 960;
  const yFor = (rul) => 10 + (maxRul - Math.max(0, Math.min(maxRul, rul))) / maxRul * 280;
  const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  line.setAttribute('class', 'identity-line');
  line.setAttribute('x1', String(xFor(0)));
  line.setAttribute('y1', String(yFor(0)));
  line.setAttribute('x2', String(xFor(maxRul)));
  line.setAttribute('y2', String(yFor(maxRul)));
  svg.append(line);
  comparable.forEach((engine) => {
    const point = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    point.setAttribute('class', 'point');
    point.setAttribute('cx', xFor(engine.actual_rul).toFixed(1));
    point.setAttribute('cy', yFor(engine.predicted_rul).toFixed(1));
    point.setAttribute('r', '5');
    const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    title.textContent = `Engine ${engine.unit_id}: reference ${formatMetric(engine.actual_rul)} cycles, estimate ${formatMetric(engine.predicted_rul)} cycles`;
    point.append(title);
    svg.append(point);
  });
  axis.replaceChildren(...Array.from({ length: 6 }, (_, index) => {
    const label = document.createElement('span');
    label.textContent = String(Math.round(maxRul * (1 - index / 5)));
    return label;
  }));
  const xLabels = svg.closest('.plot-area').querySelectorAll('.chart-x span');
  xLabels[0].textContent = '0';
  xLabels[2].textContent = String(maxRul);
  empty.hidden = true;
}

function renderComparison(engines, metrics) {
  const cnn = metrics?.cnn ?? {};
  const errors = engines.filter((engine) => Number.isFinite(engine.actual_rul) && Number.isFinite(engine.predicted_rul))
    .map((engine) => engine.predicted_rul - engine.actual_rul);
  const bias = errors.length ? errors.reduce((sum, error) => sum + error, 0) / errors.length : null;
  document.querySelector('#comparisonCount').textContent = errors.length ? String(errors.length) : '—';
  document.querySelector('#comparisonRmse').textContent = formatMetric(cnn.test_rmse);
  document.querySelector('#comparisonNasa').textContent = formatMetric(cnn.test_nasa);
  document.querySelector('#comparisonBias').textContent = Number.isFinite(bias) ? `${bias > 0 ? '+' : ''}${formatMetric(bias)}` : '—';
  document.querySelector('#comparisonBadge').textContent = errors.length ? 'TEST SCORED' : 'AWAITING TEST';
  document.querySelector('#comparisonBadge').classList.toggle('scored', Boolean(errors.length));
  renderScatter('comparisonPlot', 'comparisonPlotEmpty', 'comparisonY', engines);
  const residuals = engines.filter((engine) => Number.isFinite(engine.actual_rul) && Number.isFinite(engine.predicted_rul))
    .sort((a, b) => Math.abs(b.predicted_rul - b.actual_rul) - Math.abs(a.predicted_rul - a.actual_rul)).slice(0, 10);
  const body = document.querySelector('#residualRows');
  if (!residuals.length) return appendEmptyRow(body, 4, 'Residuals appear after test evaluation.');
  body.replaceChildren();
  residuals.forEach((engine) => {
    const error = engine.predicted_rul - engine.actual_rul;
    const row = document.createElement('tr');
    const unit = document.createElement('td');
    unit.append(createEngineButton(engine));
    row.append(unit, createCell(formatRul(engine.actual_rul)), createCell(formatRul(engine.predicted_rul), 'rul-number'), createCell(`${error > 0 ? '+' : ''}${formatMetric(error)}`, error > 0 ? 'error-positive' : error < 0 ? 'error-negative' : ''));
    body.append(row);
  });
}

function renderModelEvaluation(metrics, runMetadata) {
  const names = [
    ['linear', 'Linear regression'],
    ['lightgbm', 'LightGBM'],
    ['cnn', '1D convolutional network'],
  ];
  const body = document.querySelector('#modelRows');
  const hasScores = names.some(([key]) => metrics?.[key]);
  document.querySelector('#modelEvaluationState').textContent = hasScores ? 'TEST EVALUATION COMPLETE' : 'NO RUN RECORDED';
  if (!hasScores) appendEmptyRow(body, 5, 'No measured scores yet. Add FD001 files and run make train.');
  else {
    body.replaceChildren();
    names.forEach(([key, name]) => {
      const result = metrics[key];
      const row = document.createElement('tr');
      const status = document.createElement('td');
      const badge = document.createElement('span');
      badge.className = `drift-status ${result ? 'clear' : ''}`;
      badge.textContent = result ? 'SCORED' : 'NOT RUN';
      status.append(badge);
      row.append(createCell(name), createCell(formatMetric(result?.validation_rmse)), createCell(formatMetric(result?.test_rmse)), createCell(formatMetric(result?.test_nasa)), status);
      body.append(row);
    });
  }
  const experiments = runMetadata?.experiments ?? [];
  document.querySelector('#experimentCount').textContent = `${experiments.length} / 5`;
  const experimentBody = document.querySelector('#experimentRows');
  if (!experiments.length) return appendEmptyRow(experimentBody, 6, 'Experiment results are pending the real training run.');
  experimentBody.replaceChildren();
  experiments.forEach((experiment, index) => {
    const row = document.createElement('tr');
    row.append(createCell(`CNN ${String(index + 1).padStart(2, '0')}`), createCell(String(experiment.window_size)), createCell(String(experiment.hidden_size)), createCell(formatMetric(experiment.dropout, 2)), createCell(Number(experiment.learning_rate).toExponential(1)), createCell(formatMetric(experiment.validation_rmse)));
    experimentBody.append(row);
  });
}

function renderQuality(data) {
  const drift = data.sensor_drift ?? {};
  const sensors = Object.entries(drift);
  const flagged = sensors.filter(([, result]) => result.drifted).length;
  document.querySelector('#qualityFlagCount').textContent = sensors.length ? String(flagged) : '—';
  document.querySelector('#sensorsFlagged').innerHTML = `${sensors.length ? flagged : '—'} <small>/ 21</small>`;
  document.querySelector('#sensorResultCount').textContent = `${sensors.length} channel${sensors.length === 1 ? '' : 's'} evaluated`;
  document.querySelector('#driftRunState').textContent = sensors.length ? 'COMPLETED' : 'NOT RUN';
  document.querySelector('#driftRunState').classList.toggle('scored', sensors.length > 0);
  const rows = document.querySelector('#qualityRows');
  if (!sensors.length) return appendEmptyRow(rows, 5, 'No drift measurements recorded. Values will not be synthesized.');
  rows.replaceChildren();
  SENSOR_NAMES.forEach((sensor) => {
    const result = drift[sensor];
    const row = document.createElement('tr');
    const state = result ? result.drifted ? ['DRIFT FLAG', 'flagged'] : ['WITHIN THRESHOLD', 'clear'] : ['NOT CHECKED', ''];
    row.append(createCell(sensor.replace('_', ' ').toUpperCase()), createCell(formatMetric(result?.psi, 3), 'drift-value'), createCell(formatMetric(result?.ks_statistic, 3), 'drift-value'), createCell(formatMetric(result?.ks_p_value, 3), 'drift-value'));
    const status = createCell(state[0]);
    const badge = document.createElement('span');
    badge.className = `drift-status ${state[1]}`;
    badge.textContent = state[0];
    status.replaceChildren(badge);
    row.append(status);
    rows.append(row);
  });
}

function renderDashboard(data) {
  dashboard = data;
  const engines = data.engines ?? [];
  const values = engines.map((engine) => engine.predicted_rul).filter(Number.isFinite).sort((a, b) => a - b);
  const middle = Math.floor(values.length / 2);
  const median = values.length ? values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2 : null;
  const cnn = data.metrics?.cnn;
  document.querySelector('#engineCount').innerHTML = `${engines.length || '—'} <small>units</small>`;
  document.querySelector('#engineCountNote').textContent = engines.length ? 'Terminal test engines evaluated' : 'Awaiting terminal predictions';
  document.querySelector('#medianRul').innerHTML = `${formatMetric(median)} <small>cycles</small>`;
  document.querySelector('#validationRmse').innerHTML = `${formatMetric(cnn?.validation_rmse)} <small>cycles</small>`;
  document.querySelector('#sensorsFlagged').innerHTML = `${data.sensors_flagged ?? '—'} <small>/ 21</small>`;
  document.querySelector('#sensorCount').textContent = data.active_sensor_count ? `${data.active_sensor_count} / 21` : '— / 21';
  document.querySelector('#sensorState').classList.toggle('ready', Boolean(data.active_sensor_count));
  document.querySelector('#sensorState span').textContent = data.active_sensor_count ? 'MODEL READY' : 'NOT EVALUATED';
  document.querySelector('#exportReport').disabled = data.state !== 'evaluated';
  document.querySelector('#openInference').disabled = !data.model_available;
  setServiceStatus('ready', 'CONNECTED');
  renderReadiness(data);
  renderWatchlist(engines);
  renderModelSummary(data.metrics ?? {});
  renderRegistry();
  renderScatter('overviewPlot', 'overviewPlotEmpty', 'overviewY', engines);
  renderComparison(engines, data.metrics ?? {});
  renderModelEvaluation(data.metrics ?? {}, data.run_metadata ?? {});
  renderQuality(data);
}

async function refreshDashboard() {
  const button = document.querySelector('#refreshButton');
  button.disabled = true;
  button.classList.add('loading');
  setServiceStatus('loading', 'REFRESHING');
  try {
    const response = await fetch('/api/dashboard', { cache: 'no-store' });
    if (!response.ok) throw new Error(`Dashboard request returned ${response.status}`);
    renderDashboard(await response.json());
  } catch (error) {
    setServiceStatus('offline', 'OFFLINE');
    document.querySelector('#readinessBanner').dataset.state = 'service_offline';
    document.querySelector('#readinessSymbol span').textContent = '!';
    document.querySelector('#readinessSymbol').classList.remove('ready');
    document.querySelector('#readinessTitle').textContent = 'SERVICE NOT CONNECTED';
    document.querySelector('#readinessMessage').textContent = 'The dashboard could not reach the Prognos API.';
    document.querySelector('#readinessDetail').textContent = 'Run make serve from the Prognos project root to connect this view.';
    document.querySelector('#datasetState').classList.remove('ready');
    document.querySelector('#datasetState').querySelector('span').textContent = 'API OFFLINE';
    document.querySelectorAll('.file-check.available, .quality-file.available').forEach((item) => item.classList.remove('available'));
    document.querySelector('#fileProgress').textContent = '— / 3';
    document.querySelectorAll('.file-check small, .quality-file small').forEach((item) => { item.textContent = 'UNKNOWN'; });
    document.querySelectorAll('.file-state').forEach((item) => { item.textContent = '·'; });
    document.querySelectorAll('.quality-file-icon').forEach((item) => { item.textContent = '·'; });
    document.querySelector('#openInference').disabled = true;
    document.querySelector('#exportReport').disabled = true;
    document.querySelector('#qualityFileState').textContent = 'UNKNOWN';
    console.info('Prognos dashboard API is not reachable:', error);
  } finally {
    button.disabled = false;
    button.classList.remove('loading');
  }
}

function renderModelSummary(metrics) {
  const root = document.querySelector('#modelSummary');
  const models = [['linear', 'Linear regression'], ['lightgbm', 'LightGBM'], ['cnn', '1D convolutional network']];
  const scored = models.filter(([key]) => metrics?.[key]);
  if (!scored.length) return;
  root.replaceChildren();
  scored.forEach(([key, name]) => {
    const row = document.createElement('div');
    row.className = 'model-mini-row';
    const label = document.createElement('strong');
    label.textContent = name;
    const score = document.createElement('span');
    score.textContent = `RMSE ${formatMetric(metrics[key].test_rmse)} · NASA ${formatMetric(metrics[key].test_nasa)}`;
    row.append(label, score);
    root.append(row);
  });
}

function showEngineDetail(engine) {
  document.querySelector('#engineDialogTitle').textContent = `Engine ${String(engine.unit_id).padStart(3, '0')}`;
  document.querySelector('#engineDialogNote').textContent = `Final observed cycle ${engine.last_cycle} · held-out FD001 test unit`;
  const grid = document.querySelector('#engineDetailGrid');
  grid.replaceChildren();
  const values = [
    ['ESTIMATED RUL', formatRul(engine.predicted_rul)],
    ['REFERENCE RUL', formatRul(engine.actual_rul)],
    ['SIGNED ERROR', Number.isFinite(engine.actual_rul) ? `${engine.predicted_rul > engine.actual_rul ? '+' : ''}${formatMetric(engine.predicted_rul - engine.actual_rul)} cycles` : 'Not available'],
    ['ABSOLUTE ERROR', Number.isFinite(engine.actual_rul) ? `${formatMetric(Math.abs(engine.predicted_rul - engine.actual_rul))} cycles` : 'Not available'],
  ];
  values.forEach(([label, value]) => {
    const cell = document.createElement('div');
    cell.className = 'detail-cell';
    const name = document.createElement('span');
    name.textContent = label;
    const result = document.createElement('strong');
    result.textContent = value;
    cell.append(name, result);
    grid.append(cell);
  });
  document.querySelector('#engineDialog').showModal();
}

async function loadCycleFile(file) {
  const hint = document.querySelector('#inferenceHint');
  const engineSelect = document.querySelector('#engineSelect');
  const runButton = document.querySelector('#runInference');
  try {
    const lines = (await file.text()).split(/\r?\n/).filter((line) => line.trim());
    const engines = new Map();
    for (const line of lines) {
      const values = line.trim().split(/[\s,]+/).map(Number);
      if (values.length !== 26 || values.some((value) => !Number.isFinite(value)) || values[0] <= 0 || values[1] <= 0) {
        throw new Error('Each row must contain 26 finite values and positive unit/cycle IDs.');
      }
      if (!engines.has(values[0])) engines.set(values[0], []);
      engines.get(values[0]).push({ cycle: values[1], sensors: values.slice(5) });
    }
    if (!engines.size) throw new Error('The selected file contains no engine cycles.');
    for (const rows of engines.values()) rows.sort((a, b) => a.cycle - b.cycle);
    engineCycles = engines;
    engineSelect.replaceChildren(...[...engines.keys()].sort((a, b) => a - b).map((unitId) => {
      const option = document.createElement('option');
      option.value = String(unitId);
      option.textContent = `Engine ${unitId} · ${engines.get(unitId).length} cycles`;
      return option;
    }));
    engineSelect.disabled = false;
    runButton.disabled = false;
    hint.textContent = `${engines.size} engine trajectories loaded. The latest ${activeWindowLength} cycles will be used.`;
    document.querySelector('#inferenceResult').hidden = true;
  } catch (error) {
    engineCycles = new Map();
    engineSelect.replaceChildren(new Option('Invalid C-MAPSS file', ''));
    engineSelect.disabled = true;
    runButton.disabled = true;
    hint.textContent = error.message;
  }
}

async function scoreSelectedEngine() {
  const unitId = Number(document.querySelector('#engineSelect').value);
  const rows = engineCycles.get(unitId);
  if (!rows?.length) return;
  const windowRows = rows.slice(-activeWindowLength).map((row) => row.sensors);
  while (windowRows.length < activeWindowLength) windowRows.unshift([...windowRows[0]]);
  const resultBox = document.querySelector('#inferenceResult');
  const runButton = document.querySelector('#runInference');
  runButton.disabled = true;
  runButton.textContent = 'Scoring…';
  resultBox.hidden = true;
  try {
    const response = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ window: windowRows }) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail ?? `Prediction failed (${response.status})`);
    resultBox.replaceChildren();
    const label = document.createElement('span');
    label.className = 'eyebrow';
    label.textContent = `ENGINE ${unitId} · ESTIMATED REMAINING LIFE`;
    const value = document.createElement('strong');
    value.textContent = formatRul(result.rul);
    const note = document.createElement('p');
    note.textContent = 'Model output for the final observed cycle. Research estimate only; not an automated maintenance decision.';
    resultBox.append(label, value, note);
    resultBox.hidden = false;
  } catch (error) {
    resultBox.replaceChildren();
    const message = document.createElement('p');
    message.textContent = `Prediction unavailable: ${error.message}`;
    resultBox.append(message);
    resultBox.hidden = false;
  } finally {
    runButton.disabled = false;
    runButton.textContent = 'Run prediction';
  }
}

function exportEvaluation() {
  if (!dashboard || dashboard.state !== 'evaluated') return;
  const report = { product: 'Prognos propulsion health research prototype', dataset: 'NASA C-MAPSS FD001 simulated benchmark', note: 'Held-out benchmark output, not a flight-qualified maintenance recommendation.', metrics: dashboard.metrics, engines: dashboard.engines, run_metadata: dashboard.run_metadata, sensor_drift: dashboard.sensor_drift };
  const file = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(file);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'prognos-fd001-evaluation.json';
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

document.querySelectorAll('.rail-link[role="tab"]').forEach((tab, index, tabs) => {
  tab.addEventListener('click', () => switchView(tab.dataset.view));
  tab.addEventListener('keydown', (event) => {
    if (!['ArrowDown', 'ArrowUp', 'ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1
      : (index + (event.key === 'ArrowDown' || event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    tabs[next].focus();
    switchView(tabs[next].dataset.view);
  });
});
const updateTabOrientation = () => {
  document.querySelector('[role="tablist"]').setAttribute('aria-orientation', matchMedia('(max-width: 700px)').matches ? 'horizontal' : 'vertical');
};
window.addEventListener('resize', updateTabOrientation);
updateTabOrientation();
document.querySelectorAll('[data-go-view]').forEach((control) => control.addEventListener('click', (event) => {
  event.preventDefault();
  switchView(control.dataset.goView);
}));
window.addEventListener('popstate', () => switchView(location.hash.slice(1), false));
document.querySelector('#refreshButton').addEventListener('click', refreshDashboard);
document.querySelector('#exportReport').addEventListener('click', exportEvaluation);
document.querySelector('#engineSearch').addEventListener('input', renderRegistry);
document.querySelector('#clearEngineSearch').addEventListener('click', () => {
  document.querySelector('#engineSearch').value = '';
  renderRegistry();
  document.querySelector('#engineSearch').focus();
});
document.querySelectorAll('[data-sort]').forEach((button) => button.addEventListener('click', () => {
  const key = button.dataset.sort;
  registrySort = { key, direction: registrySort.key === key ? -registrySort.direction : 1 };
  document.querySelectorAll('[data-sort]').forEach((item) => {
    const selected = item.dataset.sort === key;
    item.closest('th').setAttribute('aria-sort', selected ? registrySort.direction === 1 ? 'ascending' : 'descending' : 'none');
    item.querySelector('span').textContent = selected ? registrySort.direction === 1 ? '↑' : '↓' : '↕';
  });
  renderRegistry();
}));
document.querySelector('#cycleFile').addEventListener('change', (event) => {
  if (event.target.files?.[0]) loadCycleFile(event.target.files[0]);
});
document.querySelector('#runInference').addEventListener('click', scoreSelectedEngine);
document.querySelector('#openInference').addEventListener('click', async () => {
  try {
    const response = await fetch('/model-info', { cache: 'no-store' });
    if (!response.ok) throw new Error('Model metadata unavailable');
    const metadata = await response.json();
    activeWindowLength = metadata.window_length;
    document.querySelector('#inferenceHint').textContent = `Expected 26 columns. The model uses the latest ${activeWindowLength} cycles for the selected engine.`;
  } catch {
    document.querySelector('#inferenceHint').textContent = 'Could not read window metadata from the local API.';
  }
  document.querySelector('#inferenceDialog').showModal();
});

switchView(location.hash.slice(1) || 'overview', false);
refreshDashboard();
window.setInterval(refreshDashboard, 30000);
