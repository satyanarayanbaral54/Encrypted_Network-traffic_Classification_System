/* ═══════════════════════════════════════════════════════════════
   NetTraffic AI – Main Application Logic
   All API calls, page navigation, charting, and interactivity
   ═══════════════════════════════════════════════════════════════ */

const API_BASE = '';

// ── Global State ────────────────────────────────────────────
let appState = {
    metadata: null,
    features: [],
    classes: [],
    testData: null,
    batchFile: null,
    batchCSVData: null,
    liveRunning: false,
};

// ── Navigation ──────────────────────────────────────────────

function navigateTo(pageId) {
    // Hide all pages
    document.querySelectorAll('.page-section').forEach(s => {
        s.classList.remove('active');
    });

    // Remove active from nav links
    document.querySelectorAll('.nav-link').forEach(l => {
        l.classList.remove('active');
    });

    // Show target page
    const target = document.getElementById(`page-${pageId}`);
    if (target) {
        target.classList.add('active');
    }

    // Activate nav link
    const navLink = document.querySelector(`[data-page="${pageId}"]`);
    if (navLink) {
        navLink.classList.add('active');
    }

    // Close mobile sidebar
    document.getElementById('sidebar').classList.remove('open');

    // Lazy-load page data
    switch (pageId) {
        case 'explorer':
            if (!appState.testData) loadExplorerData();
            break;
        case 'single':
            renderFeatureInputs();
            break;
        case 'performance':
            loadPerformanceData();
            break;
    }
}

// Mobile toggle
document.getElementById('mobileToggle').addEventListener('click', () => {
    document.getElementById('sidebar').classList.toggle('open');
});

// ── API Helpers ─────────────────────────────────────────────

async function apiFetch(url, options = {}) {
    try {
        const resp = await fetch(`${API_BASE}${url}`, options);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return await resp.json();
    } catch (err) {
        console.error(`API Error [${url}]:`, err);
        return null;
    }
}

function showLoading(text = 'Loading...') {
    document.getElementById('loadingText').textContent = text;
    document.getElementById('loadingOverlay').classList.add('active');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('active');
}

// ── Initialization ──────────────────────────────────────────

async function initApp() {
    showLoading('Loading model artifacts...');

    const data = await apiFetch('/api/metadata');
    if (data) {
        appState.metadata = data.metadata;
        appState.features = data.features || [];
        appState.classes = data.classes || [];

        // Populate home metrics
        const tm = data.metadata?.test_metrics || {};
        animateNumber('metricAccuracy', (tm.accuracy || 0.933) * 100, '%', 1);
        animateNumber('metricF1', tm.macro_f1 || 0.613, '', 3);
        document.getElementById('metricFeatures').textContent = appState.features.length || 20;
        document.getElementById('metricClasses').textContent = appState.classes.length || 5;
    }

    hideLoading();
}

function animateNumber(elementId, target, suffix = '', decimals = 1) {
    const el = document.getElementById(elementId);
    if (!el) return;
    let current = 0;
    const step = target / 40;
    const interval = setInterval(() => {
        current += step;
        if (current >= target) {
            current = target;
            clearInterval(interval);
        }
        el.textContent = current.toFixed(decimals) + suffix;
    }, 25);
}

// ── PAGE 2: Data Explorer ───────────────────────────────────

async function loadExplorerData() {
    showLoading('Loading test dataset...');

    const data = await apiFetch('/api/test-sample?n=1000');
    if (data) {
        appState.testData = data;
        renderExplorerTable(data);
        renderClassDistChart(data.class_distribution);
        populateFeatureSelect();
        renderFeatureHistogram();
    }

    const featureStats = await apiFetch('/api/feature-stats');
    if (featureStats) {
        renderCorrelationHeatmap(featureStats.correlation, featureStats.features);
    }

    hideLoading();
}

function renderExplorerTable(data) {
    const container = document.getElementById('explorerTable');
    if (!data.data || data.data.length === 0) {
        container.innerHTML = '<p style="padding:20px;text-align:center;color:var(--text-muted);">No data available</p>';
        return;
    }

    const cols = data.columns.slice(0, 12); // Show first 12 columns
    const rows = data.data.slice(0, 10);

    let html = '<table><thead><tr>';
    cols.forEach(c => { html += `<th>${escapeHtml(c)}</th>`; });
    html += '</tr></thead><tbody>';

    rows.forEach(row => {
        html += '<tr>';
        cols.forEach(c => {
            let val = row[c];
            if (typeof val === 'number') val = val.toFixed(4);
            html += `<td>${escapeHtml(String(val ?? ''))}</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

function renderClassDistChart(dist) {
    if (!dist) return;
    const classes = Object.keys(dist);
    const counts = Object.values(dist);

    const colors = {
        'BENIGN': '#10b981',
        'DDoS': '#ef4444',
        'PortScan': '#f59e0b',
        'Bot': '#818cf8',
        'Infiltration': '#dc2626',
    };

    const trace = {
        x: classes,
        y: counts,
        type: 'bar',
        marker: {
            color: classes.map(c => colors[c] || '#38bdf8'),
            line: { color: 'rgba(255,255,255,0.1)', width: 1 }
        },
        text: counts.map(c => c.toLocaleString()),
        textposition: 'auto',
        hovertemplate: '%{x}: %{y:,} flows<extra></extra>',
    };

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { family: 'Inter', color: '#94a3b8', size: 12 },
        margin: { l: 50, r: 20, t: 30, b: 50 },
        yaxis: { type: 'log', title: 'Flow Count (Log)', gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
        xaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
        height: 300,
    };

    Plotly.newPlot('classDistChart', [trace], layout, { responsive: true, displayModeBar: false });
}

function populateFeatureSelect() {
    const select = document.getElementById('featureSelect');
    select.innerHTML = '';
    appState.features.forEach(f => {
        const opt = document.createElement('option');
        opt.value = f;
        opt.textContent = f;
        select.appendChild(opt);
    });
}

function renderFeatureHistogram() {
    const feat = document.getElementById('featureSelect').value;
    if (!appState.testData || !feat) return;

    const rows = appState.testData.data;
    const values = rows.map(r => r[feat]).filter(v => v != null && isFinite(v));

    if (values.length === 0) return;

    const trace = {
        x: values,
        type: 'histogram',
        marker: {
            color: 'rgba(56, 189, 248, 0.6)',
            line: { color: 'rgba(56, 189, 248, 0.9)', width: 1 }
        },
        nbinsx: 40,
        hovertemplate: 'Range: %{x}<br>Count: %{y}<extra></extra>',
    };

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { family: 'Inter', color: '#94a3b8', size: 12 },
        margin: { l: 50, r: 20, t: 10, b: 50 },
        xaxis: { title: feat, gridcolor: 'rgba(255,255,255,0.05)' },
        yaxis: { title: 'Count', gridcolor: 'rgba(255,255,255,0.05)' },
        height: 280,
    };

    Plotly.newPlot('featureHistChart', [trace], layout, { responsive: true, displayModeBar: false });
}

function renderCorrelationHeatmap(corr, features) {
    if (!corr || !features) return;

    const z = features.map(f1 => features.map(f2 => {
        const val = corr[f1]?.[f2];
        return val != null ? parseFloat(val.toFixed(2)) : 0;
    }));

    // Shorten feature names for display
    const shortNames = features.map(f => f.length > 18 ? f.slice(0, 16) + '…' : f);

    const trace = {
        z: z,
        x: shortNames,
        y: shortNames,
        type: 'heatmap',
        colorscale: 'Viridis',
        showscale: true,
        hovertemplate: '%{y} × %{x}<br>Correlation: %{z:.2f}<extra></extra>',
        colorbar: { tickfont: { color: '#94a3b8' } },
    };

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { family: 'Inter', color: '#94a3b8', size: 10 },
        margin: { l: 140, r: 30, t: 20, b: 140 },
        height: 500,
        xaxis: { tickangle: -45 },
    };

    Plotly.newPlot('correlationHeatmap', [trace], layout, { responsive: true, displayModeBar: false });
}

// ── PAGE 3: Single Flow Prediction ──────────────────────────

function renderFeatureInputs() {
    if (appState.features.length === 0) return;
    const container = document.getElementById('featureInputs');

    // Only render once
    if (container.dataset.rendered === 'true') return;
    container.dataset.rendered = 'true';

    container.innerHTML = '';
    appState.features.forEach(feat => {
        const div = document.createElement('div');
        div.className = 'input-group';
        div.innerHTML = `
            <label for="feat_${feat}">${escapeHtml(feat)}</label>
            <input type="number" id="feat_${feat}" class="input-field" value="0" step="any" placeholder="0.0000">
        `;
        container.appendChild(div);
    });
}

async function loadPreset(type) {
    showLoading(`Loading ${type} preset...`);

    const data = await apiFetch(`/api/preset/${type}`);
    if (data && data.flow) {
        // Ensure inputs are rendered
        renderFeatureInputs();

        // Fill input values
        Object.entries(data.flow).forEach(([feat, val]) => {
            const input = document.getElementById(`feat_${feat}`);
            if (input) {
                input.value = parseFloat(val).toFixed(4);
                // Flash animation
                input.style.borderColor = 'var(--accent-primary)';
                input.style.boxShadow = '0 0 0 3px rgba(56,189,248,0.15)';
                setTimeout(() => {
                    input.style.borderColor = '';
                    input.style.boxShadow = '';
                }, 800);
            }
        });
    }

    hideLoading();
}

async function classifySingleFlow() {
    const flow = {};
    appState.features.forEach(feat => {
        const input = document.getElementById(`feat_${feat}`);
        flow[feat] = input ? parseFloat(input.value) || 0 : 0;
    });

    const btn = document.getElementById('classifyBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Classifying...';

    const data = await apiFetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ flow }),
    });

    btn.disabled = false;
    btn.innerHTML = '🚀 Classify Network Flow';

    if (!data) return;

    // Show results
    document.getElementById('singleResult').style.display = 'block';

    // Verdict
    const isBenign = data.prediction.toUpperCase().includes('BENIGN');
    const verdictHtml = `
        <div class="prediction-result ${isBenign ? 'safe' : 'threat'}">
            <div class="prediction-class">${isBenign ? '🟢' : '🔴'} ${data.prediction}</div>
            <div class="prediction-confidence">${isBenign ? 'Safe Flow — No Threat Detected' : '⚠️ Malicious Attack Detected!'}</div>
        </div>
    `;
    document.getElementById('predictionVerdict').innerHTML = verdictHtml;

    // Confidence gauge
    const conf = data.confidence * 100;
    const circumference = 2 * Math.PI * 78; // r=78
    const offset = circumference * (1 - data.confidence);
    const gaugeFill = document.getElementById('gaugeFill');
    gaugeFill.style.strokeDasharray = circumference;
    gaugeFill.style.strokeDashoffset = circumference; // start from 0
    gaugeFill.style.stroke = isBenign ? '#10b981' : '#ef4444';
    setTimeout(() => {
        gaugeFill.style.strokeDashoffset = offset;
    }, 100);
    document.getElementById('gaugeValue').textContent = conf.toFixed(1) + '%';
    document.getElementById('gaugeValue').style.color = isBenign ? '#10b981' : '#ef4444';

    // Probability bars
    renderProbabilityBars(data.probabilities);

    // Attribution bars
    renderAttributionBars(data.attributions);

    // Scroll to results
    document.getElementById('singleResult').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderProbabilityBars(probs) {
    const container = document.getElementById('probBars');
    const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

    const colors = {
        'BENIGN': '#10b981',
        'DDoS': '#ef4444',
        'PortScan': '#f59e0b',
        'Bot': '#818cf8',
        'Infiltration': '#dc2626',
    };

    let html = '';
    sorted.forEach(([cls, prob]) => {
        const pct = (prob * 100).toFixed(2);
        const color = colors[cls] || '#38bdf8';
        html += `
            <div class="progress-bar-container">
                <div class="progress-label">
                    <span>${cls}</span>
                    <span>${pct}%</span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="width:${pct}%;background:${color};"></div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

function renderAttributionBars(attrs) {
    const container = document.getElementById('attributionBars');
    const sorted = Object.entries(attrs)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 10);

    const maxAbs = Math.max(...sorted.map(([, v]) => Math.abs(v)), 0.001);

    let html = '';
    sorted.forEach(([feat, val]) => {
        const isPositive = val >= 0;
        const pct = Math.min((Math.abs(val) / maxAbs) * 100, 100).toFixed(1);
        const color = isPositive ? '#ef4444' : '#10b981';
        const shortFeat = feat.length > 25 ? feat.slice(0, 23) + '…' : feat;
        html += `
            <div class="progress-bar-container">
                <div class="progress-label">
                    <span>${escapeHtml(shortFeat)}</span>
                    <span style="color:${color}">${val >= 0 ? '+' : ''}${val.toFixed(4)}</span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="width:${pct}%;background:${color};"></div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

// ── PAGE 4: Batch Prediction ────────────────────────────────

// Drag-and-drop
const uploadZone = document.getElementById('uploadZone');
if (uploadZone) {
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith('.csv')) {
            appState.batchFile = file;
            showBatchFileInfo(`📄 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
            document.getElementById('batchClassifyBtn').disabled = false;
        }
    });
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        appState.batchFile = file;
        showBatchFileInfo(`📄 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
        document.getElementById('batchClassifyBtn').disabled = false;
    }
}

function showBatchFileInfo(text) {
    const info = document.getElementById('batchFileInfo');
    info.style.display = 'flex';
    info.innerHTML = `<span>📄</span> ${escapeHtml(text)}`;
}

async function loadBatchSample(n) {
    appState.batchFile = null;
    appState._useSample = n;
    showBatchFileInfo(`📁 Using ${n}-flow test sample`);
    document.getElementById('batchClassifyBtn').disabled = false;
}

async function runBatchClassification() {
    const btn = document.getElementById('batchClassifyBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Classifying...';

    let data;

    if (appState.batchFile) {
        const formData = new FormData();
        formData.append('file', appState.batchFile);
        try {
            const resp = await fetch(`${API_BASE}/api/predict-batch`, { method: 'POST', body: formData });
            data = await resp.json();
        } catch (err) {
            console.error(err);
        }
    } else if (appState._useSample) {
        data = await apiFetch('/api/predict-batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ use_sample: true, sample_size: appState._useSample }),
        });
    }

    btn.disabled = false;
    btn.innerHTML = '⚡ Run Batch Classification';

    if (!data) return;

    appState.batchCSVData = data;
    document.getElementById('batchResults').style.display = 'block';

    // Pie chart
    renderBatchPieChart(data.class_distribution);

    // Breakdown table
    renderBatchBreakdown(data.class_distribution, data.total_flows);

    // Results table
    renderBatchTable(data);

    document.getElementById('batchResults').scrollIntoView({ behavior: 'smooth' });
}

function renderBatchPieChart(dist) {
    const colors = {
        'BENIGN': '#10b981',
        'DDoS': '#ef4444',
        'PortScan': '#f59e0b',
        'Bot': '#818cf8',
        'Infiltration': '#dc2626',
    };

    const labels = Object.keys(dist);
    const values = Object.values(dist);

    const trace = {
        labels: labels,
        values: values,
        type: 'pie',
        hole: 0.45,
        marker: {
            colors: labels.map(l => colors[l] || '#38bdf8'),
            line: { color: '#0a0e1a', width: 2 },
        },
        textinfo: 'label+percent',
        textfont: { color: '#e2e8f0', size: 12 },
        hovertemplate: '%{label}: %{value:,} flows (%{percent})<extra></extra>',
    };

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { family: 'Inter', color: '#94a3b8' },
        margin: { l: 20, r: 20, t: 20, b: 20 },
        height: 300,
        showlegend: false,
    };

    Plotly.newPlot('batchPieChart', [trace], layout, { responsive: true, displayModeBar: false });
}

function renderBatchBreakdown(dist, total) {
    const container = document.getElementById('batchBreakdown');
    let html = '<table><thead><tr><th>Predicted Class</th><th>Flow Count</th><th>Percentage</th></tr></thead><tbody>';

    const sorted = Object.entries(dist).sort((a, b) => b[1] - a[1]);
    sorted.forEach(([cls, count]) => {
        const pct = ((count / total) * 100).toFixed(2);
        html += `<tr><td>${escapeHtml(cls)}</td><td>${count.toLocaleString()}</td><td>${pct}%</td></tr>`;
    });

    html += `<tr style="font-weight:700;border-top:2px solid var(--border-subtle);">
        <td>Total</td><td>${total.toLocaleString()}</td><td>100%</td></tr>`;
    html += '</tbody></table>';
    container.innerHTML = html;
}

function renderBatchTable(data) {
    const container = document.getElementById('batchTable');
    const cols = ['Predicted_Class', 'Confidence', ...appState.features.slice(0, 6)];
    const rows = data.results.slice(0, 50);

    let html = '<table><thead><tr>';
    html += '<th>#</th>';
    cols.forEach(c => { html += `<th>${escapeHtml(c)}</th>`; });
    html += '</tr></thead><tbody>';

    rows.forEach((row, i) => {
        const isAttack = row.Predicted_Class && !row.Predicted_Class.toUpperCase().includes('BENIGN');
        html += `<tr style="${isAttack ? 'background:rgba(239,68,68,0.05)' : ''}">`;
        html += `<td>${i + 1}</td>`;
        cols.forEach(c => {
            let val = row[c];
            if (c === 'Predicted_Class') {
                const badge = isAttack ? 'badge-danger' : 'badge-success';
                html += `<td><span class="badge ${badge}">${escapeHtml(String(val))}</span></td>`;
            } else if (typeof val === 'number') {
                html += `<td>${val.toFixed(4)}</td>`;
            } else {
                html += `<td>${escapeHtml(String(val ?? ''))}</td>`;
            }
        });
        html += '</tr>';
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

function downloadBatchCSV() {
    if (!appState.batchCSVData) return;

    const rows = appState.batchCSVData.results;
    if (!rows || rows.length === 0) return;

    const cols = appState.batchCSVData.columns;
    let csv = cols.join(',') + '\n';
    rows.forEach(row => {
        csv += cols.map(c => {
            let val = row[c] ?? '';
            if (typeof val === 'string' && val.includes(',')) val = `"${val}"`;
            return val;
        }).join(',') + '\n';
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'classified_network_traffic.csv';
    a.click();
    URL.revokeObjectURL(url);
}

// ── PAGE 5: Live Traffic Simulation ─────────────────────────

async function startLiveSimulation() {
    if (appState.liveRunning) return;

    const numPackets = parseInt(document.getElementById('numPackets').value);
    const delay = parseInt(document.getElementById('packetDelay').value);

    const btn = document.getElementById('startSimBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Streaming...';
    appState.liveRunning = true;

    // Reset counters
    document.getElementById('liveTotal').textContent = '0';
    document.getElementById('liveBenign').textContent = '0';
    document.getElementById('liveAttack').textContent = '0';
    document.getElementById('liveFeed').innerHTML = '';

    // Fetch all predictions at once
    const data = await apiFetch(`/api/simulate-batch?n=${numPackets}`);
    if (!data || !data.flows) {
        btn.disabled = false;
        btn.innerHTML = '▶️ Start Live Traffic Stream';
        appState.liveRunning = false;
        return;
    }

    const flows = data.flows;
    let benignCount = 0;
    let attackCount = 0;
    const feed = document.getElementById('liveFeed');

    // Stream flows one at a time
    for (let i = 0; i < flows.length; i++) {
        const flow = flows[i];
        const timestamp = new Date().toLocaleTimeString();

        if (flow.is_attack) {
            attackCount++;
        } else {
            benignCount++;
        }

        // Update counters with animation
        updateCounter('liveTotal', i + 1);
        updateCounter('liveBenign', benignCount);
        updateCounter('liveAttack', attackCount);

        // Add feed item
        const item = document.createElement('div');
        item.className = `feed-item ${flow.is_attack ? 'attack' : 'benign'}`;
        item.innerHTML = `
            <span class="timestamp">${timestamp}</span>
            <span class="flow-id">FLOW #${String(flow.id).padStart(3, '0')}</span>
            <span class="classification">
                ${flow.is_attack
                    ? `🚨 <strong>ATTACK: ${flow.prediction}</strong> — Immediate Inspection Advised!`
                    : `✅ Normal Traffic (BENIGN) — Conf: ${(flow.confidence * 100).toFixed(1)}%`
                }
            </span>
            <span class="badge ${flow.is_attack ? 'badge-danger' : 'badge-success'}">${(flow.confidence * 100).toFixed(0)}%</span>
        `;

        feed.insertBefore(item, feed.firstChild);

        // Keep only last 50 items in feed
        while (feed.children.length > 50) {
            feed.removeChild(feed.lastChild);
        }

        await sleep(delay);
    }

    btn.disabled = false;
    btn.innerHTML = '▶️ Start Live Traffic Stream';
    appState.liveRunning = false;

    // Show completion
    const completeItem = document.createElement('div');
    completeItem.className = 'alert alert-success';
    completeItem.innerHTML = `<span>✅</span> Real-time simulation stream concluded. ${flows.length} flows processed — ${attackCount} threats detected.`;
    feed.insertBefore(completeItem, feed.firstChild);
}

function updateCounter(id, value) {
    const el = document.getElementById(id);
    el.textContent = value;
    el.style.transform = 'scale(1.1)';
    setTimeout(() => { el.style.transform = 'scale(1)'; }, 150);
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ── PAGE 6: Model Performance ───────────────────────────────

async function loadPerformanceData() {
    const metricsContainer = document.getElementById('perfMetrics');
    if (metricsContainer.dataset.loaded === 'true') return;
    metricsContainer.dataset.loaded = 'true';

    // Metrics
    const metrics = await apiFetch('/api/metrics');
    if (metrics && !metrics.error) {
        metricsContainer.innerHTML = `
            <div class="metric-card">
                <div class="metric-label">Overall Accuracy</div>
                <div class="metric-value success">${(metrics.accuracy * 100).toFixed(2)}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Macro F1-Score</div>
                <div class="metric-value">${metrics.macro_f1.toFixed(4)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Weighted F1-Score</div>
                <div class="metric-value">${metrics.weighted_f1.toFixed(4)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Macro Recall</div>
                <div class="metric-value">${metrics.macro_recall.toFixed(4)}</div>
            </div>
        `;

        // Eval note
        if (metrics.evaluation_notes) {
            const note = document.getElementById('evalNote');
            note.style.display = 'flex';
            note.innerHTML = `<span>ℹ️</span> ${escapeHtml(metrics.evaluation_notes)}`;
        }

        // Classification report
        renderClassReport(metrics.classification_report);
    }

    // Model comparison
    const comparison = await apiFetch('/api/model-comparison');
    if (comparison && !comparison.error) {
        renderModelComparison(comparison);
    }
}

function renderModelComparison(data) {
    const container = document.getElementById('modelCompTable');
    if (!data || data.length === 0) {
        container.innerHTML = '<p style="padding:20px;text-align:center;color:var(--text-muted);">No comparison data</p>';
        return;
    }

    let html = '<table><thead><tr><th>Model</th><th>Stage</th><th>Accuracy</th><th>Macro F1</th></tr></thead><tbody>';
    data.forEach(row => {
        html += `<tr>
            <td style="color:var(--text-primary);font-weight:500;">${escapeHtml(row.Model)}</td>
            <td>${escapeHtml(row.Stage)}</td>
            <td>${typeof row.Accuracy === 'number' ? row.Accuracy.toFixed(5) : row.Accuracy}</td>
            <td>${typeof row['Macro F1'] === 'number' ? row['Macro F1'].toFixed(6) : row['Macro F1']}</td>
        </tr>`;
    });
    html += '</tbody></table>';
    container.innerHTML = html;
}

function renderClassReport(report) {
    const container = document.getElementById('classReportTable');
    if (!report) return;

    const classes = Object.keys(report).filter(k => !['accuracy', 'macro avg', 'weighted avg'].includes(k));
    const extras = ['macro avg', 'weighted avg'];

    let html = '<table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead><tbody>';

    [...classes, ...extras].forEach(cls => {
        const row = report[cls];
        if (!row) return;
        const isExtra = extras.includes(cls);
        html += `<tr style="${isExtra ? 'border-top:2px solid var(--border-subtle);font-weight:600;' : ''}">
            <td style="color:var(--text-primary);font-weight:500;">${escapeHtml(cls)}</td>
            <td>${row.precision?.toFixed(4) ?? '-'}</td>
            <td>${row.recall?.toFixed(4) ?? '-'}</td>
            <td>${row['f1-score']?.toFixed(4) ?? '-'}</td>
            <td>${row.support?.toLocaleString() ?? '-'}</td>
        </tr>`;
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

// ── PAGE 7: Explainability ──────────────────────────────────

async function loadExplainability() {
    showLoading('Analyzing random flow...');

    // Get a random preset
    const presetData = await apiFetch('/api/preset/random');
    if (!presetData || !presetData.flow) {
        hideLoading();
        return;
    }

    // Classify it
    const predData = await apiFetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ flow: presetData.flow }),
    });

    hideLoading();

    if (!predData) return;

    document.getElementById('explainResults').style.display = 'block';

    // Flow parameters table
    const flowTable = document.getElementById('explainFlowTable');
    let html = '<table><thead><tr><th>Feature</th><th>Value</th></tr></thead><tbody>';
    Object.entries(presetData.flow).forEach(([feat, val]) => {
        html += `<tr><td>${escapeHtml(feat)}</td><td>${parseFloat(val).toFixed(4)}</td></tr>`;
    });
    html += '</tbody></table>';
    flowTable.innerHTML = html;

    // Attribution chart
    const attrs = predData.attributions;
    const sorted = Object.entries(attrs).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 10);

    const trace = {
        y: sorted.map(([f]) => f.length > 22 ? f.slice(0, 20) + '…' : f).reverse(),
        x: sorted.map(([, v]) => v).reverse(),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: sorted.map(([, v]) => v >= 0 ? 'rgba(239,68,68,0.7)' : 'rgba(16,185,129,0.7)').reverse(),
            line: { width: 1, color: sorted.map(([, v]) => v >= 0 ? '#ef4444' : '#10b981').reverse() },
        },
        hovertemplate: '%{y}: %{x:.4f}<extra></extra>',
    };

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { family: 'Inter', color: '#94a3b8', size: 11 },
        margin: { l: 160, r: 30, t: 10, b: 40 },
        xaxis: { title: 'Attribution Score', gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.2)' },
        height: 350,
    };

    Plotly.newPlot('explainAttrChart', [trace], layout, { responsive: true, displayModeBar: false });

    // Rationale
    const topFeat = sorted[0][0];
    const topVal = sorted[0][1];
    const direction = topVal > 0 ? 'an attack' : 'benign behavior';
    const rationale = document.getElementById('explainRationale');
    rationale.innerHTML = `
        <strong>Prediction:</strong> <span style="color:${predData.prediction.includes('BENIGN') ? '#10b981' : '#ef4444'};font-weight:700;">${predData.prediction}</span>
        (Confidence: ${(predData.confidence * 100).toFixed(1)}%)
        <br><br>
        The flow was strongly influenced towards <strong>${direction}</strong> primarily due to anomalous
        characteristics in <code style="background:var(--bg-secondary);padding:2px 8px;border-radius:4px;color:var(--accent-primary);">${escapeHtml(topFeat)}</code>
        (standardized impact: <code style="background:var(--bg-secondary);padding:2px 8px;border-radius:4px;">${topVal >= 0 ? '+' : ''}${topVal.toFixed(4)}</code>).
    `;
}

// ── Utilities ───────────────────────────────────────────────

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ── Boot ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', initApp);
