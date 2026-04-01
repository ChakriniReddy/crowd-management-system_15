const MAX_DATA_POINTS = 20;
let soundEnabled = document.getElementById('toggle-sound').checked;
let alertSound = document.getElementById('alert-sound');
let lastAlertCount = 0;

const chartDataHistory = {
    'Global': { actual: Array(MAX_DATA_POINTS).fill(0), pred: Array(MAX_DATA_POINTS).fill(0) }
};
['A','B','C','D'].forEach(r => {
    [1,2,3,4].forEach(c => {
        chartDataHistory[`${r}${c}`] = { actual: Array(MAX_DATA_POINTS).fill(0), pred: Array(MAX_DATA_POINTS).fill(0) };
    });
});

// Top Nav Clock
setInterval(() => {
    document.getElementById('system-clock').textContent = new Date().toLocaleTimeString('en-US', {hour12:false});
}, 1000);

// Toast Notification System
let activeToasts = new Set();
function showToast(message) {
    // Deduplicate dynamic numerical changes
    const baseMsg = message.replace(/[0-9]+/g, '');
    if (activeToasts.has(baseMsg)) return;
    
    const container = document.getElementById('toast-container');
    // Limit to 2 visible toasts maximum to prevent covering dashboard
    if (container.children.length >= 2) return;
    
    activeToasts.add(baseMsg);
    
    const toast = document.createElement('div');
    toast.className = 'toast-notification';
    toast.innerHTML = `<span>⚠️ ${message}</span>`;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOutUp 0.3s forwards';
        setTimeout(() => { toast.remove(); activeToasts.delete(baseMsg); }, 300);
    }, 4000);
}

// Camera Tab Switching
function switchTab(btn, source) {
    document.querySelectorAll('.cam-tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('camera-source').value = source;
    document.getElementById('apply-settings').click();
}

// Simulation Controls
async function controlSimulation(action) {
    try {
        await fetch('/api/simulation_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({action: action})
        });
    } catch(e) { console.error(e); }
}

// Chart.js Setup
const ctx = document.getElementById('crowdChart').getContext('2d');
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', sans-serif";

const crowdChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: Array(MAX_DATA_POINTS).fill(''),
        datasets: [
            {
                label: 'Actual Crowd',
                data: Array(MAX_DATA_POINTS).fill(0),
                borderColor: '#0ea5e9',
                backgroundColor: 'rgba(14, 165, 233, 0.15)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            },
            {
                label: 'Predicted (30s+)',
                data: Array(MAX_DATA_POINTS).fill(0),
                borderColor: '#fbbf24',
                borderWidth: 2,
                borderDash: [5, 5],
                tension: 0.4,
                pointRadius: 0
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 8 } }
        },
        scales: {
            y: { beginAtZero: true, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
            x: { grid: { display: false } }
        },
        animation: { duration: 0 }
    }
});

// Update Settings API
document.getElementById('apply-settings').addEventListener('click', async () => {
    const sourceSelect = document.getElementById('camera-source').value;
    const finalSource = sourceSelect === 'custom' ? document.getElementById('custom-video-input').value : sourceSelect;

    const payload = {
        detection_active: document.getElementById('toggle-detection').checked,
        camera_source: finalSource,
        threshold: parseInt(document.getElementById('crowd-threshold').value),
        show_heatmap: document.getElementById('toggle-heatmap').checked,
        show_grid: document.getElementById('toggle-grid').checked,
        sound_alerts: document.getElementById('toggle-sound').checked
    };
    
    soundEnabled = payload.sound_alerts;

    try {
        await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        // Quick visual confirm
        const btn = document.getElementById('apply-settings');
        btn.textContent = "Applied ✓";
        btn.style.background = "#10b981";
        setTimeout(() => {
            btn.textContent = "Apply Parameters";
            btn.style.background = "#0ea5e9";
        }, 1500);

    } catch (e) {
        console.error("Failed to update settings:", e);
    }
});

// Slider value sync
document.getElementById('crowd-threshold').addEventListener('input', (e) => {
    document.getElementById('threshold-val').textContent = e.target.value;
});

// Data Fetching
async function fetchData() {
    try {
        const res = await fetch('/api/data');
        const data = await res.json();
        
        // Top Counters
        const actualTotal = data.counts.total;
        const predTotal = data.predictions.total;
        
        document.getElementById('total-count').textContent = actualTotal;
        document.getElementById('predicted-count').textContent = predTotal;
        document.getElementById('fps-counter').textContent = data.fps;

        // Chart Update
        const viewSelected = document.getElementById('chart-view-selector').value;
        const timeNow = new Date().toLocaleTimeString('en-US', {hour12:false, minute:'2-digit', second:'2-digit'});
        
        crowdChart.data.labels.push(timeNow);
        crowdChart.data.labels.shift();

        // Update Global History
        chartDataHistory['Global'].actual.push(actualTotal);
        chartDataHistory['Global'].actual.shift();
        chartDataHistory['Global'].pred.push(predTotal);
        chartDataHistory['Global'].pred.shift();

        // Risk Grid Update
        const riskGrid = document.getElementById('risk-grid-container');
        riskGrid.innerHTML = '';

        if(data.grid_predictions) {
            for(const [gridId, stats] of Object.entries(data.grid_predictions)) {
                // Update Grid History
                if(chartDataHistory[gridId]) {
                    chartDataHistory[gridId].actual.push(stats.current);
                    chartDataHistory[gridId].actual.shift();
                    chartDataHistory[gridId].pred.push(stats.forecast);
                    chartDataHistory[gridId].pred.shift();
                }

                const cell = document.createElement('div');
                let levelClass = 'low';
                if (stats.risk === 'WARNING') levelClass = 'medium';
                if (stats.risk === 'HIGH') levelClass = 'high';
                
                cell.className = `risk-cell risk-${levelClass} grid-card`;
                
                cell.innerHTML = `
                    <strong>${gridId}</strong>
                    <div class="grid-metrics">${stats.current} | ${stats.forecast}</div>
                    <div class="grid-tooltip">
                        <div>Current: <span style="color:#0ea5e9">${stats.current}</span></div>
                        <div>Forecast: <span style="color:#fbbf24">${stats.forecast}</span></div>
                        <div>Risk: <span class="risk-text-${levelClass}">${stats.risk}</span></div>
                    </div>
                `;
                riskGrid.appendChild(cell);
            }
        }

        // Apply selected chart view
        crowdChart.data.datasets[0].data = [...chartDataHistory[viewSelected].actual];
        crowdChart.data.datasets[1].data = [...chartDataHistory[viewSelected].pred];
        
        // Update Chart Legend dynamically based on selection
        if (viewSelected !== "Global") {
            crowdChart.data.datasets[0].label = `Zone ${viewSelected} Actual`;
            crowdChart.data.datasets[1].label = `Zone ${viewSelected} Forecast`;
        } else {
            crowdChart.data.datasets[0].label = 'Total Actual Crowd';
            crowdChart.data.datasets[1].label = 'Total Forecast (30s+)';
        }

        crowdChart.update();

        // Incident Logs Update
        const logsTbody = document.getElementById('logs-tbody');
        logsTbody.innerHTML = '';
        if (data.logs && data.logs.length > 0) {
            let criticalCount = 0;
            data.logs.forEach(log => {
                const tr = document.createElement('tr');
                if(log.level === 'red') tr.className = 'log-red';
                else if(log.level === 'yellow') tr.className = 'log-yellow';
                
                if(log.level === 'red' && log.message.includes('CRITICAL')) criticalCount++;
                
                tr.innerHTML = `
                    <td>${log.time}</td>
                    <td>${log.level.toUpperCase()}</td>
                    <td>${log.message}</td>
                `;
                logsTbody.appendChild(tr);
            });

            // Trigger Toasts & Sound on NEW critical events
            if (data.alerts && data.alerts.length > 0) {
                data.alerts.forEach(alertMsg => showToast(alertMsg));
                
                const alertsCurrently = data.alerts.length;
                const criticalAlertsActive = data.alerts.some(a => a.includes("CRITICAL") || a.includes("GLOBAL") || a.includes("WARNING"));
                if (criticalAlertsActive && alertsCurrently > lastAlertCount && soundEnabled) {
                    alertSound.play().catch(e => console.log("Audio play blocked by browser."));
                }
                lastAlertCount = alertsCurrently;
            } else {
                lastAlertCount = 0;
            }
        }
        
        // System Telemetry Update
        document.getElementById('system-latency').textContent = (data.latency || '---') + 'ms';
        document.getElementById('cams-online').textContent = data.cameras_online || '2/2';
        
        const engineStatusEl = document.getElementById('engine-status');
        if (document.getElementById('toggle-detection').checked) {
            engineStatusEl.textContent = 'RUNNING';
            engineStatusEl.className = 'status-ok';
        } else {
            engineStatusEl.textContent = 'OFFLINE';
            engineStatusEl.className = 'status-err';
        }

    } catch(e) {
        console.error("Error fetching data:", e);
    }
}

// Poll every 500ms
setInterval(fetchData, 500);
