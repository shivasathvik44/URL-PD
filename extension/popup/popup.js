/**
 * Controller script for the Extension Toolbar Popup.
 * Queries background state, checks local service health, and renders
 * active tab risk verdicts safely using textContent.
 */

const webExt = typeof browser !== 'undefined' ? browser : chrome;

// DOM Elements
const serviceDot = document.getElementById('service-dot');
const serviceStatusText = document.getElementById('service-status-text');
const tabUrlText = document.getElementById('tab-url-text');
const verdictBadge = document.getElementById('verdict-badge');
const riskScoreText = document.getElementById('risk-score-text');
const popupReasonsList = document.getElementById('popup-reasons-list');
const btnOpenOptions = document.getElementById('btn-open-options');
const btnRescan = document.getElementById('btn-rescan');

let currentTab = null;

async function checkServiceHealth() {
  const stored = await webExt.storage.local.get(['serviceUrl', 'token']);
  const serviceUrl = stored.serviceUrl || 'http://127.0.0.1:8000';
  const token = stored.token || '';

  const client = new ApiClient({ serviceUrl, token });
  const health = await client.checkHealth();

  if (health.ok && health.data) {
    serviceDot.className = 'status-dot status-connected';
    serviceStatusText.textContent = `Online (v${health.data.version}, Model Ready)`;
  } else {
    serviceDot.className = 'status-dot status-offline';
    serviceStatusText.textContent = health.error || 'Service Offline';
  }
}

async function renderTabStatus() {
  const tabs = await webExt.tabs.query({ active: true, currentWindow: true });
  if (!tabs || !tabs[0]) {
    tabUrlText.textContent = 'No active tab';
    return;
  }

  currentTab = tabs[0];
  const url = currentTab.url || '';

  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    tabUrlText.textContent = url.startsWith('about:') ? 'Firefox Internal Page' : 'Internal Page (Not Scanned)';
    verdictBadge.className = 'verdict-badge badge-neutral';
    verdictBadge.textContent = 'N/A';
    riskScoreText.textContent = '—';
    btnRescan.disabled = true;
    renderReasons([]);
    return;
  }

  // Display URL
  try {
    const parsed = new URL(url);
    tabUrlText.textContent = `${parsed.hostname}${parsed.pathname}`;
  } catch {
    tabUrlText.textContent = url;
  }

  // Ask background for latest tab state
  try {
    const response = await webExt.runtime.sendMessage({
      type: 'GET_TAB_STATUS',
      tabId: currentTab.id,
    });

    if (response && response.ok && response.state) {
      const state = response.state;

      if (state.status === 'scanning') {
        verdictBadge.className = 'verdict-badge badge-neutral';
        verdictBadge.textContent = 'SCANNING...';
        riskScoreText.textContent = '...';
        renderReasons([]);
        return;
      }

      if (state.status === 'error') {
        verdictBadge.className = 'verdict-badge badge-neutral';
        verdictBadge.textContent = 'ERROR';
        riskScoreText.textContent = '—';
        renderReasons([{ message: state.error || 'Analysis failed' }]);
        return;
      }

      const classification = state.classification || 'unknown';
      if (classification === 'phishing') {
        verdictBadge.className = 'verdict-badge badge-danger';
        verdictBadge.textContent = 'PHISHING';
      } else if (classification === 'safe') {
        verdictBadge.className = 'verdict-badge badge-success';
        verdictBadge.textContent = 'SAFE';
      } else {
        verdictBadge.className = 'verdict-badge badge-warning';
        verdictBadge.textContent = 'UNCERTAIN';
      }

      if (typeof state.risk_score === 'number') {
        riskScoreText.textContent = state.risk_score.toFixed(2);
      } else {
        riskScoreText.textContent = '—';
      }

      renderReasons(state.reasons || []);
    } else {
      verdictBadge.className = 'verdict-badge badge-neutral';
      verdictBadge.textContent = 'NOT SCANNED';
      riskScoreText.textContent = '—';
      renderReasons([]);
    }
  } catch (err) {
    console.error('Failed to get tab status:', err);
  }
}

function renderReasons(reasons) {
  popupReasonsList.textContent = '';

  if (!reasons || reasons.length === 0) {
    const li = document.createElement('li');
    li.className = 'reasons-empty';
    li.textContent = 'No security issues detected.';
    popupReasonsList.appendChild(li);
    return;
  }

  for (const r of reasons.slice(0, 3)) { // Show top 3 in popup
    const li = document.createElement('li');
    li.textContent = r.message || String(r);
    popupReasonsList.appendChild(li);
  }
}

// Button listeners
btnOpenOptions.addEventListener('click', () => {
  if (webExt.runtime.openOptionsPage) {
    webExt.runtime.openOptionsPage();
  } else {
    window.open(webExt.runtime.getURL('options/options.html'));
  }
});

btnRescan.addEventListener('click', async () => {
  if (!currentTab || !currentTab.url) return;
  btnRescan.disabled = true;
  btnRescan.textContent = 'Scanning...';

  try {
    await webExt.runtime.sendMessage({
      type: 'RESCAN_TAB',
      tabId: currentTab.id,
      url: currentTab.url,
    });
    // Poll briefly for updated status
    setTimeout(async () => {
      await renderTabStatus();
      btnRescan.disabled = false;
      btnRescan.textContent = 'Rescan Page';
    }, 800);
  } catch (err) {
    btnRescan.disabled = false;
    btnRescan.textContent = 'Rescan Page';
  }
});

// Initial load
document.addEventListener('DOMContentLoaded', async () => {
  await Promise.all([checkServiceHealth(), renderTabStatus()]);
});
