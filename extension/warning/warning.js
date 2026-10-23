/**
 * Controller script for the Interstitial Warning page.
 * Uses safe DOM manipulation (never innerHTML) to prevent DOM-based XSS
 * from untrusted URLs or detector messages.
 */

const webExt = typeof browser !== 'undefined'
  ? browser
  : (typeof chrome !== 'undefined' ? chrome : {});

// DOM Elements (safely resolved in browser context)
const getEl = (id) => (typeof document !== 'undefined' ? document.getElementById(id) : null);
const targetDomainEl = getEl('target-domain');
const targetUrlEl = getEl('target-url');
const classificationBadgeEl = getEl('classification-badge');
const riskScoreEl = getEl('risk-score');
const reasonsListEl = getEl('reasons-list');
const btnBack = getEl('btn-back');
const btnToggleAdvanced = getEl('btn-toggle-advanced');
const advancedPanel = getEl('advanced-panel');
const chkAcknowledge = getEl('chk-acknowledge');
const btnContinue = getEl('btn-continue');

let targetUrl = '';
let currentTabId = null;

/**
 * Validate that a target parameter is a well-formed HTTP/HTTPS URL.
 * Prevents javascript: or data: URIs from triggering malicious redirects.
 * @param {string} rawUrl
 * @returns {string | null}
 */
function sanitizeTargetUrl(rawUrl) {
  if (!rawUrl || typeof rawUrl !== 'string') return null;
  try {
    const parsed = new URL(rawUrl);
    if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
      return parsed.href;
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Safely render a list of Reason objects using DOM text nodes.
 * @param {Array<{ code: string, message: string, severity: string }>} reasons
 */
function renderReasons(reasons) {
  reasonsListEl.textContent = ''; // Clear existing items

  if (!Array.isArray(reasons) || reasons.length === 0) {
    const li = document.createElement('li');
    li.className = 'reason-item';
    li.textContent = 'High probability of credential impersonation or deceptive domain structure.';
    reasonsListEl.appendChild(li);
    return;
  }

  for (const r of reasons) {
    if (!r || !r.message) continue;
    const li = document.createElement('li');
    li.className = 'reason-item';

    if (r.severity) {
      const sevSpan = document.createElement('span');
      const sevClass = `sev-${String(r.severity).toLowerCase()}`;
      sevSpan.className = `reason-severity ${sevClass}`;
      sevSpan.textContent = String(r.severity).toUpperCase();
      li.appendChild(sevSpan);
    }

    const textNode = document.createTextNode(` ${r.message}`);
    li.appendChild(textNode);
    reasonsListEl.appendChild(li);
  }
}

async function init() {
  const params = new URLSearchParams(window.location.search);
  const rawTarget = params.get('target');
  targetUrl = sanitizeTargetUrl(rawTarget);

  if (!targetUrl) {
    targetDomainEl.textContent = 'Unknown or Invalid URL';
    targetUrlEl.textContent = 'No valid destination was specified.';
    btnContinue.disabled = true;
    return;
  }

  // Display domain and URL safely via textContent
  try {
    const parsed = new URL(targetUrl);
    targetDomainEl.textContent = parsed.hostname;
    targetUrlEl.textContent = targetUrl;
  } catch {
    targetDomainEl.textContent = targetUrl;
  }

  // Securely resolve the current tab hosting this warning page
  currentTabId = await resolveWarningTabId(webExt);

  // If tab ID could not be identified safely, disable continue action and display error
  if (currentTabId === null) {
    if (btnContinue) btnContinue.disabled = true;
    if (chkAcknowledge) chkAcknowledge.disabled = true;
    if (advancedPanel) {
      const tabErr = document.createElement('p');
      tabErr.className = 'warning-text';
      tabErr.style.color = '#EF4444';
      tabErr.textContent = 'Unable to securely identify current browser tab. Continuing is disabled.';
      advancedPanel.appendChild(tabErr);
    }
  }

  // Fetch cached analysis result from background script
  try {
    const response = await webExt.runtime.sendMessage({
      type: 'GET_ANALYSIS',
      url: targetUrl,
    });

    if (response && response.ok && response.data) {
      const data = response.data;
      if (data.classification) {
        classificationBadgeEl.textContent = String(data.classification).toUpperCase();
      }
      if (typeof data.risk_score === 'number') {
        riskScoreEl.textContent = data.risk_score.toFixed(2);
      }
      renderReasons(data.reasons);
    } else {
      renderReasons([]);
    }
  } catch {
    renderReasons([]);
  }
}

/**
 * Securely identify the tab containing this warning page.
 * Uses tabs.getCurrent() to avoid misidentifying background tabs.
 * Does NOT fall back to tabs.query({ active: true }) which would resolve the wrong tab for background tabs.
 * @param {Object} [api] - WebExtension API reference (defaults to webExt)
 * @returns {Promise<number | null>}
 */
async function resolveWarningTabId(api = webExt) {
  try {
    if (api && api.tabs && typeof api.tabs.getCurrent === 'function') {
      const tab = await api.tabs.getCurrent();
      if (tab && typeof tab.id === 'number') {
        return tab.id;
      }
    }
    return null;
  } catch (err) {
    console.error('Failed to get current tab via tabs.getCurrent():', err);
    return null;
  }
}

// Button actions and event listeners (guarded for non-DOM test environments)
if (typeof document !== 'undefined') {
  if (btnBack) {
    btnBack.addEventListener('click', () => {
      if (window.history && window.history.length > 1) {
        window.history.back();
      } else {
        window.location.replace('about:blank');
      }
    });
  }

  if (btnToggleAdvanced && advancedPanel) {
    btnToggleAdvanced.addEventListener('click', () => {
      const isHidden = advancedPanel.classList.contains('hidden');
      if (isHidden) {
        advancedPanel.classList.remove('hidden');
        btnToggleAdvanced.textContent = 'Hide Options ▲';
      } else {
        advancedPanel.classList.add('hidden');
        btnToggleAdvanced.textContent = 'Advanced Options ▼';
      }
    });
  }

  if (chkAcknowledge && btnContinue) {
    chkAcknowledge.addEventListener('change', () => {
      btnContinue.disabled = !chkAcknowledge.checked || !targetUrl || currentTabId === null;
    });
  }

  if (btnContinue) {
    btnContinue.addEventListener('click', async () => {
      if (!targetUrl || !chkAcknowledge || !chkAcknowledge.checked || currentTabId === null) return;

      btnContinue.disabled = true;
      btnContinue.textContent = 'Proceeding...';

      try {
        await webExt.runtime.sendMessage({
          type: 'ACKNOWLEDGE_WARNING',
          url: targetUrl,
          tabId: currentTabId,
        });
      } catch (err) {
        console.error('Failed to register acknowledgment:', err);
      }

      window.location.replace(targetUrl);
    });
  }

  document.addEventListener('DOMContentLoaded', init);
}

// Export functions for testability in Node
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    sanitizeTargetUrl,
    renderReasons,
    resolveWarningTabId,
    init,
  };
}
