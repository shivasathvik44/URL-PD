/**
 * Background Service Script for Phishing URL Detector Firefox Extension.
 * Observes top-level navigations, queries local detector service, caches results,
 * updates status badges, and redirects dangerous URLs to the warning interstitial.
 */

// Universal WebExtension API reference
const webExt = typeof browser !== 'undefined'
  ? browser
  : (typeof chrome !== 'undefined' ? chrome : {});

// Fallback loader for Node test environments
let AnalysisCacheClass = typeof AnalysisCache !== 'undefined' ? AnalysisCache : globalThis.AnalysisCache;
let InFlightTrackerClass = typeof InFlightTracker !== 'undefined' ? InFlightTracker : globalThis.InFlightTracker;
let ContinueOnceManagerClass = typeof ContinueOnceManager !== 'undefined' ? ContinueOnceManager : globalThis.ContinueOnceManager;
let ApiClientClass = typeof ApiClient !== 'undefined' ? ApiClient : globalThis.ApiClient;

if (typeof require !== 'undefined') {
  if (!AnalysisCacheClass) {
    const c = require('./cache.js');
    AnalysisCacheClass = c.AnalysisCache;
    InFlightTrackerClass = c.InFlightTracker;
    ContinueOnceManagerClass = c.ContinueOnceManager;
  }
  if (!ApiClientClass) {
    ApiClientClass = require('./api_client.js').ApiClient;
  }
}

// Initialize components
const cache = new AnalysisCacheClass();
const inFlight = new InFlightTrackerClass();
const continueOnce = new ContinueOnceManagerClass();
const apiClient = new ApiClientClass();

// In-memory tab status map: tabId -> { url, classification, risk_score, reasons, status, timestamp }
const tabState = new Map();
// Monotonic per-tab navigation sequence counter to defeat asynchronous race conditions
const tabNavSeq = new Map();

/**
 * Determine if a URL is eligible for phishing inspection.
 * Rejects non-HTTP(S) schemes, browser internals, and extension pages.
 * @param {string} urlString
 * @returns {boolean}
 */
function isEligibleUrl(urlString) {
  if (!urlString || typeof urlString !== 'string') return false;
  try {
    const parsed = new URL(urlString);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return false;
    }
    // Never inspect extension internal pages (e.g. warning.html or popup)
    const extBase = (webExt.runtime && typeof webExt.runtime.getURL === 'function')
      ? webExt.runtime.getURL('')
      : '';
    if (extBase && urlString.startsWith(extBase)) {
      return false;
    }
    if (urlString.startsWith('moz-extension://') || urlString.startsWith('chrome-extension://')) {
      return false;
    }
    return true;
  } catch {
    return false;
  }
}

/**
 * Set the action badge on a specific tab.
 * @param {number} tabId
 * @param {string} text
 * @param {string} color
 */
async function updateBadge(tabId, text, color) {
  try {
    if (webExt.action && webExt.action.setBadgeText) {
      await webExt.action.setBadgeText({ text, tabId });
      if (color && webExt.action.setBadgeBackgroundColor) {
        await webExt.action.setBadgeBackgroundColor({ color, tabId });
      }
    } else if (webExt.browserAction && webExt.browserAction.setBadgeText) {
      await webExt.browserAction.setBadgeText({ text, tabId });
      if (color && webExt.browserAction.setBadgeBackgroundColor) {
        await webExt.browserAction.setBadgeBackgroundColor({ color, tabId });
      }
    }
  } catch {
    // Tab may have closed before badge update
  }
}

/**
 * Handle scan results and redirect tab if classified as phishing.
 * @param {number} tabId
 * @param {string} url
 * @param {Object} result
 */
/**
 * Handle scan results and redirect tab if classified as phishing.
 * Rejects stale results if a newer navigation has occurred on the same tab.
 * @param {number} tabId
 * @param {string} url
 * @param {Object} result
 * @param {number} [seq] - Navigation sequence number when the scan was initiated
 */
async function handleScanResult(tabId, url, result, seq) {
  // 1. Guard against navigation sequence races (a newer navigation started on this tab)
  if (seq !== undefined && tabNavSeq.get(tabId) !== seq) {
    return;
  }

  // 2. Verify tab still exists and has not navigated away to another URL
  try {
    const currentTab = await webExt.tabs.get(tabId);
    if (!currentTab || currentTab.url !== url) {
      return; // Navigation race condition: user already left this URL
    }
  } catch {
    return; // Tab closed or invalid
  }

  // 3. Re-verify sequence number after awaiting tabs.get
  if (seq !== undefined && tabNavSeq.get(tabId) !== seq) {
    return;
  }

  const classification = result.classification || 'uncertain';
  const riskScore = typeof result.risk_score === 'number' ? result.risk_score : 0.0;

  // 4. Safely store completed status in tabState
  tabState.set(tabId, {
    url,
    status: 'completed',
    classification,
    risk_score: riskScore,
    reasons: result.reasons || [],
    ml_probability: result.ml_probability,
    timings_ms: result.timings_ms || {},
    seq,
    timestamp: Date.now(),
  });

  // 5. Set visual badge
  if (classification === 'phishing') {
    await updateBadge(tabId, '!', '#EF4444'); // Red !
  } else if (classification === 'safe') {
    await updateBadge(tabId, 'OK', '#10B981'); // Green OK
  } else {
    await updateBadge(tabId, '?', '#F59E0B'); // Amber ?
  }

  // 6. Interstitial Warning Redirection
  if (classification === 'phishing') {
    // Check if user already acknowledged bypass for this specific navigation
    if (continueOnce.has(tabId, url)) {
      return;
    }

    const warningPage = webExt.runtime.getURL(
      `warning/warning.html?target=${encodeURIComponent(url)}`
    );

    try {
      await webExt.tabs.update(tabId, { url: warningPage });
    } catch (err) {
      console.error('Failed to redirect to warning page:', err.message);
    }
  }
}

/**
 * Handle analysis error (service down, timeout, auth failed).
 * Rejects stale error updates if a newer navigation has occurred.
 * @param {number} tabId
 * @param {string} url
 * @param {Object} errResult
 * @param {number} [seq] - Navigation sequence number
 */
async function handleScanError(tabId, url, errResult, seq) {
  if (seq !== undefined && tabNavSeq.get(tabId) !== seq) {
    return;
  }

  try {
    const currentTab = await webExt.tabs.get(tabId);
    if (!currentTab || currentTab.url !== url) {
      return;
    }
  } catch {
    return;
  }

  if (seq !== undefined && tabNavSeq.get(tabId) !== seq) {
    return;
  }

  tabState.set(tabId, {
    url,
    status: 'error',
    classification: 'unavailable',
    error: errResult.error || 'Detector service unavailable',
    seq,
    timestamp: Date.now(),
  });
  await updateBadge(tabId, '', '#9CA3AF');
}

/**
 * Top-level navigation listener.
 * Observes top-level frames only (frameId === 0).
 */
async function handleNavigation(details) {
  // 1. Only top-level frame navigations
  if (details.frameId !== 0) return;

  const url = details.url;
  const tabId = details.tabId;

  // Advance sequence number for this tab to invalidate in-flight older scans
  const seq = (tabNavSeq.get(tabId) || 0) + 1;
  tabNavSeq.set(tabId, seq);

  // 2. Reject ineligible schemes and internal URLs
  if (!isEligibleUrl(url)) {
    tabState.delete(tabId);
    await updateBadge(tabId, '', '#9CA3AF');
    return;
  }

  // 3. Check continue-once token (if user clicked 'Continue to website' on warning page)
  if (continueOnce.consume(tabId, url)) {
    // User explicitly bypassed warning for this single navigation
    await updateBadge(tabId, '!', '#EF4444');
    return;
  }

  // 4. Set scanning status
  tabState.set(tabId, { url, status: 'scanning', seq, timestamp: Date.now() });
  await updateBadge(tabId, '...', '#6B7280');

  // 5. Query local memory cache
  const cached = cache.get(url);
  if (cached) {
    await handleScanResult(tabId, url, cached, seq);
    return;
  }

  // 6. In-flight request deduplication & dispatch
  try {
    const res = await inFlight.run(url, () => apiClient.analyzeUrl(url));
    if (res.ok && res.data) {
      cache.set(url, res.data);
      await handleScanResult(tabId, url, res.data, seq);
    } else {
      await handleScanError(tabId, url, res, seq);
    }
  } catch (err) {
    await handleScanError(tabId, url, { error: err.message }, seq);
  }
}

/**
 * Clean up tab state when tab is closed.
 */
function handleTabRemoved(tabId) {
  tabState.delete(tabId);
  tabNavSeq.delete(tabId);
}

/**
 * Message handler for popup and warning page communication.
 */
if (webExt.runtime && webExt.runtime.onMessage && typeof webExt.runtime.onMessage.addListener === 'function') {
  webExt.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (!message || typeof message !== 'object') return false;

    if (message.type === 'GET_TAB_STATUS') {
      const tabId = message.tabId;
      const state = tabState.get(tabId) || { status: 'idle' };
      sendResponse({ ok: true, state });
      return false;
    }

    if (message.type === 'GET_ANALYSIS') {
      const url = message.url;
      const cached = cache.get(url);
      if (cached) {
        sendResponse({ ok: true, data: cached });
      } else {
        // Find in tabState if available
        for (const st of tabState.values()) {
          if (st.url === url && st.status === 'completed') {
            sendResponse({ ok: true, data: st });
            return false;
          }
        }
        sendResponse({ ok: false, error: 'No analysis found for this URL' });
      }
      return false;
    }

    if (message.type === 'ACKNOWLEDGE_WARNING') {
      const { url, tabId } = message;
      if (url && typeof url === 'string') {
        continueOnce.acknowledge(tabId, url);
        sendResponse({ ok: true });
      } else {
        sendResponse({ ok: false, error: 'Invalid URL' });
      }
      return false;
    }

    if (message.type === 'RESCAN_TAB') {
      const { tabId, url } = message;
      if (isEligibleUrl(url)) {
        const seq = (tabNavSeq.get(tabId) || 0) + 1;
        tabNavSeq.set(tabId, seq);
        tabState.set(tabId, { url, status: 'scanning', seq, timestamp: Date.now() });
        updateBadge(tabId, '...', '#6B7280');
        apiClient.analyzeUrl(url)
          .then((res) => {
            if (res.ok && res.data) {
              cache.set(url, res.data);
              handleScanResult(tabId, url, res.data, seq);
            } else {
              handleScanError(tabId, url, res, seq);
            }
          })
          .catch((err) => handleScanError(tabId, url, { error: err.message }, seq));
        sendResponse({ ok: true });
      } else {
        sendResponse({ ok: false, error: 'Ineligible URL' });
      }
      return false;
    }

    if (message.type === 'UPDATE_CONFIG') {
      loadSettings().then(() => sendResponse({ ok: true }));
      return true; // Async response
    }

    return false;
  });
}

/**
 * Load user settings from extension storage.
 */
async function loadSettings() {
  try {
    if (webExt.storage && webExt.storage.local && typeof webExt.storage.local.get === 'function') {
      const stored = await webExt.storage.local.get(['serviceUrl', 'token']);
      apiClient.updateConfig({
        serviceUrl: stored.serviceUrl || 'http://127.0.0.1:8000',
        token: stored.token || '',
      });
    }
  } catch (err) {
    console.error('Failed to load settings:', err.message);
  }
}

// Subscribe to browser navigation events
if (webExt.webNavigation && webExt.webNavigation.onCommitted && typeof webExt.webNavigation.onCommitted.addListener === 'function') {
  webExt.webNavigation.onCommitted.addListener(handleNavigation);
}
if (webExt.tabs && webExt.tabs.onRemoved && typeof webExt.tabs.onRemoved.addListener === 'function') {
  webExt.tabs.onRemoved.addListener(handleTabRemoved);
}

// Initial settings load
loadSettings();

// Export functions for testability in Node
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    isEligibleUrl,
    handleNavigation,
    handleScanResult,
    handleScanError,
    handleTabRemoved,
    tabState,
    tabNavSeq,
    cache,
    inFlight,
    continueOnce,
    apiClient,
    webExt,
    loadSettings,
  };
}
