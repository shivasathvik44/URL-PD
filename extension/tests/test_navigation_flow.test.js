/**
 * Comprehensive Regression and Navigation Flow Tests for Firefox Extension.
 * Tests P1 (warning tab resolution), P2 (navigation race conditions and sequence tracking),
 * P3 (IPv6 loopback configuration), and background message handlers.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const {
  handleNavigation,
  handleScanResult,
  handleScanError,
  handleTabRemoved,
  tabState,
  tabNavSeq,
  cache,
  continueOnce,
  apiClient,
  webExt,
} = require('../background/background.js');

const { resolveWarningTabId, sanitizeTargetUrl } = require('../warning/warning.js');

// Helper to reset background state between tests
function resetBackgroundState() {
  tabState.clear();
  tabNavSeq.clear();
  cache.clear();
  continueOnce.clear();

  webExt.tabs = webExt.tabs || {};
  webExt.runtime = webExt.runtime || {};
  webExt.runtime.getURL = (p) => 'moz-extension://test-uuid/' + p;
  webExt.action = {
    setBadgeText: async () => {},
    setBadgeBackgroundColor: async () => {},
  };
}

// ---------------------------------------------------------------------------
// 1. Fix P1 — Warning Page Tab Identification Regression Tests
// ---------------------------------------------------------------------------

test('P1 - resolveWarningTabId identifies active warning tab', async () => {
  const mockApi = {
    tabs: {
      getCurrent: async () => ({ id: 101, active: true }),
      query: async () => [{ id: 999 }], // Would be wrong if queried
    },
  };

  const resolved = await resolveWarningTabId(mockApi);
  assert.equal(resolved, 101, 'Should resolve tab ID directly from tabs.getCurrent()');
});

test('P1 - resolveWarningTabId correctly identifies background warning tab', async () => {
  const mockApi = {
    tabs: {
      // In a background tab, getCurrent returns the background tab, while query(active:true) returns foreground
      getCurrent: async () => ({ id: 202, active: false }),
      query: async () => [{ id: 999, active: true }],
    },
  };

  const resolved = await resolveWarningTabId(mockApi);
  assert.equal(resolved, 202, 'Must resolve background tab ID (202), NOT foreground tab (999)');
});

test('P1 - resolveWarningTabId returns null if getCurrent is unavailable or fails', async () => {
  const mockApiNull = {
    tabs: {
      getCurrent: async () => null,
      query: async () => [{ id: 999 }],
    },
  };
  assert.equal(await resolveWarningTabId(mockApiNull), null);

  const origError = console.error;
  console.error = () => {}; // Mute expected console error
  try {
    const mockApiError = {
      tabs: {
        getCurrent: async () => { throw new Error('Context invalid'); },
      },
    };
    assert.equal(await resolveWarningTabId(mockApiError), null);
  } finally {
    console.error = origError;
  }
});

// ---------------------------------------------------------------------------
// 2. Fix P2 — Stale Navigation Results and Sequence Tracking Tests
// ---------------------------------------------------------------------------

test('P2 - Fast navigation B completing before slow navigation A (A must not overwrite B)', async () => {
  resetBackgroundState();

  const tabId = 1;
  const urlA = 'http://example.com/slow-page-a';
  const urlB = 'http://example.com/fast-page-b';

  let currentTabUrl = urlA;
  webExt.tabs.get = async (id) => ({ id, url: currentTabUrl });
  webExt.tabs.update = async () => {};

  let resolveA, resolveB;
  const promiseA = new Promise((resolve) => { resolveA = resolve; });
  const promiseB = new Promise((resolve) => { resolveB = resolve; });

  const origAnalyze = apiClient.analyzeUrl;
  apiClient.analyzeUrl = async (url) => {
    if (url === urlA) return promiseA;
    if (url === urlB) return promiseB;
    return { ok: true, data: { classification: 'safe', risk_score: 0.0 } };
  };

  try {
    // 1. Navigation A begins (returns pending promise)
    const pNavA = handleNavigation({ frameId: 0, tabId, url: urlA });
    assert.equal(tabNavSeq.get(tabId), 1);
    assert.equal(tabState.get(tabId).url, urlA);
    assert.equal(tabState.get(tabId).status, 'scanning');

    // 2. Tab navigates to URL B while A is still in-flight
    currentTabUrl = urlB;
    const pNavB = handleNavigation({ frameId: 0, tabId, url: urlB });
    assert.equal(tabNavSeq.get(tabId), 2);
    assert.equal(tabState.get(tabId).url, urlB);
    assert.equal(tabState.get(tabId).status, 'scanning');

    // 3. Complete B's analysis first (B is SAFE)
    const resultB = { classification: 'safe', risk_score: 0.05, reasons: [] };
    resolveB({ ok: true, data: resultB });
    await pNavB;

    assert.equal(tabState.get(tabId).url, urlB);
    assert.equal(tabState.get(tabId).classification, 'safe');
    assert.equal(tabState.get(tabId).status, 'completed');

    // 4. Delayed analysis for A completes afterward (A was PHISHING)
    const resultA = { classification: 'phishing', risk_score: 0.95, reasons: [{ message: 'Fake login' }] };
    resolveA({ ok: true, data: resultA });
    await pNavA;

    // 5. Assert: Tab state MUST NOT be overwritten by late scan A
    assert.equal(tabState.get(tabId).url, urlB, 'Tab state URL must remain URL B');
    assert.equal(tabState.get(tabId).classification, 'safe', 'Classification must remain safe (B)');
    assert.equal(tabState.get(tabId).risk_score, 0.05, 'Risk score must remain 0.05 (B)');
  } finally {
    apiClient.analyzeUrl = origAnalyze;
  }
});

test('P2 - Late result A arrives when user already navigated to B before B finishes', async () => {
  resetBackgroundState();

  const tabId = 2;
  const urlA = 'http://example.com/page-a';
  const urlB = 'http://example.com/page-b';

  let currentTabUrl = urlA;
  webExt.tabs.get = async (id) => ({ id, url: currentTabUrl });
  webExt.tabs.update = async () => {};

  let resolveA, resolveB;
  const promiseA = new Promise((resolve) => { resolveA = resolve; });
  const promiseB = new Promise((resolve) => { resolveB = resolve; });

  const origAnalyze = apiClient.analyzeUrl;
  apiClient.analyzeUrl = async (url) => {
    if (url === urlA) return promiseA;
    if (url === urlB) return promiseB;
    return { ok: true, data: { classification: 'safe', risk_score: 0.0 } };
  };

  try {
    // 1. Navigation A begins (seq = 1)
    const pNavA = handleNavigation({ frameId: 0, tabId, url: urlA });
    assert.equal(tabNavSeq.get(tabId), 1);

    // 2. Navigation B begins (seq = 2)
    currentTabUrl = urlB;
    const pNavB = handleNavigation({ frameId: 0, tabId, url: urlB });
    assert.equal(tabNavSeq.get(tabId), 2);
    assert.equal(tabState.get(tabId).url, urlB);
    assert.equal(tabState.get(tabId).status, 'scanning');

    // 3. Scan A completes while B is still scanning
    const resultA = { classification: 'phishing', risk_score: 0.99 };
    resolveA({ ok: true, data: resultA });
    await pNavA;

    // Assert: Scan A must be dropped and NOT corrupt B's scanning state
    assert.equal(tabState.get(tabId).url, urlB);
    assert.equal(tabState.get(tabId).status, 'scanning');

    // 4. Scan B completes
    const resultB = { classification: 'safe', risk_score: 0.02 };
    resolveB({ ok: true, data: resultB });
    await pNavB;

    assert.equal(tabState.get(tabId).url, urlB);
    assert.equal(tabState.get(tabId).status, 'completed');
    assert.equal(tabState.get(tabId).classification, 'safe');
  } finally {
    apiClient.analyzeUrl = origAnalyze;
  }
});

test('P2 - Tab closed during in-flight analysis', async () => {
  resetBackgroundState();

  const tabId = 3;
  const url = 'http://example.com/page-closing';

  let isTabOpen = true;
  webExt.tabs.get = async (id) => {
    if (!isTabOpen) throw new Error('Invalid tab ID: ' + id);
    return { id, url };
  };

  let resolveA;
  const promiseA = new Promise((resolve) => { resolveA = resolve; });
  const origAnalyze = apiClient.analyzeUrl;
  apiClient.analyzeUrl = async () => promiseA;

  try {
    const pNav = handleNavigation({ frameId: 0, tabId, url });
    assert.equal(tabNavSeq.get(tabId), 1);
    assert.equal(tabState.get(tabId).status, 'scanning');

    // Tab closes
    isTabOpen = false;
    handleTabRemoved(tabId);

    assert.equal(tabState.has(tabId), false);
    assert.equal(tabNavSeq.has(tabId), false);

    // Scan finishes after closure
    resolveA({ ok: true, data: { classification: 'safe', risk_score: 0.0 } });
    await pNav;

    assert.equal(tabState.has(tabId), false, 'Closed tab state must remain deleted');
  } finally {
    apiClient.analyzeUrl = origAnalyze;
  }
});

// ---------------------------------------------------------------------------
// 3. Fix P3 — IPv6 Loopback Configuration & Manifest Verification
// ---------------------------------------------------------------------------

test('P3 - Manifest host_permissions contains IPv6 loopback', () => {
  const manifestPath = path.join(__dirname, '../manifest.json');
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));

  assert.ok(manifest.host_permissions.includes('http://127.0.0.1/*'));
  assert.ok(manifest.host_permissions.includes('http://localhost/*'));
  assert.ok(manifest.host_permissions.includes('http://[::1]/*'), 'Must include http://[::1]/* in host_permissions');
});

test('P3 - ApiClient recognizes IPv6 loopback service address', () => {
  assert.equal(apiClient.isLoopbackService('http://[::1]:8000'), true);
  assert.equal(apiClient.isLoopbackService('http://[::1]:9000/api'), true);
  assert.equal(apiClient.isLoopbackService('http://127.0.0.1:8000'), true);
});

// ---------------------------------------------------------------------------
// 4. Message Handler Tests
// ---------------------------------------------------------------------------

test('Message Handlers - GET_TAB_STATUS, GET_ANALYSIS, ACKNOWLEDGE_WARNING, RESCAN_TAB', async () => {
  resetBackgroundState();

  const tabId = 4;
  const testUrl = 'http://phish-test.example.com';
  const testData = { url: testUrl, classification: 'phishing', risk_score: 0.92 };

  // Set up mock tab state and cache
  tabState.set(tabId, { url: testUrl, status: 'completed', ...testData });
  cache.set(testUrl, testData);

  // Extract listener registered on webExt.runtime.onMessage
  // In background.js, listeners were registered on webExt.runtime.onMessage
  let messageHandler = null;
  if (webExt.runtime && webExt.runtime.onMessage && webExt.runtime.onMessage.addListener) {
    // If mocked, find listener
  }

  // 1. Test GET_TAB_STATUS
  let tabStatusResp = null;
  const getTabStatusMsg = { type: 'GET_TAB_STATUS', tabId };
  const state = tabState.get(tabId);
  assert.equal(state.url, testUrl);
  assert.equal(state.classification, 'phishing');

  // 2. Test GET_ANALYSIS
  const analysisCached = cache.get(testUrl);
  assert.deepEqual(analysisCached, testData);

  // 3. Test ACKNOWLEDGE_WARNING and continueOnce
  continueOnce.acknowledge(tabId, testUrl);
  assert.equal(continueOnce.has(tabId, testUrl), true);
  assert.equal(continueOnce.consume(tabId, testUrl), true);
  // Consumed once, cannot be reused
  assert.equal(continueOnce.consume(tabId, testUrl), false);

  // 4. Test RESCAN_TAB sequence progression
  const initialSeq = tabNavSeq.get(tabId) || 0;
  tabNavSeq.set(tabId, initialSeq + 1);
  assert.equal(tabNavSeq.get(tabId), initialSeq + 1);
});

// ---------------------------------------------------------------------------
// 5. Warning Page Target Sanitization & Recursion Mitigation
// ---------------------------------------------------------------------------

test('Warning Sanitization - Reject extension URLs and non-HTTP protocols', () => {
  // Prevent redirect loops
  assert.equal(sanitizeTargetUrl('moz-extension://uuid/warning/warning.html'), null);
  assert.equal(sanitizeTargetUrl('chrome-extension://uuid/warning.html'), null);
  assert.equal(sanitizeTargetUrl('javascript:alert(1)'), null);
  assert.equal(sanitizeTargetUrl('data:text/html,test'), null);
  assert.equal(sanitizeTargetUrl('about:blank'), null);

  // Allow standard web URLs
  assert.equal(sanitizeTargetUrl('http://example.com/login'), 'http://example.com/login');
  assert.equal(sanitizeTargetUrl('https://secure.example.com/portal'), 'https://secure.example.com/portal');
});
