/**
 * Controller script for the Extension Options Page.
 * Validates loopback service URL, stores credentials securely in browser.storage.local,
 * and verifies live connectivity to the local service.
 */

const webExt = typeof browser !== 'undefined' ? browser : chrome;

// DOM Elements
const serviceUrlInput = document.getElementById('service-url');
const authTokenInput = document.getElementById('auth-token');
const btnToggleToken = document.getElementById('btn-toggle-token');
const chkLayer2 = document.getElementById('chk-layer2');
const chkLayer4 = document.getElementById('chk-layer4');
const btnSave = document.getElementById('btn-save');
const btnTest = document.getElementById('btn-test');
const statusAlert = document.getElementById('status-alert');

function showAlert(message, type = 'info') {
  statusAlert.className = `status-alert ${type}`;
  statusAlert.textContent = message;
  statusAlert.classList.remove('hidden');
}

function hideAlert() {
  statusAlert.classList.add('hidden');
}

/**
 * Validate that the service URL is strictly a loopback HTTP endpoint.
 * Remote endpoints are rejected to prevent leakage of browsing data.
 * @param {string} urlString
 * @returns {boolean}
 */
function isLoopbackAddress(urlString) {
  try {
    const parsed = new URL(urlString);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return false;
    }
    const host = parsed.hostname.toLowerCase();
    return (
      host === '127.0.0.1' ||
      host === 'localhost' ||
      host === '[::1]' ||
      host.startsWith('127.')
    );
  } catch {
    return false;
  }
}

async function loadOptions() {
  try {
    const stored = await webExt.storage.local.get([
      'serviceUrl',
      'token',
      'runLayer2',
      'runLayer4',
    ]);

    serviceUrlInput.value = stored.serviceUrl || 'http://127.0.0.1:8000';
    authTokenInput.value = stored.token || '';
    chkLayer2.checked = stored.runLayer2 !== false;
    chkLayer4.checked = stored.runLayer4 !== false;
  } catch (err) {
    showAlert(`Failed to load options: ${err.message}`, 'error');
  }
}

async function saveOptions() {
  hideAlert();
  const rawUrl = serviceUrlInput.value.trim();
  const token = authTokenInput.value.trim();
  const runLayer2 = chkLayer2.checked;
  const runLayer4 = chkLayer4.checked;

  if (!rawUrl) {
    showAlert('Service address cannot be empty.', 'error');
    return;
  }

  if (!isLoopbackAddress(rawUrl)) {
    showAlert(
      'Security Policy Violation: Service address must be a loopback HTTP endpoint (e.g. http://127.0.0.1:8000 or http://localhost:8000). Remote addresses are not permitted to protect browsing privacy.',
      'error'
    );
    return;
  }

  try {
    await webExt.storage.local.set({
      serviceUrl: rawUrl,
      token,
      runLayer2,
      runLayer4,
    });

    // Notify background script to refresh its configuration
    await webExt.runtime.sendMessage({ type: 'UPDATE_CONFIG' });

    showAlert('Settings saved successfully. Background client reloaded.', 'success');
  } catch (err) {
    showAlert(`Failed to save settings: ${err.message}`, 'error');
  }
}

async function testConnection() {
  hideAlert();
  const rawUrl = serviceUrlInput.value.trim();
  const token = authTokenInput.value.trim();

  if (!isLoopbackAddress(rawUrl)) {
    showAlert('Service address must be a loopback endpoint (127.0.0.1 or localhost).', 'error');
    return;
  }

  btnTest.disabled = true;
  btnTest.textContent = 'Testing...';

  const client = new ApiClient({ serviceUrl: rawUrl, token });

  try {
    // 1. Health check
    const health = await client.checkHealth();
    if (!health.ok) {
      showAlert(
        `Health check failed: ${health.error}. Please ensure "python -m detector_service" is running.`,
        'error'
      );
      return;
    }

    // 2. Authentication and scan verification (offline lexical dry run)
    const scanTest = await client.analyzeUrl('http://127.0.0.1/test-auth-probe', {
      runLayer2: false,
      runLayer4: false,
    });

    if (scanTest.ok) {
      showAlert(
        `Connection verified! Local detector v${health.data.version} is online, ML model is loaded (schema v${health.data.schema_version}), and authentication succeeded.`,
        'success'
      );
    } else if (scanTest.status === 401) {
      showAlert(
        'Service is online, but authentication failed! Please verify the token copied from data/processed/service_token.',
        'error'
      );
    } else {
      showAlert(`Service reachable, but probe returned: ${scanTest.error}`, 'error');
    }
  } catch (err) {
    showAlert(`Connection test failed: ${err.message}`, 'error');
  } finally {
    btnTest.disabled = false;
    btnTest.textContent = 'Test Connection';
  }
}

// Toggle token visibility
btnToggleToken.addEventListener('click', () => {
  if (authTokenInput.type === 'password') {
    authTokenInput.type = 'text';
    btnToggleToken.textContent = 'Hide';
  } else {
    authTokenInput.type = 'password';
    btnToggleToken.textContent = 'Show';
  }
});

btnSave.addEventListener('click', saveOptions);
btnTest.addEventListener('click', testConnection);

document.addEventListener('DOMContentLoaded', loadOptions);
