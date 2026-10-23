/**
 * Automated Unit Tests for ApiClient.
 * Uses Node's built-in node:test and node:assert.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const { ApiClient } = require('../background/api_client.js');

test('ApiClient - Loopback endpoint validation', () => {
  const client = new ApiClient({ serviceUrl: 'http://127.0.0.1:8000' });
  assert.equal(client.isLoopbackService('http://127.0.0.1:8000'), true);
  assert.equal(client.isLoopbackService('http://localhost:8000'), true);
  assert.equal(client.isLoopbackService('http://127.0.0.2:9000'), true);
  assert.equal(client.isLoopbackService('http://[::1]:8000'), true);

  // Prohibited external / remote addresses
  assert.equal(client.isLoopbackService('http://example.com'), false);
  assert.equal(client.isLoopbackService('https://api.phishing-detector.org'), false);
  assert.equal(client.isLoopbackService('http://192.168.1.1:8000'), false);
  assert.equal(client.isLoopbackService('http://10.0.0.1:8000'), false);
  assert.equal(client.isLoopbackService('javascript:alert(1)'), false);
  assert.equal(client.isLoopbackService('not-a-url'), false);
});

test('ApiClient - Header construction and token isolation', () => {
  const client = new ApiClient({
    serviceUrl: 'http://127.0.0.1:8000',
    token: 'secret-token-xyz',
  });

  const headers = client.getHeaders();
  assert.equal(headers['X-Detector-Client'], 'firefox-extension');
  assert.equal(headers['Authorization'], 'Bearer secret-token-xyz');
  assert.equal(headers['X-Detector-Token'], 'secret-token-xyz');
  assert.equal(headers['Content-Type'], 'application/json');

  // Refusal to send token to non-loopback endpoints
  const externalClient = new ApiClient({
    serviceUrl: 'https://evil-third-party.example.com',
    token: 'secret-token-xyz',
  });
  assert.throws(
    () => externalClient.getHeaders(),
    /Refusing to send headers to non-loopback/
  );
});

test('ApiClient - Health check query with mocked fetch', async () => {
  const client = new ApiClient({ serviceUrl: 'http://127.0.0.1:8000' });

  // Mock global fetch
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    assert.equal(url, 'http://127.0.0.1:8000/api/v1/health');
    assert.equal(options.method, 'GET');
    assert.equal(options.headers?.Host, undefined, 'Must not attempt to set forbidden Host header');
    assert.equal(options.headers?.host, undefined);
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: 'ok',
        service: 'phishing-url-detector',
        version: '0.5.0',
        model_loaded: true,
        schema_version: 2,
      }),
    };
  };

  try {
    const res = await client.checkHealth();
    assert.equal(res.ok, true);
    assert.equal(res.data.status, 'ok');
    assert.equal(res.data.model_loaded, true);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('ApiClient - analyzeUrl successful submission with mock', async () => {
  const client = new ApiClient({
    serviceUrl: 'http://127.0.0.1:8000',
    token: 'my-token',
  });

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    assert.equal(url, 'http://127.0.0.1:8000/api/v1/analyze');
    assert.equal(options.method, 'POST');
    const body = JSON.parse(options.body);
    assert.equal(body.url, 'http://paypal.com-verify.tk/login');
    assert.equal(options.headers['Authorization'], 'Bearer my-token');
    assert.equal(options.headers['X-Detector-Client'], 'firefox-extension');

    return {
      status: 200,
      json: async () => ({
        url: 'http://paypal.com-verify.tk/login',
        classification: 'phishing',
        risk_score: 0.94,
        reasons: [{ code: 'l1_brand', message: 'Brand in wrong place', severity: 'high' }],
      }),
    };
  };

  try {
    const res = await client.analyzeUrl('http://paypal.com-verify.tk/login');
    assert.equal(res.ok, true);
    assert.equal(res.data.classification, 'phishing');
    assert.equal(res.data.risk_score, 0.94);
    assert.equal(res.data.reasons.length, 1);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('ApiClient - Error status code mappings', async () => {
  const client = new ApiClient({ serviceUrl: 'http://127.0.0.1:8000' });
  const originalFetch = globalThis.fetch;

  try {
    // 401 Unauthorized
    globalThis.fetch = async () => ({ status: 401, json: async () => ({ detail: 'Unauthorized' }) });
    const res401 = await client.analyzeUrl('https://example.com');
    assert.equal(res401.ok, false);
    assert.equal(res401.status, 401);
    assert.match(res401.error, /Authentication failed/);

    // 403 Forbidden
    globalThis.fetch = async () => ({ status: 403, json: async () => ({ detail: 'Forbidden' }) });
    const res403 = await client.analyzeUrl('https://example.com');
    assert.equal(res403.ok, false);
    assert.equal(res403.status, 403);
    assert.match(res403.error, /Access forbidden/);

    // 422 Unprocessable Content
    globalThis.fetch = async () => ({ status: 422, json: async () => ({ detail: 'Invalid URL authority' }) });
    const res422 = await client.analyzeUrl('https://example.com');
    assert.equal(res422.ok, false);
    assert.equal(res422.status, 422);
    assert.match(res422.error, /Invalid URL authority/);

    // 503 Service Saturated
    globalThis.fetch = async () => ({ status: 503, json: async () => ({ detail: 'Saturated' }) });
    const res503 = await client.analyzeUrl('https://example.com');
    assert.equal(res503.ok, false);
    assert.equal(res503.status, 503);
    assert.match(res503.error, /saturated/);

    // Network connection failure
    globalThis.fetch = async () => { throw new Error('ECONNREFUSED'); };
    const resConn = await client.analyzeUrl('https://example.com');
    assert.equal(resConn.ok, false);
    assert.equal(resConn.status, 0);
    assert.match(resConn.error, /Detector service unavailable/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('ApiClient - Malformed response handling', async () => {
  const client = new ApiClient({ serviceUrl: 'http://127.0.0.1:8000' });
  const originalFetch = globalThis.fetch;

  try {
    // 200 OK but response is missing classification
    globalThis.fetch = async () => ({ status: 200, json: async () => ({ unexpected: true }) });
    const res = await client.analyzeUrl('https://example.com');
    assert.equal(res.ok, false);
    assert.match(res.error, /Malformed response/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
