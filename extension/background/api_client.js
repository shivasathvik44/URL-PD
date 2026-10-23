/**
 * API Client for Phishing URL Detector Local Service.
 * Communicates with the local FastAPI backend with strict loopback validation,
 * authentication headers, timeout management, and robust error classification.
 */

class ApiClient {
  /**
   * @param {Object} options
   * @param {string} options.serviceUrl - Base URL of local service (default http://127.0.0.1:8000).
   * @param {string} options.token - Bearer authentication token.
   * @param {number} options.timeoutMs - Timeout for network requests in ms (default 15000).
   */
  constructor({ serviceUrl = 'http://127.0.0.1:8000', token = '', timeoutMs = 15000 } = {}) {
    this.serviceUrl = serviceUrl.replace(/\/+$/, '');
    this.token = token ? token.trim() : '';
    this.timeoutMs = timeoutMs;
  }

  updateConfig({ serviceUrl, token, timeoutMs }) {
    if (serviceUrl !== undefined) this.serviceUrl = serviceUrl.replace(/\/+$/, '');
    if (token !== undefined) this.token = token ? token.trim() : '';
    if (timeoutMs !== undefined) this.timeoutMs = timeoutMs;
  }

  /**
   * Strictly verify that the configured service endpoint is a loopback address.
   * This guarantees that auth tokens and browsing URLs are never sent over the network
   * to external or public servers.
   * @param {string} urlString
   * @returns {boolean}
   */
  isLoopbackService(urlString = this.serviceUrl) {
    try {
      const parsed = new URL(urlString);
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        return false;
      }
      const hostname = parsed.hostname.toLowerCase();
      return (
        hostname === '127.0.0.1' ||
        hostname === 'localhost' ||
        hostname === '[::1]' ||
        hostname.startsWith('127.')
      );
    } catch {
      return false;
    }
  }

  /**
   * Build request headers. Auth tokens are ONLY attached if the target is verified loopback.
   * @returns {HeadersInit}
   */
  getHeaders() {
    if (!this.isLoopbackService()) {
      throw new Error('Refusing to send headers to non-loopback service endpoint');
    }

    const headers = {
      'Content-Type': 'application/json',
      'X-Detector-Client': 'firefox-extension',
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
      headers['X-Detector-Token'] = this.token;
    }

    return headers;
  }

  /**
   * Health check query. Lightweight and does not trigger DNS lookups or ML inference.
   * @returns {Promise<{ ok: boolean, data?: Object, error?: string, status?: number }>}
   */
  async checkHealth() {
    if (!this.isLoopbackService()) {
      return {
        ok: false,
        status: 0,
        error: 'Invalid service address. Must be a loopback HTTP URL (e.g. http://127.0.0.1:8000)',
      };
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), Math.min(this.timeoutMs, 5000));

    try {
      const endpoint = `${this.serviceUrl}/api/v1/health`;
      const res = await fetch(endpoint, {
        method: 'GET',
        signal: controller.signal,
      });

      if (!res.ok) {
        return {
          ok: false,
          status: res.status,
          error: `Health check failed with HTTP ${res.status}`,
        };
      }

      const data = await res.json();
      if (data && data.status === 'ok') {
        return { ok: true, status: 200, data };
      }
      return { ok: false, status: 200, error: 'Unexpected health response structure', data };
    } catch (err) {
      const isTimeout = err.name === 'AbortError';
      return {
        ok: false,
        status: 0,
        error: isTimeout
          ? 'Health check timed out'
          : `Service unreachable: ${err.message || 'connection refused'}`,
      };
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Submit a URL to the detector service.
   * @param {string} url - Target URL to analyze.
   * @param {Object} [options]
   * @param {boolean} [options.runLayer2=true]
   * @param {boolean} [options.runLayer4=true]
   * @returns {Promise<{ ok: boolean, data?: Object, error?: string, status?: number }>}
   */
  async analyzeUrl(url, options = {}) {
    if (!url || typeof url !== 'string') {
      return { ok: false, status: 400, error: 'Target URL is required' };
    }

    if (!this.isLoopbackService()) {
      return {
        ok: false,
        status: 0,
        error: 'Service address must be a loopback HTTP URL (127.0.0.1 or localhost)',
      };
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const headers = this.getHeaders();
      const endpoint = `${this.serviceUrl}/api/v1/analyze`;

      const payload = {
        url: url.trim(),
        run_layer2: options.runLayer2 !== false,
        run_layer4: options.runLayer4 !== false,
      };

      const res = await fetch(endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      let json = null;
      try {
        json = await res.json();
      } catch {
        json = null;
      }

      if (res.status === 200) {
        if (!json || typeof json !== 'object' || !json.classification) {
          return { ok: false, status: 200, error: 'Malformed response from detector service' };
        }
        return { ok: true, status: 200, data: json };
      }

      if (res.status === 401) {
        return {
          ok: false,
          status: 401,
          error: 'Authentication failed. Please check or set your token in the extension options.',
        };
      }

      if (res.status === 403) {
        return {
          ok: false,
          status: 403,
          error: 'Access forbidden by local service. Missing or invalid client header.',
        };
      }

      if (res.status === 422) {
        const detail = (json && json.detail) ? json.detail : 'Invalid URL submitted';
        return { ok: false, status: 422, error: `Validation error: ${detail}` };
      }

      if (res.status === 503) {
        return {
          ok: false,
          status: 503,
          error: 'Detector service is saturated with concurrent scans. Please try again shortly.',
        };
      }

      return {
        ok: false,
        status: res.status,
        error: (json && json.detail) || `Service error: HTTP ${res.status}`,
      };
    } catch (err) {
      const isTimeout = err.name === 'AbortError';
      return {
        ok: false,
        status: 0,
        error: isTimeout
          ? 'Analysis request timed out'
          : `Detector service unavailable: ${err.message || 'connection failed'}`,
      };
    } finally {
      clearTimeout(timer);
    }
  }
}

// Support CommonJS export for Node.js test runner and browser global
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ApiClient };
} else {
  globalThis.ApiClient = ApiClient;
}
