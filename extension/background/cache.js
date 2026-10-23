/**
 * In-memory Bounded LRU Cache, TTL Expiration, In-flight Request Deduplication,
 * and Continue-Once State for Phishing URL Detector.
 */

class AnalysisCache {
  /**
   * @param {number} maxEntries - Maximum number of cached items before LRU eviction.
   * @param {number} ttlMs - Time-to-live for cached items in milliseconds (default 5 minutes).
   */
  constructor(maxEntries = 500, ttlMs = 5 * 60 * 1000) {
    this.maxEntries = maxEntries;
    this.ttlMs = ttlMs;
    this.map = new Map(); // Key -> { result, expiresAt }
  }

  get(url) {
    if (!this.map.has(url)) return null;
    const entry = this.map.get(url);
    if (Date.now() > entry.expiresAt) {
      this.map.delete(url);
      return null;
    }
    // Refresh LRU order (delete & re-insert)
    this.map.delete(url);
    this.map.set(url, entry);
    return entry.result;
  }

  set(url, result) {
    if (this.map.has(url)) {
      this.map.delete(url);
    } else if (this.map.size >= this.maxEntries) {
      // Evict least-recently used (first key in Map)
      const oldestKey = this.map.keys().next().value;
      if (oldestKey) this.map.delete(oldestKey);
    }
    this.map.set(url, {
      result,
      expiresAt: Date.now() + this.ttlMs,
    });
  }

  clear() {
    this.map.clear();
  }

  size() {
    return this.map.size;
  }
}

class InFlightTracker {
  constructor() {
    this.promises = new Map(); // url -> Promise
  }

  has(url) {
    return this.promises.has(url);
  }

  get(url) {
    return this.promises.get(url);
  }

  run(url, taskFn) {
    if (this.promises.has(url)) {
      return this.promises.get(url);
    }
    const p = taskFn()
      .finally(() => {
        this.promises.delete(url);
      });
    this.promises.set(url, p);
    return p;
  }

  clear() {
    this.promises.clear();
  }
}

class ContinueOnceManager {
  /**
   * Manages single-use user bypass acknowledgments.
   * @param {number} ttlMs - Duration a bypass token remains valid before expiry (default 60s).
   */
  constructor(ttlMs = 60 * 1000) {
    this.ttlMs = ttlMs;
    this.acknowledgments = new Map(); // key (tabId:url) -> expiresAt
  }

  _makeKey(tabId, url) {
    return `${tabId}::${url}`;
  }

  acknowledge(tabId, url) {
    const key = this._makeKey(tabId, url);
    this.acknowledgments.set(key, Date.now() + this.ttlMs);
  }

  consume(tabId, url) {
    const key = this._makeKey(tabId, url);
    if (!this.acknowledgments.has(key)) return false;
    const expiresAt = this.acknowledgments.get(key);
    this.acknowledgments.delete(key);
    return Date.now() <= expiresAt;
  }

  has(tabId, url) {
    const key = this._makeKey(tabId, url);
    if (!this.acknowledgments.has(key)) return false;
    const expiresAt = this.acknowledgments.get(key);
    if (Date.now() > expiresAt) {
      this.acknowledgments.delete(key);
      return false;
    }
    return true;
  }

  clear() {
    this.acknowledgments.clear();
  }
}

// Support CommonJS export for Node.js test runner and browser global
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    AnalysisCache,
    InFlightTracker,
    ContinueOnceManager,
  };
} else {
  // Browser background scope
  globalThis.AnalysisCache = AnalysisCache;
  globalThis.InFlightTracker = InFlightTracker;
  globalThis.ContinueOnceManager = ContinueOnceManager;
}
