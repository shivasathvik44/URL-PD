/**
 * Automated Unit Tests for Cache, InFlightTracker, and ContinueOnceManager.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const { AnalysisCache, InFlightTracker, ContinueOnceManager } = require('../background/cache.js');

test('AnalysisCache - Set and Get caching', () => {
  const cache = new AnalysisCache(10, 1000);
  const data = { classification: 'safe', risk_score: 0.1 };

  cache.set('https://example.com', data);
  assert.deepEqual(cache.get('https://example.com'), data);
  assert.equal(cache.get('https://other.com'), null);
});

test('AnalysisCache - LRU Eviction', () => {
  const cache = new AnalysisCache(2, 60000); // Max 2 entries

  cache.set('url-1', { id: 1 });
  cache.set('url-2', { id: 2 });
  assert.equal(cache.size(), 2);

  // Access url-1 so url-2 becomes least recently used
  cache.get('url-1');

  // Insert url-3, which should evict url-2
  cache.set('url-3', { id: 3 });
  assert.equal(cache.size(), 2);
  assert.notEqual(cache.get('url-1'), null);
  assert.notEqual(cache.get('url-3'), null);
  assert.equal(cache.get('url-2'), null);
});

test('AnalysisCache - TTL Expiration', async () => {
  const cache = new AnalysisCache(10, 50); // 50ms TTL

  cache.set('url-expiring', { id: 99 });
  assert.notEqual(cache.get('url-expiring'), null);

  // Wait 60ms for expiration
  await new Promise((resolve) => setTimeout(resolve, 60));
  assert.equal(cache.get('url-expiring'), null);
});

test('InFlightTracker - Deduplication of concurrent requests', async () => {
  const tracker = new InFlightTracker();
  let callCount = 0;

  const mockSlowFetch = () => {
    callCount++;
    return new Promise((resolve) => setTimeout(() => resolve({ result: 'done' }), 40));
  };

  // Launch two concurrent requests for the exact same URL
  const p1 = tracker.run('https://concurrent.example.com', mockSlowFetch);
  const p2 = tracker.run('https://concurrent.example.com', mockSlowFetch);

  assert.equal(tracker.has('https://concurrent.example.com'), true);

  const [res1, res2] = await Promise.all([p1, p2]);
  assert.deepEqual(res1, { result: 'done' });
  assert.deepEqual(res2, { result: 'done' });
  // Ensure the underlying task function was called exactly ONCE
  assert.equal(callCount, 1);
  assert.equal(tracker.has('https://concurrent.example.com'), false);
});

test('ContinueOnceManager - Single-use acknowledgment and scope', () => {
  const mgr = new ContinueOnceManager(1000);
  const tabId = 42;
  const url = 'http://phish.example.com';

  assert.equal(mgr.has(tabId, url), false);
  assert.equal(mgr.consume(tabId, url), false);

  // User acknowledges warning
  mgr.acknowledge(tabId, url);
  assert.equal(mgr.has(tabId, url), true);

  // Wrong tab should not be allowed
  assert.equal(mgr.has(999, url), false);
  assert.equal(mgr.consume(999, url), false);

  // Wrong URL should not be allowed
  assert.equal(mgr.has(tabId, 'http://other.com'), false);

  // Consume once
  assert.equal(mgr.consume(tabId, url), true);

  // Subsequent navigation to the same URL is NOT bypassed (single use)
  assert.equal(mgr.consume(tabId, url), false);
  assert.equal(mgr.has(tabId, url), false);
});
