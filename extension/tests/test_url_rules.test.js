/**
 * Automated Unit Tests for URL Eligibility and Navigation Rules.
 * Verifies scheme restrictions, internal page exclusions, and redirect-loop mitigations.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const { isEligibleUrl } = require('../background/background.js');

test('URL Eligibility - Scheme validation', () => {
  // Supported web schemes
  assert.equal(isEligibleUrl('http://example.com'), true);
  assert.equal(isEligibleUrl('https://example.com'), true);
  assert.equal(isEligibleUrl('https://subdomain.domain.org/path/to/page?q=test#hash'), true);
  assert.equal(isEligibleUrl('http://127.0.0.1:3000/app'), true);

  // Ineligible / unsupported schemes
  assert.equal(isEligibleUrl('ftp://ftp.example.com/file.zip'), false);
  assert.equal(isEligibleUrl('file:///home/user/document.pdf'), false);
  assert.equal(isEligibleUrl('ws://example.com/socket'), false);
  assert.equal(isEligibleUrl('wss://example.com/socket'), false);
  assert.equal(isEligibleUrl('mailto:security@example.com'), false);
  assert.equal(isEligibleUrl('tel:+123456789'), false);
});

test('URL Eligibility - Browser internal & special schemes', () => {
  assert.equal(isEligibleUrl('about:blank'), false);
  assert.equal(isEligibleUrl('about:config'), false);
  assert.equal(isEligibleUrl('about:debugging'), false);
  assert.equal(isEligibleUrl('chrome://settings'), false);
  assert.equal(isEligibleUrl('javascript:alert(document.cookie)'), false);
  assert.equal(isEligibleUrl('data:text/html,<h1>Hello</h1>'), false);
  assert.equal(isEligibleUrl('view-source:https://example.com'), false);
});

test('URL Eligibility - Extension internal pages (prevent recursion/loops)', () => {
  assert.equal(isEligibleUrl('moz-extension://a1b2c3d4-e5f6-7890-abcd-ef1234567890/warning/warning.html'), false);
  assert.equal(isEligibleUrl('moz-extension://a1b2c3d4-e5f6-7890-abcd-ef1234567890/popup/popup.html'), false);
  assert.equal(isEligibleUrl('chrome-extension://abcdefghijklmnop/warning/warning.html'), false);
});

test('URL Eligibility - Malformed and invalid input types', () => {
  assert.equal(isEligibleUrl(''), false);
  assert.equal(isEligibleUrl(null), false);
  assert.equal(isEligibleUrl(undefined), false);
  assert.equal(isEligibleUrl(12345), false);
  assert.equal(isEligibleUrl({}), false);
  assert.equal(isEligibleUrl('not a valid url'), false);
  assert.equal(isEligibleUrl('://no-protocol'), false);
});

test('Warning Page Target Sanitization - Validating query param target', () => {
  // Helper matching the validation logic in warning.js
  function sanitizeWarningTarget(rawTarget) {
    if (!rawTarget) return null;
    try {
      const parsed = new URL(rawTarget);
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        return null;
      }
      if (rawTarget.startsWith('moz-extension://') || rawTarget.startsWith('chrome-extension://')) {
        return null;
      }
      return rawTarget;
    } catch {
      return null;
    }
  }

  // Legitimate phishing target passed to warning page
  const validTarget = 'http://paypal-verification.phish.com/login';
  assert.equal(sanitizeWarningTarget(validTarget), validTarget);

  // Attack vectors attempted via target param
  assert.equal(sanitizeWarningTarget('javascript:alert(1)'), null);
  assert.equal(sanitizeWarningTarget('data:text/html,<script>alert(1)</script>'), null);
  assert.equal(sanitizeWarningTarget('moz-extension://uuid/warning/warning.html'), null);
  assert.equal(sanitizeWarningTarget('about:blank'), null);
  assert.equal(sanitizeWarningTarget(''), null);
  assert.equal(sanitizeWarningTarget('garbage-not-url'), null);
});
