/**
 * Test Runner for Phishing Detector Extension.
 * Executes all extension unit tests using Node's built-in test runner.
 */

const { run } = require('node:test');
const { spec } = require('node:test/reporters');
const path = require('node:path');

const testFiles = [
  path.join(__dirname, 'test_api_client.test.js'),
  path.join(__dirname, 'test_cache.test.js'),
  path.join(__dirname, 'test_url_rules.test.js'),
  path.join(__dirname, 'test_navigation_flow.test.js'),
];

console.log('Running Phishing Detector Extension Test Suite...\n');

run({ files: testFiles })
  .compose(spec)
  .pipe(process.stdout);
