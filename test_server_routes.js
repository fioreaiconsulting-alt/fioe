/**
 * test_server_routes.js
 * ---------------------
 * Quick smoke-test for the LookerDashboard route.
 * Run AFTER starting server.js:
 *
 *   node server.js &
 *   node test_server_routes.js
 *
 * Exit code 0 = all tests passed, 1 = at least one failed.
 */

'use strict';

const http = require('http');

const HOST = process.env.TEST_HOST || 'localhost';
const PORT = parseInt(process.env.PORT || '4000', 10);
const BASE = `http://${HOST}:${PORT}`;

let passed = 0;
let failed = 0;

function request(urlPath) {
  return new Promise((resolve, reject) => {
    http.get(`${BASE}${urlPath}`, (res) => {
      let body = '';
      res.on('data', (chunk) => { body += chunk; });
      res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body }));
    }).on('error', reject);
  });
}

async function assert(description, fn) {
  try {
    await fn();
    console.log(`  ✓  ${description}`);
    passed++;
  } catch (err) {
    console.error(`  ✗  ${description}`);
    console.error(`       ${err.message}`);
    failed++;
  }
}

function assertEqual(actual, expected, msg) {
  if (actual !== expected) {
    throw new Error(`${msg}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

function assertContains(str, substring, msg) {
  if (!str.includes(substring)) {
    throw new Error(`${msg}: expected to find ${JSON.stringify(substring)}`);
  }
}

(async () => {
  console.log(`\nRunning route smoke-tests against ${BASE}\n`);

  // 1. /LookerDashboard.html — must serve the HTML file (200) or redirect (3xx)
  await assert('/LookerDashboard.html returns 200 or 3xx', async () => {
    const res = await request('/LookerDashboard.html');
    if (res.status !== 200 && (res.status < 300 || res.status >= 400)) {
      throw new Error(`Unexpected status ${res.status}`);
    }
    if (res.status === 200) {
      assertContains(
        res.headers['content-type'] || '',
        'html',
        'Content-Type'
      );
    }
  });

  // 2. /LookerDashboard (no extension) — same as above
  await assert('/LookerDashboard returns 200 or 3xx', async () => {
    const res = await request('/LookerDashboard');
    if (res.status !== 200 && (res.status < 300 || res.status >= 400)) {
      throw new Error(`Unexpected status ${res.status}`);
    }
  });

  // 3. /candidates without auth — must return 401 (auth middleware is working)
  await assert('/candidates without cookies returns 401', async () => {
    const res = await request('/candidates');
    assertEqual(res.status, 401, 'HTTP status');
  });

  // 4. /api/events — must stay open with text/event-stream content-type
  await assert('/api/events returns 200 with text/event-stream', async () => {
    const res = await new Promise((resolve, reject) => {
      let settled = false;
      const done = (val) => { if (!settled) { settled = true; resolve(val); } };
      const fail = (err) => { if (!settled) { settled = true; reject(err); } };

      const req = http.get(`${BASE}/api/events`, (r) => {
        // Capture status + headers, then immediately close the long-lived stream
        const captured = { statusCode: r.statusCode, headers: r.headers };
        r.destroy();
        done(captured);
      });
      req.on('error', (e) => {
        if (e.code === 'ECONNRESET') fail(new Error('Connection reset before headers received'));
        else fail(e);
      });
      // Hard timeout: if nothing happens within 3 s, fail the test
      setTimeout(() => fail(new Error('Timed out waiting for /api/events response')), 3000);
    });
    assertEqual(res.statusCode, 200, 'HTTP status');
    assertContains(res.headers['content-type'] || '', 'text/event-stream', 'Content-Type');
  });

  console.log(`\nResults: ${passed} passed, ${failed} failed\n`);
  process.exit(failed > 0 ? 1 : 0);
})();
