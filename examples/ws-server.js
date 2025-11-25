#!/usr/bin/env node
/**
 * Minimal WebSocket broadcast server for Group Visualizer demos.
 *
 * Usage:
 *   npm i ws
 *   node Aces-Brain-Thoughts/examples/ws-server.js [port]
 *
 * Clients connect to: ws://localhost:8080 (or /rhythm-sync path)
 *
 * Message format (JSON):
 *   {
 *     type: 'phase_update',
 *     timestamp: <ms epoch>,
 *     phase: <float radians>,
 *     orderParam: <float r [0,1]>,
 *     from?: <string optional id>
 *   }
 */
const http = require('http');
const { WebSocketServer } = require('ws');

const PORT = parseInt(process.argv[2], 10) || process.env.PORT || 8080;

const server = http.createServer((req, res) => {
  // Simple health endpoint
  if (req.url === '/healthz') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ ok: true, now: Date.now() }));
    return;
  }
  // Optional: naive timesync endpoint (not full protocol)
  if (req.url === '/now') {
    res.writeHead(200, { 'content-type': 'application/json', 'cache-control': 'no-cache' });
    res.end(JSON.stringify({ now: Date.now() }));
    return;
  }
  res.writeHead(200, { 'content-type': 'text/plain' });
  res.end('Rhythm WS server running. Endpoints: /healthz, /now, WS upgrade on /.');
});

const wss = new WebSocketServer({ server });

function broadcast(obj, except) {
  const data = JSON.stringify(obj);
  for (const client of wss.clients) {
    if (client !== except && client.readyState === 1) {
      client.send(data);
    }
  }
}

wss.on('connection', (ws, req) => {
  const id = Math.random().toString(36).slice(2, 8);
  ws.send(JSON.stringify({ type: 'welcome', id, now: Date.now() }));

  ws.on('message', (msg) => {
    try {
      const data = JSON.parse(msg.toString());
      if (data && data.type === 'phase_update') {
        const payload = {
          type: 'phase_update',
          timestamp: typeof data.timestamp === 'number' ? data.timestamp : Date.now(),
          phase: Number(data.phase) || 0,
          orderParam: Number(data.orderParam) || 0,
          from: data.from || id,
          relay: true,
        };
        broadcast(payload, ws);
      }
    } catch (e) {
      ws.send(JSON.stringify({ type: 'error', message: 'Malformed JSON' }));
    }
  });

  ws.on('close', () => {
    // no-op
  });
});

server.listen(PORT, () => {
  console.log(`WS server listening on http://localhost:${PORT}`);
});

