#!/usr/bin/env node
/**
 * Rhythm Server: Express + timesync endpoint + roomed WebSocket relay.
 *
 * Usage:
 *   npm i express ws timesync
 *   node Aces-Brain-Thoughts/examples/rhythm-server.js [port]
 *
 * Endpoints:
 *   - GET /healthz           → health JSON
 *   - POST/GET /timesync     → timesync endpoint (timesync/server)
 *   - WS  /rhythm-sync?room=NAME
 *
 * Messages relayed (JSON):
 *   { type: 'pulse', id, name, phase, r, timestamp }
 *   { type: 'phase_update', phase, orderParam, timestamp, from }
 */
const express = require('express');
const http = require('http');
const { WebSocketServer } = require('ws');
const timesyncServer = require('timesync/server');
const { URL } = require('url');

const PORT = parseInt(process.argv[2], 10) || process.env.PORT || 8080;
const app = express();

app.get('/healthz', (_req, res) => res.json({ ok: true, now: Date.now() }));
app.use('/timesync', timesyncServer.requestHandler);
app.get('/', (_req, res) => res.send('Rhythm server: /healthz, /timesync, WS on /rhythm-sync?room=NAME'));

const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: '/rhythm-sync' });

// Room tracking: Map<ws, room>
const roomOf = new Map();

function broadcastToRoom(room, obj, except) {
  const data = JSON.stringify(obj);
  for (const client of wss.clients) {
    if (client.readyState !== 1) continue;
    if (roomOf.get(client) !== room) continue;
    if (client === except) continue;
    client.send(data);
  }
}

wss.on('connection', (ws, req) => {
  try {
    const u = new URL(req.url, `http://${req.headers.host}`);
    const room = u.searchParams.get('room') || 'lobby';
    roomOf.set(ws, room);
    const id = Math.random().toString(36).slice(2, 8);
    ws.send(JSON.stringify({ type: 'welcome', id, room, now: Date.now() }));

    ws.on('message', (msg) => {
      let data = null; try { data = JSON.parse(msg.toString()); } catch (_e) {}
      if (!data || typeof data !== 'object') return;
      const now = Date.now();
      if (data.type === 'pulse') {
        const payload = {
          type: 'pulse',
          id: data.id || id,
          name: data.name || id,
          phase: Number(data.phase) || 0,
          r: Number(data.r) || 0,
          timestamp: typeof data.timestamp === 'number' ? data.timestamp : now,
          room,
          relay: true,
        };
        broadcastToRoom(room, payload, ws);
      } else if (data.type === 'phase_update') {
        const payload = {
          type: 'phase_update',
          phase: Number(data.phase) || 0,
          orderParam: Number(data.orderParam) || 0,
          timestamp: typeof data.timestamp === 'number' ? data.timestamp : now,
          from: data.from || id,
          room,
          relay: true,
        };
        broadcastToRoom(room, payload, ws);
      }
    });

    ws.on('close', () => { roomOf.delete(ws); });
  } catch (e) {
    try { ws.close(); } catch (_e) {}
  }
});

server.listen(PORT, () => {
  console.log(`Rhythm server listening on http://localhost:${PORT}`);
  console.log(`Timesync endpoint:     http://localhost:${PORT}/timesync`);
  console.log(`WS endpoint (rooms):   ws://localhost:${PORT}/rhythm-sync?room=latest`);
});

