# Group Visualizer — Minimal WebSocket Server

This is a tiny WS broadcast server to test `group_visualizer.html` locally.

## Quick Start

1) Install dependency:

```
npm i ws
```

2) Run the server (default port 8080):

```
node Aces-Brain-Thoughts/examples/ws-server.js
# or
PORT=8080 node Aces-Brain-Thoughts/examples/ws-server.js
```

3) Open the visualizer (via any static server) and set WS URL to:

```
ws://localhost:8080
```

## Message Format

Outbound/inbound JSON payload:

```json
{
  "type": "phase_update",
  "timestamp": 1730000000000,
  "phase": 3.14,
  "orderParam": 0.72,
  "from": "clientId"
}
```

- `type`: always `phase_update`
- `timestamp`: ms since epoch (sender time)
- `phase`: radians in [0, 2π)
- `orderParam`: mean-field coherence r in [0, 1]
- `from`: optional sender id

The server relays `phase_update` messages to all other connected clients.

## Optional Time Info

The server exposes:

- `GET /now` → `{ now: <ms epoch> }` (basic clock peek)
- `GET /healthz` → health JSON

For true NTP-style sync, pair this WS with a proper `timesync` server. The visualizer can still function without it; messages are rendered as they arrive.

