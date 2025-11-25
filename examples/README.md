# Group Visualizer — Minimal WebSocket Server

This folder contains two servers to test `group_visualizer.html` locally.
It also includes a minimal browser demo of the rhythm entrainment architecture under
`examples/rhythm-entrainment/`.

## Quick Start

1) Install dependencies:

```
npm i express ws timesync
```

2a) Minimal WS relay (no rooms, no timesync):

```
node Aces-Brain-Thoughts/examples/ws-server.js
```

2b) Full rhythm server (rooms + timesync endpoint):

```
node Aces-Brain-Thoughts/examples/rhythm-server.js
```

3) Open the visualizer (via any static server) and set WS URL to, for example:

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

Servers relay messages to other clients. `rhythm-server.js` supports rooms via `?room=NAME` and exposes `/timesync` for full clock alignment.

## Optional Time Info

The server exposes:

- `GET /now` → `{ now: <ms epoch> }` (basic clock peek)
- `GET /healthz` → health JSON

For NTP-style sync, use `rhythm-server.js` and point the visualizer at `ws://localhost:8080/rhythm-sync?room=latest` (or another room). The visualizer derives the `/timesync` URL from the WS host automatically on Connect.

## Rhythm Entrainment Demo

Open the demo page via any static server:

```
python3 -m http.server 8080
# then open http://localhost:8080/Aces-Brain-Thoughts/examples/rhythm-entrainment/index.html
```

- Start/Stop: runs the AudioWorklet beat source and Kuramoto Worker
- BPM: adjusts frames-per-beat inside the worklet
- K: sets global coupling strength in the Worker
- Connect HR Monitor: maps BLE heart rate to BPM (requires HTTPS)

Files:
- `examples/rhythm-entrainment/index.html` – UI and controls
- `examples/rhythm-entrainment/app.js` – wiring of AudioWorklet + Worker + UI
- `examples/rhythm-entrainment/timing-worklet.js` – sample-accurate beat notifications
- `examples/rhythm-entrainment/kuramoto-worker.js` – mean-field Kuramoto solver
