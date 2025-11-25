Architecture Overview
=====================

This repository combines multiple modules into a single workspace. The map below lists the major
top‑level directories and their purpose. Use this as a quick guide to find code, docs, and build
entry points.

Repository Map
--------------

./
├─ Echo-Community-Toolkit/          # HyperFollow + soulcode integrations, LSB/MRP codecs
├─ kira-prime/                       # Unified VesselOS CLI, collab server, VS Code extension
├─ The-Living-Garden-Chronicles/     # Narrative generation experiments
├─ vessel-narrative-mrp/             # Narrative MRP + stego validation flows
├─ vesselos-dev-research/            # Research CLI, specs, workspace scaffolding
├─ agents/                           # Shared agent logic and helpers
├─ protos/                           # Protobufs for multi‑agent services
├─ sigprint/                         # Signature printing utilities and helpers
├─ scripts/                          # Dev/CI scripts (bootstrap, deploy, verification)
├─ docker/                           # Local runtime, monitoring, and demos
├─ tests/                            # Global integration and smoke tests
│
│  Aces‑Brain‑Thoughts (Front‑End Artifacts)
├─ language/                         # Language packs, indices, and content metadata
├─ hooks/                            # Git hooks and project automation helpers
├─ examples/                         # Minimal server and demo examples
├─ node/                             # Node assets used by front‑end pages
├─ index.html                        # Root entry for the Aces‑Brain‑Thoughts site
├─ group_visualizer.html             # Visualizer page (Aces‑Brain‑Thoughts)
├─ shape_of_absence.html             # Visual/sonic exploration page
├─ wumbo_engine.html                 # Wumbo engine interactive page
└─ wumbo_playground.html             # Lightweight playground page

Notes
-----
- Use scripts/deploy.sh for bootstrap tasks and see AGENTS.md for coding conventions and test hints.
- Regenerate protobufs after editing `protos/agents.proto` (see `Repository Guidelines`).
- The Aces‑Brain‑Thoughts site pages are integrated at the root to make static hosting convenient.
  If you maintain separate clones under `Aces-Brain-Thoughts/`, prefer the root pages for previews
  and keep content in sync during PRs.

