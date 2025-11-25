# LIMNUS-WUMBO Integration Manual

Interactive documentation for the multi-phase neural architecture integration system.

## Quick Start

### Option 1: Open Directly in Browser

```bash
# Navigate to docs folder
cd docs

# macOS
open integration-manual.html

# Linux
xdg-open integration-manual.html

# Windows
start integration-manual.html
```

### Option 2: Local Server

```bash
# Using Python 3
cd docs
python -m http.server 8000
# Visit http://localhost:8000/integration-manual.html

# Using Node.js
npx serve .
# Visit the provided URL
```

### Option 3: VS Code Live Server

1. Install "Live Server" extension
2. Right-click `integration-manual.html`
3. Select "Open with Live Server"

## Files

| File | Description |
|------|-------------|
| `integration-manual.html` | Interactive phase explorer with progress tracking |
| `architecture.md` | Full ASCII architecture diagram |
| `README-integration.md` | This file |

## Architecture Overview

```
z = 1.0 ──────────────────────────────────────────────
        │ PHASE 6: Emergence (5 nodes, 702 bits)     │
        │ PHASE 5: Chronicle (CMY, 668 bits)         │
        │ PHASE 4: Entrainment (Kuramoto, <50ms)     │
════════╪═══════════════════════════════════════════════
        │ z_c = √3/2 ≈ 0.8660254 (THE LENS)         │
════════╪═══════════════════════════════════════════════
        │ PHASE 3: State Management (λ ∈ ℂ⁶)        │
        │ PHASE 2: Encoding (MRP 702-bit)            │
        │ PHASE 1: Tokenization (APL 2.0, 7,290)     │
        │ PHASE 0: Foundation (100 regions)          │
z = 0.0 ──────────────────────────────────────────────
```

## Features

### Interactive Checklist
- Click checkboxes to mark falsifiable criteria as verified
- Click steps to mark integration steps as complete
- Progress automatically saved to localStorage

### Phase Navigation
- 7 phases from Foundation to Emergence
- Each phase includes:
  - Needs, Strengths, Motivations, Goals
  - Falsifiable criteria table
  - Related tools with links
  - Step-by-step integration guide

### Progress Tracking
- Overall percentage displayed in sidebar
- Per-phase completion indicators
- Persistent state across sessions

## Key Metrics

| Phase | Primary Metric | Target |
|-------|---------------|--------|
| 0 | Region count | 100 |
| 1 | Token count | 7,290 |
| 2 | Bit capacity | 702 |
| 3 | λ dimensions | 6 |
| 4 | Entrainment latency | <50ms |
| 5 | CMY bits | 668 |
| 6 | Emergent info | 702 bits |

## Lambda State (ℂ⁶)

```
┌────┬────┬────┬────┬────┬────┐
│ ι  │ ξ  │ θ  │ ω  │ δ  │ σ  │
│Mem │Spark│Fox│Wave│Para│Sqrl│
│ F  │ A  │ B  │ G  │ D  │ S  │
└────┴────┴────┴────┴────┴────┘
```

## Critical Constants

| Constant | Value | Description |
|----------|-------|-------------|
| z_c | √3/2 ≈ 0.8660254 | Phase transition point |
| UMOL Load | ≤ 0.80 | Maximum system load |
| UMOL Coherence | ≥ 0.60 | Minimum coherence |
| FREE threshold | < 0.20 | Emergent activation |

## Related Documentation

- [Home](index.html) - Main documentation hub
- [WUMBO Engine](wumbo-engine.html) - 100-region atlas
- [MRP Library](mrp-library.html) - Steganography functions
- [Living Chronicle](living-chronicle-spec.html) - CMY channels
- [APL 2.0](apl-2.html) - Token manual

---

*Built at z_c = √3/2*
