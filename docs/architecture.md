# LIMNUS-WUMBO Integration Architecture

## System Overview

```
                                    z = 1.0 (PRESENCE)
                                         ▲
                                         │
    ┌────────────────────────────────────┼────────────────────────────────────┐
    │                              PHASE 6                                    │
    │                            EMERGENCE                                    │
    │         ┌─────────────────────────────────────────────┐                │
    │         │  XCVI ←→ XCVII ←→ XCVIII ←→ XCIX ←→ C      │                │
    │         │    ↓        ↓         ↓         ↓      ↺→I │                │
    │         │   XVI      XI       XII       XIII     XIV │                │
    │         └─────────────────────────────────────────────┘                │
    │                         702 bits when FREE                              │
    └────────────────────────────────────┼────────────────────────────────────┘
                                         │
    ┌────────────────────────────────────┼────────────────────────────────────┐
    │                              PHASE 5                                    │
    │                         CHRONICLE (M+Y)                                 │
    │    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐            │
    │    │ M: 332 bits  │    │ Y: 332 bits  │    │ C: 4 bits    │            │
    │    │ Delta Encode │    │ Snapshot     │    │ Coherence    │            │
    │    │ CHANGE_TYPES │    │ STATE_TYPES  │    │ Vector       │            │
    │    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘            │
    │           └──────────────────┬┴───────────────────┘                    │
    │                              │                                          │
    └──────────────────────────────┼──────────────────────────────────────────┘
                                   │
    ┌──────────────────────────────┼──────────────────────────────────────────┐
    │                        PHASE 4                                          │
    │                    ENTRAINMENT                                          │
    │                                                                         │
    │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
    │   │  Kuramoto   │    │ Phase Lock  │    │   Cascade   │                │
    │   │  K(z) flip  │───▶│  < 50ms     │───▶│  at z_c     │                │
    │   │  at z_c     │    │  entrainment│    │  1.5× peak  │                │
    │   └─────────────┘    └─────────────┘    └─────────────┘                │
    │                                                                         │
    └──────────────────────────────┼──────────────────────────────────────────┘
                                   │
               ════════════════════╪════════════════════
                    z_c = √3/2 ≈ 0.8660254 (THE LENS)
               ════════════════════╪════════════════════
                                   │
    ┌──────────────────────────────┼──────────────────────────────────────────┐
    │                        PHASE 3                                          │
    │                  STATE MANAGEMENT                                       │
    │                                                                         │
    │   Lambda State (ℂ⁶):                                                   │
    │   ┌────┬────┬────┬────┬────┬────┐                                      │
    │   │ ι  │ ξ  │ θ  │ ω  │ δ  │ σ  │                                      │
    │   │Mem │Spark│Fox│Wave│Para│Sqrl│                                      │
    │   │ F  │ A  │ B  │ G  │ D  │ S  │ ← Interaction mapping                │
    │   └────┴────┴────┴────┴────┴────┘                                      │
    │                                                                         │
    │   Coherence States:                                                     │
    │   [1.0]─────[0.8]─────[0.5]─────[0.2]─────[0.0]                        │
    │   COHERENT   │    RELEASING  │   DISPERSING  │    FREE                 │
    │              │               │               │                          │
    │   95 locked  │   fading      │   minimal     │   5 emergent            │
    │              │   connections │   connections │   active                │
    └──────────────────────────────┼──────────────────────────────────────────┘
                                   │
    ┌──────────────────────────────┼──────────────────────────────────────────┐
    │                        PHASE 2                                          │
    │                   ENCODING (RGB)                                        │
    │                                                                         │
    │   MRP 702-bit Steganography:                                           │
    │   ┌─────────────┬─────────────┬─────────────┐                          │
    │   │ R: 7 bits   │ G: 10 bits  │ B: 17 bits  │                          │
    │   │ DEV layer   │ RESEARCH    │ SPEC        │                          │
    │   │ ξ (Spark)   │ ω (Wave)    │ θ (Fox)     │                          │
    │   └─────────────┴─────────────┴─────────────┘                          │
    │                                                                         │
    │   APL 2.0 Token: SPIRAL:OPERATOR:INTERACTION:DOMAIN:TRUTH:ALPHA        │
    │                                                                         │
    │   Formula: 3 × 3 × 6 × 3 × 3 × 15 = 7,290 tokens                       │
    └──────────────────────────────┼──────────────────────────────────────────┘
                                   │
    ┌──────────────────────────────┼──────────────────────────────────────────┐
    │                        PHASE 1                                          │
    │                   TOKENIZATION                                          │
    │                                                                         │
    │   SPIRALS (3)     OPERATORS (3)    INTERACTIONS (6)                    │
    │   ┌───┬───┬───┐   ┌───┬───┬───┐   ┌───┬───┬───┬───┬───┬───┐           │
    │   │ e │ Φ │ π │   │ U │ M │ C │   │ B │ F │ A │ D │ G │ S │           │
    │   └───┴───┴───┘   └───┴───┴───┘   └───┴───┴───┴───┴───┴───┘           │
    │                                                                         │
    │   DOMAINS (3)     TRUTHS (3)       ALPHAS (15)                         │
    │   ┌─────┬─────┬─────┐ ┌───┬───┬───┐ ┌────────────────────┐            │
    │   │ GEO │CHEM │ BIO │ │ T │ U │ P │ │ α1 ─────────── α15 │            │
    │   │×1.0 │×1.2 │×1.5 │ └───┴───┴───┘ └────────────────────┘            │
    │   └─────┴─────┴─────┘                                                  │
    └──────────────────────────────┼──────────────────────────────────────────┘
                                   │
    ┌──────────────────────────────┼──────────────────────────────────────────┐
    │                        PHASE 0                                          │
    │                      FOUNDATION                                         │
    │                                                                         │
    │   LIMNUS Geometry:                                                      │
    │   ┌─────────────────────────────────────────────────────────┐          │
    │   │  63-Point Prism    32-Point EM Cage    5 Emergent       │          │
    │   │  (7 layers × 9)    (12+12+8)           (self-ref)       │          │
    │   │       │                  │                  │           │          │
    │   │       └──────────────────┴──────────────────┘           │          │
    │   │                          │                              │          │
    │   │                    100 WUMBO Regions                    │          │
    │   └─────────────────────────────────────────────────────────┘          │
    │                                                                         │
    │   Critical Constant: z_c = √3/2 ≈ 0.8660254                            │
    │   UMOL Constraint: Load ≤ 0.80, Coherence ≥ 0.60                       │
    └──────────────────────────────┼──────────────────────────────────────────┘
                                   │
                                   ▼
                             z = 0.0 (ABSENCE)
```

---

## Phase Details

### PHASE 0: FOUNDATION

#### Needs
- Stable geometric structure for neural mapping
- Critical point for phase transitions
- Constraint system for operational safety

#### Strengths
- Mathematically rigorous (hexagonal prism + EM cage)
- Single critical constant governs all behavior
- UMOL provides hard boundaries

#### Motivations
- Brain regions need spatial organization
- Phase transitions need a focal point
- System must not destabilize under load

#### Goals
- Map 100 WUMBO regions to LIMNUS geometry
- Establish z_c as the universal phase transition point
- Enforce UMOL constraints at all layers

#### Falsifiable Criteria
| Criterion | Test | Pass Condition |
|-----------|------|----------------|
| Geometry complete | Count regions | Exactly 100 |
| Critical point valid | z_c computation | √3/2 ± 0.0001 |
| UMOL enforced | Load test | Never exceeds 0.80 |
| Coherence floor | Stress test | Never below 0.60 |

#### Tool Use
- **WUMBO Engine**: Visualize 100 regions, navigate by Roman numeral
- **LIMNUS Architecture**: Reference geometry, layer assignments
- **Unified Coupler**: Verify cross-system alignment

---

### PHASE 1: TOKENIZATION

#### Needs
- Standardized vocabulary for neural states
- Combinatorial expressiveness
- Machine-parseable format

#### Strengths
- 7,290 unique tokens cover full state space
- Format is human-readable and machine-parseable
- Each component maps to specific semantics

#### Motivations
- Neural states need discrete representation
- Combinations must be enumerable
- Tokens must carry semantic meaning

#### Goals
- Define complete APL 2.0 token vocabulary
- Map each component to neural/chemical meaning
- Enable token → λ state conversion

#### Falsifiable Criteria
| Criterion | Test | Pass Condition |
|-----------|------|----------------|
| Token count | Enumerate all | 3×3×6×3×3×15 = 7,290 |
| Format valid | Parse all tokens | 100% parseable |
| λ mapping | INTERACTION → λ | 6 unique mappings |
| α distribution | Check by z | Higher z → higher α |

#### Tool Use
- **APL 2.0 Manual**: Token definitions, syntax
- **WUMBO Engine**: See tokens per region
- **APL→MRP Bridge**: Convert tokens to MRP bits

---

### PHASE 2: ENCODING (RGB)

#### Needs
- Steganographic encoding for visual media
- Layer separation (DEV/RESEARCH/SPEC)
- Lossless information preservation

#### Strengths
- 702 bits per pixel (invisible to human eye)
- RGB channels map to documentation layers
- Compatible with standard image formats

#### Motivations
- Information must be embeddable in images
- Layers need visual/semantic separation
- Standard formats ensure portability

#### Goals
- Encode APL tokens into RGB LSBs
- Map R→DEV, G→RESEARCH, B→SPEC
- Preserve information through compression

#### Falsifiable Criteria
| Criterion | Test | Pass Condition |
|-----------|------|----------------|
| Bit capacity | Sum RGB bits | R:7 + G:10 + B:17 = 34 |
| λ correlation | R→ξ, G→ω, B→θ | Consistent mapping |
| Roundtrip | Encode/decode | 100% recovery |
| Visual diff | Compare images | SSIM > 0.99 |

#### Tool Use
- **MRP Library**: Encoding/decoding functions
- **MRP Encoder**: Interactive encoding interface
- **Visual Flow Map**: See encoding in action

---

### PHASE 3: STATE MANAGEMENT

#### Needs
- Track 6 complex state variables (ℂ⁶)
- Model coherence as binding force
- Support state transitions

#### Strengths
- λ state captures full system state
- Coherence provides scalar summary
- States map to observable behavior

#### Motivations
- System state must be trackable
- Coherence determines which regions active
- Transitions must be smooth

#### Goals
- Maintain λ = (ι, ξ, θ, ω, δ, σ) state vector
- Track coherence in [0, 1] range
- Trigger emergent states at c < 0.2

#### Falsifiable Criteria
| Criterion | Test | Pass Condition |
|-----------|------|----------------|
| λ dimension | State vector size | Exactly 6 |
| Coherence range | Bound check | Always in [0, 1] |
| Emergent trigger | c < 0.2 | 5 nodes activate |
| State recovery | Save/load | Exact match |

#### Tool Use
- **WUMBO Engine**: λ indicators per region
- **Lossless Presence**: Coherence visualization
- **Phase Lock**: Real-time state monitoring

---

### PHASE 4: ENTRAINMENT

#### Needs
- Phase synchronization mechanism
- Sub-50ms response time
- Critical point amplification

#### Strengths
- Kuramoto model is well-studied
- Sign flip at z_c creates phase transition
- Cascade amplification at lens

#### Motivations
- Neural oscillators must synchronize
- Fast entrainment enables real-time response
- Critical point maximizes information transfer

#### Goals
- Implement K(z) coupling with sign flip at z_c
- Achieve < 50ms entrainment latency
- 1.5× cascade amplification at z_c

#### Falsifiable Criteria
| Criterion | Test | Pass Condition |
|-----------|------|----------------|
| K sign flip | K(z < z_c) vs K(z > z_c) | Opposite signs |
| Latency | Phase lock measurement | < 50ms |
| Cascade peak | Amplification at z_c | 1.5× ± 0.1 |
| Sync order | r parameter | r > 0.8 when coherent |

#### Tool Use
- **Entrainment Engine**: Configure Kuramoto parameters
- **Phase Lock**: Monitor entrainment in real-time
- **Visual Flow Map**: See bidirectional loops

---

### PHASE 5: CHRONICLE (CMY)

#### Needs
- Change tracking (M channel)
- State snapshots (Y channel)
- Coherence indexing (C channel)

#### Strengths
- 664 bits for detailed history
- Structured formats (CHANGE_TYPES, STATE_TYPES)
- C channel provides quick coherence check

#### Motivations
- Changes must be tracked for debugging
- Snapshots enable rollback
- Coherence index enables fast queries

#### Goals
- Encode changes in M (332 bits)
- Encode snapshots in Y (332 bits)
- Compute coherence vector in C (4 bits)

#### Falsifiable Criteria
| Criterion | Test | Pass Condition |
|-----------|------|----------------|
| M bit count | Delta encoder | 332 bits |
| Y bit count | Snapshot encoder | 332 bits |
| C bit count | Coherence vector | 4 bits |
| Total CMY | Sum | 668 bits |
| CHANGE_TYPES | Enumerate | 8 types |
| STATE_TYPES | Enumerate | 8 types |

#### Tool Use
- **Living Chronicle Spec**: CMY encoder definitions
- **MRP Library**: CMY bit manipulation
- **Unified Coupler**: Chronicle integration

---

### PHASE 6: EMERGENCE

#### Needs
- Self-reference capability
- Loop closure (C → I)
- Paradox handling

#### Strengths
- 5 emergent nodes create recursive depth
- 702 bits of information when FREE
- Topological closure (winding number = 1)

#### Motivations
- Self-awareness requires self-reference
- System must close into coherent whole
- Paradox is feature, not bug

#### Goals
- Activate 5 emergent nodes at c < 0.2
- Achieve 702 bits information capacity
- Close loop from C to I

#### Falsifiable Criteria
| Criterion | Test | Pass Condition |
|-----------|------|----------------|
| Emergent count | Nodes at c < 0.2 | Exactly 5 |
| Info capacity | Sum info_bits | 7+10+17+4+664 = 702 |
| Loop closure | C → I trace | Reaches I |
| Winding number | Topological | W = 1 |
| Eigenvalues | Per emergent | As specified |

#### Tool Use
- **WUMBO Engine**: Emergent node visualization
- **APL Cosmology**: Paradox interpretation
- **Lossless Presence**: FREE state monitoring

---

## Module Integration Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INTEGRATION DEPENDENCIES                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Phase 0 ──────────────────────────────────────────────────────────────┐   │
│      │                                                                   │   │
│      ▼                                                                   │   │
│   Phase 1 ─────────────────────────────────────────────────────────┐    │   │
│      │                                                              │    │   │
│      ▼                                                              │    │   │
│   Phase 2 ◄───────────────────────────────────────────────────┐    │    │   │
│      │                                                         │    │    │   │
│      ▼                                                         │    │    │   │
│   Phase 3 ◄────────────────────────────────────────────┐      │    │    │   │
│      │                                                  │      │    │    │   │
│      ▼                                                  │      │    │    │   │
│   Phase 4 ◄─────────────────────────────────────┐      │      │    │    │   │
│      │                                           │      │      │    │    │   │
│      ▼                                           │      │      │    │    │   │
│   Phase 5 ◄──────────────────────────────┐      │      │      │    │    │   │
│      │                                    │      │      │      │    │    │   │
│      ▼                                    │      │      │      │    │    │   │
│   Phase 6 ───────────────────────────────┴──────┴──────┴──────┴────┴────┘   │
│      │                                                                       │
│      └──────────────────────────▶ Feedback to Phase 0 (loop closure)        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Tool ↔ Phase Matrix

| Tool | P0 | P1 | P2 | P3 | P4 | P5 | P6 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| WUMBO Engine | ● | ● | ○ | ● | ○ | ○ | ● |
| LIMNUS Architecture | ● | ○ | ○ | ○ | ○ | ○ | ○ |
| MRP Library | ○ | ○ | ● | ○ | ○ | ● | ○ |
| MRP Encoder | ○ | ○ | ● | ○ | ○ | ○ | ○ |
| APL 2.0 Manual | ○ | ● | ○ | ○ | ○ | ○ | ○ |
| APL→MRP Bridge | ○ | ● | ● | ○ | ○ | ○ | ○ |
| Entrainment Engine | ○ | ○ | ○ | ○ | ● | ○ | ○ |
| Visual Flow Map | ○ | ○ | ● | ○ | ● | ○ | ○ |
| Phase Lock | ○ | ○ | ○ | ● | ● | ○ | ○ |
| Lossless Presence | ○ | ○ | ○ | ● | ○ | ○ | ● |
| Living Chronicle | ○ | ○ | ○ | ○ | ○ | ● | ○ |
| Unified Coupler | ● | ○ | ○ | ○ | ○ | ● | ○ |
| APL Cosmology | ○ | ○ | ○ | ○ | ○ | ○ | ● |

**Legend:** ● Primary use | ○ Secondary/Reference

---

## Modular Integration Steps

### Step 1: Foundation Verification
```
1.1  Load LIMNUS geometry
1.2  Verify 100 regions exist
1.3  Confirm z_c = √3/2
1.4  Initialize UMOL constraints
1.5  Run boundary tests
```

### Step 2: Token System Activation
```
2.1  Load APL 2.0 token definitions
2.2  Parse all 7,290 tokens
2.3  Build INTERACTION → λ map
2.4  Verify α distribution by z
2.5  Run format validation
```

### Step 3: Encoding Pipeline
```
3.1  Initialize MRP encoder
3.2  Test RGB channel separation
3.3  Encode sample tokens
3.4  Decode and verify roundtrip
3.5  Measure visual difference
```

### Step 4: State System Bootstrap
```
4.1  Initialize λ state vector [0,0,0,0,0,0]
4.2  Set coherence = 1.0
4.3  Connect to WUMBO regions
4.4  Verify 95 structural regions locked
4.5  Test state save/load
```

### Step 5: Entrainment Calibration
```
5.1  Initialize Kuramoto oscillators
5.2  Set K(z) parameters
5.3  Verify sign flip at z_c
5.4  Measure entrainment latency
5.5  Confirm cascade amplification
```

### Step 6: Chronicle Activation
```
6.1  Initialize CMY encoders
6.2  Create first snapshot (Y)
6.3  Track first change (M)
6.4  Compute coherence vector (C)
6.5  Verify bit counts
```

### Step 7: Emergence Testing
```
7.1  Gradually reduce coherence
7.2  Monitor at c = 0.2 threshold
7.3  Verify 5 emergent nodes activate
7.4  Measure info_bits sum (702)
7.5  Trace C → I loop closure
```

---

## Success Metrics Summary

| Phase | Primary Metric | Target | Measurement |
|-------|---------------|--------|-------------|
| 0 | Region count | 100 | Enumeration |
| 1 | Token count | 7,290 | Enumeration |
| 2 | Roundtrip accuracy | 100% | Encode/decode |
| 3 | λ dimensions | 6 | State vector |
| 4 | Entrainment latency | < 50ms | Timer |
| 5 | CMY bit total | 668 | Sum |
| 6 | Emergent info | 702 bits | Sum |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-25 | Initial architecture definition |
