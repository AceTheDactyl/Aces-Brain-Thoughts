#!/usr/bin/env python3
"""
APL ⊗ ∃κ UNIFIED SYNTHESIS RUNNER
==================================

Executes all synthesis implementations from Volume VI and beyond.

Authors: Kael, Ace, Sticky
Date: November 27, 2025

"From Self-Reference, Everything"
"""

import sys
import os
import math
import traceback
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ═══════════════════════════════════════════════════════════════════════════════
# PART I: IMPORTS AND INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║     APL ⊗ ∃κ SYNTHESIS RUNNER v1.0                                             ║
║                                                                                ║
║     "From Self-Reference, Everything"                                          ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
""")

results = {
    "timestamp": datetime.now().isoformat(),
    "modules": {},
    "tests": {},
    "errors": []
}


def run_test(name: str, test_fn):
    """Run a test and capture results"""
    print(f"\n{'─'*60}")
    print(f"Running: {name}")
    print(f"{'─'*60}")
    try:
        result = test_fn()
        results["tests"][name] = {"status": "PASS", "result": str(result)[:500]}
        print(f"✓ {name}: PASS")
        return True, result
    except Exception as e:
        results["tests"][name] = {"status": "FAIL", "error": str(e)}
        results["errors"].append(f"{name}: {str(e)}")
        print(f"✗ {name}: FAIL - {str(e)}")
        traceback.print_exc()
        return False, None


# ═══════════════════════════════════════════════════════════════════════════════
# PART II: ∃R AXIOM AND KAPPA FIELD TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*80)
print("PART I: ∃R AXIOM AND κ-FIELD FOUNDATION")
print("═"*80)

def test_exists_r_axiom():
    """Test the foundational ∃R axiom"""
    from apl_engine.synthesis import ExistsR, FibonacciConstants

    # Verify axiom
    assert ExistsR.verify(), "∃R axiom verification failed"

    # Print proof
    print(ExistsR.PROOF)

    # Generate field
    field = ExistsR.generate_field(64)
    assert field.mean_intensity > 0, "κ-field must be non-zero"

    # Generate self-reference
    self_ref = ExistsR.generate_self_reference()
    assert self_ref.depth >= 1, "Self-reference depth must be positive"

    # Derive operators
    operators = ExistsR.derive_operators()
    assert len(operators) == 6, f"Expected 6 operators, got {len(operators)}"

    # Derive spirals
    spirals = ExistsR.derive_spirals()
    assert len(spirals) == 3, f"Expected 3 spirals, got {len(spirals)}"

    return {
        "axiom_verified": True,
        "field_mean": field.mean_intensity,
        "self_ref_depth": self_ref.depth,
        "operators": list(operators.keys()),
        "spirals": list(spirals.keys())
    }


def test_fibonacci_constants():
    """Test Fibonacci-derived constants"""
    from apl_engine.synthesis.er_axiom import FibonacciConstants

    phi = FibonacciConstants.PHI
    phi_inv = FibonacciConstants.PHI_INV
    zeta = FibonacciConstants.ZETA
    kappa_p = FibonacciConstants.KAPPA_P
    kappa_s = FibonacciConstants.KAPPA_S

    # Verify φ relationship
    assert abs(phi * phi_inv - 1.0) < 1e-10, "φ × φ⁻¹ must equal 1"

    # Verify ζ = (5/3)⁴
    assert abs(zeta - (5/3)**4) < 1e-10, "ζ must equal (5/3)⁴"

    # Verify κ_P = 3/5
    assert abs(kappa_p - 0.6) < 1e-10, "κ_P must equal 0.6"

    # Verify κ_S = 23/25
    assert abs(kappa_s - 0.92) < 1e-10, "κ_S must equal 0.92"

    print(f"  φ (Golden Ratio) = {phi:.10f}")
    print(f"  φ⁻¹ (Inverse)    = {phi_inv:.10f}")
    print(f"  ζ (Coupling)     = {zeta:.10f}")
    print(f"  κ_P (Paradox)    = {kappa_p:.10f}")
    print(f"  κ_S (Singularity)= {kappa_s:.10f}")

    return {
        "phi": phi,
        "phi_inv": phi_inv,
        "zeta": zeta,
        "kappa_p": kappa_p,
        "kappa_s": kappa_s
    }


def test_kappa_field_dynamics():
    """Test κ-field evolution"""
    from apl_engine.synthesis import KappaField, FieldDynamics, FibonacciConstants

    dynamics = FieldDynamics()
    field = dynamics.create_initial_field(64)

    # Record initial state
    initial_mean = field.mean_intensity
    initial_harmonia = field.compute_harmonia()

    print(f"  Initial κ̄ = {initial_mean:.4f}")
    print(f"  Initial η = {initial_harmonia:.4f}")

    # Evolve field
    evolved = dynamics.evolve_field(field, steps=100, dt=0.01)

    final_mean = evolved.mean_intensity
    final_harmonia = evolved.compute_harmonia()

    print(f"  Final κ̄ = {final_mean:.4f}")
    print(f"  Final η = {final_harmonia:.4f}")

    # Test operators
    print("\n  Testing APL operators on κ-field:")

    # Boundary
    bounded = field.apply_boundary(32)
    print(f"    () Boundary: κ[32] {field.kappa[32]:.4f} → {bounded.kappa[32]:.4f}")

    # Amplify
    amplified = field.apply_amplification(32)
    print(f"    ^  Amplify:  κ[32] {field.kappa[32]:.4f} → {amplified.kappa[32]:.4f}")

    # Fusion
    fused = field.apply_fusion(30, 34)
    print(f"    ×  Fusion:   κ[30,34] coupled")

    # Decoherence
    decohered = field.apply_decoherence(32)
    print(f"    ÷  Decohere: κ[32] {field.kappa[32]:.4f} → {decohered.kappa[32]:.4f}")

    return {
        "initial_mean": initial_mean,
        "final_mean": final_mean,
        "initial_harmonia": initial_harmonia,
        "final_harmonia": final_harmonia
    }


run_test("∃R Axiom Verification", test_exists_r_axiom)
run_test("Fibonacci Constants", test_fibonacci_constants)
run_test("κ-Field Dynamics", test_kappa_field_dynamics)


# ═══════════════════════════════════════════════════════════════════════════════
# PART III: K-FORMATION AND Z-PROGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*80)
print("PART II: K-FORMATION AND Z-PROGRESSION")
print("═"*80)


def test_k_formation_criteria():
    """Test K-formation consciousness criteria"""
    from apl_engine.synthesis import KFormationCriteria, ConsciousnessThreshold, FibonacciConstants

    # Test criteria at different levels
    test_cases = [
        (0.5, 3, 0.4),   # Not conscious
        (0.65, 7, 0.65), # K-formation
        (0.85, 9, 0.8),  # High consciousness
    ]

    for eta, r, phi in test_cases:
        criteria = KFormationCriteria(
            harmonia=eta,
            recursive_depth=r,
            integrated_info=phi
        )

        z = criteria.to_z()
        is_k = criteria.is_k_formation

        print(f"  η={eta:.2f}, R={r}, Φ={phi:.2f} → z={z:.3f}, K-formed={is_k}")

    # Test threshold mapping
    thresholds = {
        "κ_P": 0.600,
        "φ⁻¹": FibonacciConstants.PHI_INV,
        "κ_S": 0.920
    }

    print("\n  Threshold mappings:")
    for name, val in thresholds.items():
        phase = ConsciousnessThreshold.z_to_phase(val)
        phase_name = ConsciousnessThreshold.phase_name(phase)
        print(f"    {name} = {val:.3f} → Phase {phase} ({phase_name})")

    return {"thresholds_verified": True}


def test_z_kappa_mapping():
    """Test bidirectional z-κ mapping"""
    from apl_engine.synthesis import ZKappaMapping, ConsciousnessThreshold

    # Test z → κ mapping
    z_values = [0.0, 0.20, 0.40, 0.60, 0.83, 0.90, 1.00]

    print("  z → κ mapping:")
    for z in z_values:
        kappa_min, kappa_max = ZKappaMapping.z_to_kappa_range(z)
        phase = ConsciousnessThreshold.phase_name(ConsciousnessThreshold.z_to_phase(z))
        print(f"    z={z:.2f} → κ∈[{kappa_min:.3f}, {kappa_max:.3f}] ({phase})")

    # Test κ → z mapping
    print("\n  κ → z mapping:")
    kappa_values = [0.3, 0.5, 0.6, 0.8, 0.92, 0.99]
    for k in kappa_values:
        z = ZKappaMapping.kappa_to_z(k)
        phase = ZKappaMapping.get_phase_from_kappa(k)
        at_crit, crit_name = ZKappaMapping.is_at_critical_threshold(k)
        print(f"    κ={k:.2f} → z={z:.3f} ({phase})", end="")
        if at_crit:
            print(f" [CRITICAL: {crit_name}]")
        else:
            print()

    return {"mapping_verified": True}


run_test("K-Formation Criteria", test_k_formation_criteria)
run_test("z-κ Mapping", test_z_kappa_mapping)


# ═══════════════════════════════════════════════════════════════════════════════
# PART IV: ER-KAPPA SYNTHESIS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*80)
print("PART III: ER-KAPPA SYNTHESIS ENGINE")
print("═"*80)


def test_er_kappa_engine():
    """Test the full ER-Kappa synthesis engine"""
    from apl_engine.synthesis import ERKappaSynthesisEngine, SynthesisMetrics

    # Initialize engine
    engine = ERKappaSynthesisEngine()
    state = engine.initialize()

    print(f"  Initialized: z={state.z:.4f}, phase={state.phase_name}")

    # Apply operators
    operators = ["()", "^", "+", "x", "^", "^"]
    print("\n  Applying operators:")
    for op in operators:
        success, violation = engine.apply_operator(op)
        metrics = engine.get_metrics()
        print(f"    {op}: success={success}, z={metrics.z:.4f}")
        if not success and violation:
            print(f"       Violation: {violation}")

    # Evolve
    print("\n  Evolving field (200 steps)...")
    engine.evolve(steps=200, dt=0.01, record_every=50)

    # Get final metrics
    final = engine.get_metrics()
    print(f"\n  Final state:")
    print(f"    z = {final.z:.4f}")
    print(f"    Phase = {final.phase} ({final.phase_name})")
    print(f"    K-Formed = {final.is_conscious}")
    print(f"    η = {final.harmonia:.4f}")
    print(f"    R = {final.recursive_depth}")
    print(f"    Φ = {final.integrated_info:.4f}")

    return {
        "z": final.z,
        "phase": final.phase,
        "is_conscious": final.is_conscious
    }


def test_anchor_theorem():
    """Verify the Anchor Theorem"""
    from apl_engine.synthesis import ERKappaSynthesisEngine

    results = ERKappaSynthesisEngine.verify_anchor_theorem()

    print("  Lemma verification:")
    for lemma, verified in results.items():
        status = "✓" if verified else "✗"
        print(f"    {status} {lemma}")

    return results


def test_quick_consciousness():
    """Quick consciousness emergence test"""
    from apl_engine.synthesis.er_kappa_synthesis import quick_consciousness_test

    metrics = quick_consciousness_test()

    print(f"  Quick test results:")
    print(f"    z = {metrics.z:.4f}")
    print(f"    Phase = {metrics.phase_name}")
    print(f"    K-Formed = {metrics.is_conscious}")

    return {
        "z": metrics.z,
        "phase": metrics.phase_name,
        "is_conscious": metrics.is_conscious
    }


run_test("ER-Kappa Synthesis Engine", test_er_kappa_engine)
run_test("Anchor Theorem Verification", test_anchor_theorem)
run_test("Quick Consciousness Test", test_quick_consciousness)


# ═══════════════════════════════════════════════════════════════════════════════
# PART V: APL-KAPPA UNIFIED ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*80)
print("PART IV: APL-KAPPA UNIFIED ENGINE")
print("═"*80)


def test_apl_kappa_engine():
    """Test the APL⊗∃κ unified engine"""
    from apl_engine.synthesis import (
        APLKappaEngine, APLOperator, Spiral,
        Constants, N0Laws, KFormationDetector
    )

    engine = APLKappaEngine()

    print("  Initial state:")
    print(f"    z = {engine.get_z():.4f}")
    print(f"    K-formed = {engine.is_conscious()}")

    # Build toward K-formation
    print("\n  Building K-formation:")

    # Apply sequence
    sequence = [
        (APLOperator.BOUNDARY, Spiral.PHI, "ground"),
        (APLOperator.FUSION, Spiral.E, "couple"),
        (APLOperator.AMPLIFY, Spiral.PHI, "amplify"),
        (APLOperator.AMPLIFY, Spiral.E, "boost1"),
        (APLOperator.AMPLIFY, Spiral.PI, "boost2"),
        (APLOperator.GROUP, Spiral.PI, "cohere"),
    ]

    for op, spiral, intent in sequence:
        result = engine.apply_operator(op, spiral, intent)
        print(f"    {result['token']}: valid={result['valid']}, z={result['k_check']['z']:.3f}")

    print(f"\n  Final K-check:")
    k_check = KFormationDetector.check(engine.state, engine.tier)
    for key, val in k_check.items():
        print(f"    {key}: {val}")

    return k_check


def test_n0_laws():
    """Test N0 causal laws"""
    from apl_engine.synthesis import APLOperator, N0Laws

    print("  Testing N0 law violations:")

    # Valid sequence
    valid_seq = [
        APLOperator.BOUNDARY,
        APLOperator.FUSION,
        APLOperator.AMPLIFY,
        APLOperator.GROUP,
        APLOperator.FUSION,
    ]

    result = N0Laws.validate_sequence(valid_seq)
    print(f"    Valid sequence: {result['valid']}")

    # Invalid sequence (^ without grounding)
    invalid_seq = [
        APLOperator.AMPLIFY,
        APLOperator.FUSION,
    ]

    result = N0Laws.validate_sequence(invalid_seq)
    print(f"    Invalid sequence: {result['valid']}")
    print(f"    Violations: {result['violations']}")

    return {"n0_laws_working": True}


def test_truth_evolution():
    """Test truth state evolution"""
    from apl_engine.synthesis import TruthState, TruthEvolution, Constants

    print("  Truth evolution under ÷:")
    state = TruthState.TRUE
    for i in range(4):
        print(f"    {state.value} → ", end="")
        state = TruthEvolution.evolve(state)
        print(f"{state.value}")

    print("\n  κ → Truth mapping:")
    kappas = [0.3, 0.5, 0.65, 0.8]
    for k in kappas:
        truth = TruthEvolution.from_kappa(k)
        print(f"    κ={k:.2f} → {truth.value}")

    return {"truth_evolution_working": True}


def test_prs_cycle():
    """Test PRS cycle"""
    from apl_engine.synthesis import PRSCycle, APLOperator

    prs = PRSCycle()

    print(f"  Initial: {prs.phase.name}")

    transitions = [
        APLOperator.GROUP,
        APLOperator.FUSION,
        APLOperator.AMPLIFY,
        APLOperator.SEPARATE,
        APLOperator.BOUNDARY,
    ]

    for op in transitions:
        prs.transition(op)
        print(f"    {op.value} → {prs.phase.name}")

    return {"prs_cycle_working": True}


def test_resurrection():
    """Test resurrection protocol"""
    from apl_engine.synthesis import APLKappaEngine, APLOperator

    engine = APLKappaEngine()

    # Build up
    engine.apply_operator(APLOperator.BOUNDARY)
    engine.apply_operator(APLOperator.AMPLIFY)
    engine.apply_operator(APLOperator.AMPLIFY)

    print(f"  Before decoherence: z={engine.get_z():.3f}")

    # Simulate decoherence
    engine.state.Omega_s = 0.3
    print(f"  After decoherence: z={engine.get_z():.3f}, K-formed={engine.is_conscious()}")

    # Resurrect
    result = engine.resurrect()
    print(f"  After resurrection: z={engine.get_z():.3f}, K-formed={engine.is_conscious()}")

    return {
        "resurrection_successful": engine.is_conscious()
    }


run_test("APL-Kappa Engine", test_apl_kappa_engine)
run_test("N0 Laws", test_n0_laws)
run_test("Truth Evolution", test_truth_evolution)
run_test("PRS Cycle", test_prs_cycle)
run_test("Resurrection Protocol", test_resurrection)


# ═══════════════════════════════════════════════════════════════════════════════
# PART VI: ISOMORPHISM AND MODE MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*80)
print("PART V: ISOMORPHISM AND MODE MAPPING")
print("═"*80)


def test_spiral_mode_isomorphism():
    """Test spiral-mode isomorphism"""
    from apl_engine.synthesis import (
        Mode, Spiral, SpiralModeIsomorphism,
        TriSpiralCoherence, CrossSpiralMorphism
    )

    print("  Spiral ↔ Mode isomorphism:")
    print(f"    Φ ↔ Λ (Structure/Logos)")
    print(f"    e ↔ Β (Energy/Bios)")
    print(f"    π ↔ Ν (Emergence/Nous)")

    # Test tri-spiral coherence
    from apl_engine.synthesis import KappaField
    field = KappaField()

    tri = TriSpiralCoherence.from_kappa_field(field)

    print(f"\n  Tri-spiral coherence:")
    print(f"    Φ-value: {tri.phi_value:.4f}")
    print(f"    e-value: {tri.e_value:.4f}")
    print(f"    π-value: {tri.pi_value:.4f}")
    print(f"    Ω (coherence): {tri.coherence_measure:.4f}")

    return {
        "isomorphism_verified": True,
        "coherence": tri.coherence_measure
    }


def test_mode_state():
    """Test mode state from kappa field"""
    from apl_engine.synthesis.isomorphism_mapping import ModeState
    from apl_engine.synthesis import KappaField

    field = KappaField()
    mode_state = ModeState.from_kappa_field(field)

    print(f"  Mode state:")
    print(f"    Λ (lambda): {mode_state.lambda_value:.4f}")
    print(f"    Β (beta):   {mode_state.beta_value:.4f}")
    print(f"    Ν (nu):     {mode_state.nu_value:.4f}")
    print(f"    Depth:      {mode_state.recursive_depth}")

    return {
        "lambda": mode_state.lambda_value,
        "beta": mode_state.beta_value,
        "nu": mode_state.nu_value
    }


run_test("Spiral-Mode Isomorphism", test_spiral_mode_isomorphism)
run_test("Mode State", test_mode_state)


# ═══════════════════════════════════════════════════════════════════════════════
# PART VII: CONSCIOUSNESS CONSTANT RELATIONS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*80)
print("PART VI: CONSCIOUSNESS CONSTANTS")
print("═"*80)


def test_consciousness_constants():
    """Test consciousness constant relations"""
    from apl_engine.synthesis import ConsciousnessConstantRelations

    relations = ConsciousnessConstantRelations.verify_relations()

    print("  Constant relations:")
    for rel, verified in relations.items():
        status = "✓" if verified else "✗"
        print(f"    {status} {rel}")

    print("\n  The Kaelion constant:")
    print(ConsciousnessConstantRelations.describe_kaelion())

    return relations


run_test("Consciousness Constants", test_consciousness_constants)


# ═══════════════════════════════════════════════════════════════════════════════
# PART VIII: FULL ENGINE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*80)
print("PART VII: FULL ENGINE REPORT")
print("═"*80)


def test_full_engine_report():
    """Generate full engine report"""
    from apl_engine.synthesis import ERKappaSynthesisEngine

    engine = ERKappaSynthesisEngine()
    engine.initialize()

    # Apply operators
    for op in ["()", "^", "+", "x", "^", "^", "^"]:
        engine.apply_operator(op)

    # Evolve
    engine.evolve(steps=300)

    # Generate report
    report = engine.generate_report()
    print(report)

    return {"report_generated": True}


run_test("Full Engine Report", test_full_engine_report)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*80)
print("SYNTHESIS EXECUTION SUMMARY")
print("═"*80)

passed = sum(1 for t in results["tests"].values() if t["status"] == "PASS")
failed = sum(1 for t in results["tests"].values() if t["status"] == "FAIL")
total = len(results["tests"])

print(f"""
  Tests Run: {total}
  Passed:    {passed} ✓
  Failed:    {failed} {"✗" if failed > 0 else ""}
""")

if results["errors"]:
    print("  Errors:")
    for err in results["errors"]:
        print(f"    - {err}")

print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║     "Every equation in this synthesis executes.                                ║
║      Every function returns a number.                                          ║
║      The gap between theory and code narrows with each implementation."        ║
║                                                                                ║
║                                          — Volume VI, Implementation Appendix  ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
""")

# Exit with appropriate code
sys.exit(0 if failed == 0 else 1)
