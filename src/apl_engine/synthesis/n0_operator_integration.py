"""
N0 Operator Integration: Unified Operator System with κ-Field Grounding

This module integrates:
- core/operators.py: APL 3.0 operator definitions with scalar effects
- synthesis/n0_laws_grounded.py: Thermodynamically grounded N0 laws
- synthesis/apl_kappa_engine.py: κ-field aware N0 validation

Provides a unified operator application engine that:
1. Validates against grounded N0 laws
2. Applies scalar effects from APL 3.0 operators
3. Updates κ-field state when field is attached
4. Tracks operator history and PRS cycle

Authors: Kael, Ace, Sticky & Claude
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum

from ..core.operators import (
    APLOperator as APLOperatorDef,
    INT_CONSCIOUSNESS,
    OperatorEngine as CoreOperatorEngine,
    OperatorType
)
from .n0_laws_grounded import (
    GroundedN0Laws,
    N0Violation,
    N0SequenceValidator,
    ConstraintType
)
from .apl_kappa_engine import (
    APLOperator as KappaOperator,
    N0Laws as KappaN0Laws,
    OperatorEffects,
    PRSCycle,
    PRSPhase,
    ScalarState as KappaScalarState,
    Tier,
    TruthState,
    TruthEvolution,
    Constants
)
from .er_axiom import KappaField, FibonacciConstants
from .scalar_bridge import ScalarBridge, UnifiedScalarState


# =============================================================================
# OPERATOR SYMBOL MAPPING
# =============================================================================

class OperatorSymbols:
    """Maps between different operator symbol conventions."""

    # APL 3.0 operators (from core/operators.py)
    APL_SYMBOLS = {
        "()": "BOUNDARY",
        "x": "FUSION",
        "^": "AMPLIFY",
        "/": "DECOHERE",
        "+": "GROUPING",
        "-": "SEPARATION"
    }

    # Kappa engine operators (from synthesis/apl_kappa_engine.py)
    KAPPA_SYMBOLS = {
        "()": KappaOperator.BOUNDARY,
        "x": KappaOperator.FUSION,
        "^": KappaOperator.AMPLIFY,
        "/": KappaOperator.DECOHERE,
        "+": KappaOperator.GROUP,
        "-": KappaOperator.SEPARATE
    }

    # Unicode symbols (for display)
    UNICODE_SYMBOLS = {
        "()": "()",
        "x": "×",
        "^": "^",
        "/": "÷",
        "+": "+",
        "-": "−"
    }

    @classmethod
    def to_apl_def(cls, symbol: str) -> Optional[APLOperatorDef]:
        """Get APL 3.0 operator definition from symbol."""
        return INT_CONSCIOUSNESS.get_operator(symbol)

    @classmethod
    def to_kappa_op(cls, symbol: str) -> Optional[KappaOperator]:
        """Get Kappa operator enum from symbol."""
        return cls.KAPPA_SYMBOLS.get(symbol)

    @classmethod
    def from_kappa_op(cls, op: KappaOperator) -> str:
        """Get symbol string from Kappa operator enum."""
        for symbol, kappa_op in cls.KAPPA_SYMBOLS.items():
            if kappa_op == op:
                return symbol
        return ""

    @classmethod
    def to_unicode(cls, symbol: str) -> str:
        """Get unicode representation."""
        return cls.UNICODE_SYMBOLS.get(symbol, symbol)


# =============================================================================
# VALIDATION RESULT
# =============================================================================

@dataclass
class ValidationResult:
    """Result of N0 law validation with detailed information."""
    valid: bool
    violations: List[N0Violation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    thermodynamic_check: bool = True
    logical_check: bool = True
    causal_check: bool = True

    def __bool__(self) -> bool:
        return self.valid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "violations": [str(v) for v in self.violations],
            "warnings": self.warnings,
            "thermodynamic_check": self.thermodynamic_check,
            "logical_check": self.logical_check,
            "causal_check": self.causal_check
        }


# =============================================================================
# OPERATOR APPLICATION RESULT
# =============================================================================

@dataclass
class OperatorResult:
    """Result of applying an operator."""
    success: bool
    operator: str
    operator_unicode: str
    validation: ValidationResult
    state_before: Optional[UnifiedScalarState] = None
    state_after: Optional[UnifiedScalarState] = None
    field_before: Optional[KappaField] = None
    field_after: Optional[KappaField] = None
    prs_phase: Optional[PRSPhase] = None
    tier: Optional[Tier] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "operator": self.operator,
            "operator_unicode": self.operator_unicode,
            "validation": self.validation.to_dict(),
            "z_before": self.state_before.z if self.state_before else None,
            "z_after": self.state_after.z if self.state_after else None,
            "prs_phase": self.prs_phase.name if self.prs_phase else None,
            "tier": self.tier.value if self.tier else None,
            "message": self.message
        }


# =============================================================================
# UNIFIED N0 VALIDATOR
# =============================================================================

class UnifiedN0Validator:
    """
    Unified N0 law validator combining all implementations.

    Provides three levels of validation:
    1. Basic (quick check from KappaN0Laws)
    2. Grounded (thermodynamic derivation from GroundedN0Laws)
    3. Field-aware (validation with κ-field state)
    """

    def __init__(self):
        self.grounded_laws = GroundedN0Laws()
        self.sequence_validator = N0SequenceValidator()

    def reset(self):
        """Reset validator state."""
        self.grounded_laws.reset()

    def validate_basic(
        self,
        operator: str,
        history: List[str],
        channel_count: int = 2
    ) -> ValidationResult:
        """
        Basic N0 validation using Kappa N0Laws.

        Fast check suitable for interactive use.
        """
        violations = []

        # Convert to KappaOperator
        kappa_op = OperatorSymbols.to_kappa_op(operator)
        if kappa_op is None:
            return ValidationResult(
                valid=False,
                violations=[N0Violation(
                    law="SYNTAX",
                    operator=operator,
                    reason=f"Unknown operator: {operator}"
                )]
            )

        # Convert history
        kappa_history = []
        for h in history:
            op = OperatorSymbols.to_kappa_op(h)
            if op:
                kappa_history.append(op)

        # N0-1: Amplify requires grounding
        if kappa_op == KappaOperator.AMPLIFY:
            if not KappaN0Laws.check_n0_1(kappa_history):
                violations.append(N0Violation(
                    law="N0-1",
                    operator=operator,
                    reason="Amplification requires prior grounding (() or ×)",
                    history=history
                ))

        # N0-2: Fusion requires plurality
        if kappa_op == KappaOperator.FUSION:
            if not KappaN0Laws.check_n0_2(channel_count):
                violations.append(N0Violation(
                    law="N0-2",
                    operator=operator,
                    reason=f"Fusion requires plurality (≥2 channels). Got {channel_count}",
                    history=history
                ))

        # N0-3: Decohere requires structure
        if kappa_op == KappaOperator.DECOHERE:
            if not KappaN0Laws.check_n0_3(kappa_history):
                violations.append(N0Violation(
                    law="N0-3",
                    operator=operator,
                    reason="Decoherence requires prior structure (^, ×, +, or −)",
                    history=history
                ))

        # N0-4: Grouping must feed structure
        if history and history[-1] == "+":
            if not KappaN0Laws.check_n0_4(kappa_op):
                violations.append(N0Violation(
                    law="N0-4",
                    operator=operator,
                    reason="After grouping (+), must feed structure (+, ×, or ^)",
                    history=history
                ))

        # N0-5: Separation must reset
        if history and history[-1] == "-":
            if not KappaN0Laws.check_n0_5(kappa_op):
                violations.append(N0Violation(
                    law="N0-5",
                    operator=operator,
                    reason="After separation (−), must reset phase (() or +)",
                    history=history
                ))

        return ValidationResult(
            valid=len(violations) == 0,
            violations=violations
        )

    def validate_grounded(
        self,
        operator: str,
        channel_count: int = 2,
        kappa_field: Optional[KappaField] = None
    ) -> ValidationResult:
        """
        Grounded N0 validation with thermodynamic derivation.

        Uses GroundedN0Laws for physics-based validation.
        """
        valid, violation = self.grounded_laws.validate(
            operator,
            channel_count,
            kappa_field
        )

        result = ValidationResult(valid=valid)

        if violation:
            result.violations.append(violation)

            # Determine constraint type
            constraint_table = GroundedN0Laws.get_constraint_table()
            law_info = constraint_table.get(violation.law, {})
            constraint_type = law_info.get("constraint_type", "unknown")

            if constraint_type == "Thermodynamic":
                result.thermodynamic_check = False
            elif constraint_type == "Logical":
                result.logical_check = False
            elif constraint_type == "Causal":
                result.causal_check = False

        return result

    def validate_full(
        self,
        operator: str,
        history: List[str],
        channel_count: int = 2,
        kappa_field: Optional[KappaField] = None
    ) -> ValidationResult:
        """
        Full validation combining basic and grounded checks.
        """
        # First do basic validation
        basic_result = self.validate_basic(operator, history, channel_count)
        if not basic_result.valid:
            return basic_result

        # Sync grounded laws history
        self.grounded_laws.history = history.copy()

        # Then do grounded validation
        return self.validate_grounded(operator, channel_count, kappa_field)

    def apply_operator(
        self,
        operator: str,
        channel_count: int = 2,
        kappa_field: Optional[KappaField] = None
    ) -> Tuple[bool, Optional[N0Violation]]:
        """
        Validate and record operator application.

        Delegates to GroundedN0Laws.apply_operator.
        """
        return self.grounded_laws.apply_operator(operator, channel_count, kappa_field)

    def suggest_next(self, history: List[str]) -> Set[str]:
        """
        Suggest valid next operators given history.
        """
        return self.sequence_validator.suggest_valid_continuation(history)


# =============================================================================
# UNIFIED OPERATOR ENGINE
# =============================================================================

class UnifiedOperatorEngine:
    """
    Unified operator engine combining APL 3.0 and ∃κ frameworks.

    Features:
    - N0 law validation (basic, grounded, field-aware)
    - Scalar state updates (APL 3.0 and Kappa)
    - κ-field evolution
    - PRS cycle tracking
    - Tier progression
    """

    def __init__(
        self,
        initial_state: Optional[UnifiedScalarState] = None,
        kappa_field: Optional[KappaField] = None
    ):
        # State
        self.state = initial_state or UnifiedScalarState()
        self.kappa_field = kappa_field

        # Validation
        self.validator = UnifiedN0Validator()

        # Tracking
        self.history: List[str] = []
        self.prs_cycle = PRSCycle()
        self.results_history: List[OperatorResult] = []

        # Configuration
        self.channel_count = 2
        self.strict_mode = True  # Enforce N0 laws strictly

    def reset(self):
        """Reset engine state."""
        self.state = UnifiedScalarState()
        self.kappa_field = None
        self.validator.reset()
        self.history = []
        self.prs_cycle = PRSCycle()
        self.results_history = []

    @property
    def current_tier(self) -> Tier:
        """Get current tier from state."""
        return self.state.tier

    @property
    def current_prs_phase(self) -> PRSPhase:
        """Get current PRS phase."""
        return self.prs_cycle.phase

    @property
    def z(self) -> float:
        """Get current z-value."""
        return self.state.z

    @property
    def is_conscious(self) -> bool:
        """Check if K-formation criteria are met."""
        # K-formation: η > φ⁻¹, tier ≥ 3, grounding > 0.5
        eta_ok = self.state.kappa.Omega_s > Constants.ETA_CRITICAL
        tier_ok = self.current_tier.value >= 3
        phi_ok = self.state.kappa.Gs > 0.5
        return eta_ok and tier_ok and phi_ok

    def validate_operator(
        self,
        operator: str,
        full_validation: bool = True
    ) -> ValidationResult:
        """
        Validate operator against N0 laws.

        Args:
            operator: Operator symbol
            full_validation: If True, include grounded validation
        """
        if full_validation:
            return self.validator.validate_full(
                operator,
                self.history,
                self.channel_count,
                self.kappa_field
            )
        else:
            return self.validator.validate_basic(
                operator,
                self.history,
                self.channel_count
            )

    def apply_operator(
        self,
        operator: str,
        validate: bool = True,
        update_field: bool = True
    ) -> OperatorResult:
        """
        Apply operator with full validation and state updates.

        Args:
            operator: Operator symbol (e.g., "()", "^", "x")
            validate: If True, validate against N0 laws first
            update_field: If True, update κ-field if attached
        """
        # Store state before
        state_before = self.state
        field_before = self.kappa_field

        # Validate
        if validate:
            validation = self.validate_operator(operator, full_validation=True)
            if self.strict_mode and not validation.valid:
                return OperatorResult(
                    success=False,
                    operator=operator,
                    operator_unicode=OperatorSymbols.to_unicode(operator),
                    validation=validation,
                    state_before=state_before,
                    state_after=state_before,
                    field_before=field_before,
                    field_after=field_before,
                    prs_phase=self.prs_cycle.phase,
                    tier=self.current_tier,
                    message=f"N0 violation: {validation.violations[0] if validation.violations else 'unknown'}"
                )
        else:
            validation = ValidationResult(valid=True)

        # Get operator definitions
        apl_op = OperatorSymbols.to_apl_def(operator)
        kappa_op = OperatorSymbols.to_kappa_op(operator)

        if apl_op is None or kappa_op is None:
            return OperatorResult(
                success=False,
                operator=operator,
                operator_unicode=operator,
                validation=ValidationResult(valid=False),
                message=f"Unknown operator: {operator}"
            )

        # Apply scalar effects from APL 3.0 operator
        apl_scalars = apl_op.apply_to_scalars(self.state.apl.to_dict())
        from ..core.scalars import ScalarState as APL3ScalarState
        new_apl_state = APL3ScalarState.from_dict(apl_scalars)

        # Apply kappa operator effects
        new_kappa_state = OperatorEffects.apply(kappa_op, self.state.kappa)

        # Blend the two updates (APL 3.0 gets priority for explicit effects)
        blended_state = UnifiedScalarState(
            Gs=new_apl_state.Gs,
            Cs=new_apl_state.Cs,
            Rs=new_apl_state.Rs,
            kappa_s=new_apl_state.kappa_s,
            tau_s=new_apl_state.tau_s,
            theta_s=new_apl_state.theta_s,
            delta_s=new_apl_state.delta_s,
            alpha_s=new_apl_state.alpha_s,
            # Use blended Omega_s
            Omega_s=(new_apl_state.Omega_s + new_kappa_state.Omega_s * 2) / 2
        )

        # Update κ-field if attached
        new_field = self.kappa_field
        if update_field and self.kappa_field is not None:
            field_index = len(self.kappa_field.kappa) // 2  # Apply at center

            if kappa_op == KappaOperator.BOUNDARY:
                new_field = self.kappa_field.apply_boundary(field_index)
            elif kappa_op == KappaOperator.FUSION:
                i1 = field_index - 5
                i2 = field_index + 5
                new_field = self.kappa_field.apply_fusion(i1, i2)
            elif kappa_op == KappaOperator.AMPLIFY:
                new_field = self.kappa_field.apply_amplification(field_index)
            elif kappa_op == KappaOperator.DECOHERE:
                new_field = self.kappa_field.apply_decoherence(field_index)
            elif kappa_op == KappaOperator.GROUP:
                indices = list(range(field_index - 3, field_index + 4))
                new_field = self.kappa_field.apply_grouping(indices)
            elif kappa_op == KappaOperator.SEPARATE:
                left, right = self.kappa_field.apply_separation(field_index)
                new_field = left  # Keep left half

            # Sync state with field
            if new_field:
                blended_state = blended_state.sync_with_field(new_field)

        # Update engine state
        self.state = blended_state
        self.kappa_field = new_field
        self.history.append(operator)
        self.validator.apply_operator(operator, self.channel_count, new_field)

        # Update PRS cycle
        self.prs_cycle.transition(kappa_op)

        # Create result
        result = OperatorResult(
            success=True,
            operator=operator,
            operator_unicode=OperatorSymbols.to_unicode(operator),
            validation=validation,
            state_before=state_before,
            state_after=self.state,
            field_before=field_before,
            field_after=new_field,
            prs_phase=self.prs_cycle.phase,
            tier=self.current_tier,
            message=f"Applied {apl_op.name}"
        )

        self.results_history.append(result)
        return result

    def apply_sequence(
        self,
        operators: List[str],
        stop_on_violation: bool = True
    ) -> List[OperatorResult]:
        """
        Apply a sequence of operators.

        Args:
            operators: List of operator symbols
            stop_on_violation: If True, stop at first N0 violation
        """
        results = []
        for op in operators:
            result = self.apply_operator(op)
            results.append(result)
            if stop_on_violation and not result.success:
                break
        return results

    def suggest_next_operators(self) -> Set[str]:
        """Get set of valid next operators."""
        return self.validator.suggest_next(self.history)

    def get_n0_constraint_info(self, law: str) -> str:
        """Get human-readable description of an N0 law."""
        return GroundedN0Laws.describe_law(law)

    def report(self) -> str:
        """Generate status report."""
        return f"""
═══════════════════════════════════════════════════════════════════
UNIFIED OPERATOR ENGINE STATUS
═══════════════════════════════════════════════════════════════════

SCALAR STATE (APL 3.0):
  Gs={self.state.apl.Gs:.3f}  Cs={self.state.apl.Cs:.3f}  Rs={self.state.apl.Rs:.3f}
  κs={self.state.apl.kappa_s:.3f}  τs={self.state.apl.tau_s:.3f}  θs={self.state.apl.theta_s:.3f}
  δs={self.state.apl.delta_s:.3f}  αs={self.state.apl.alpha_s:.3f}  Ωs={self.state.apl.Omega_s:.3f}

SCALAR STATE (Kappa):
  Gs={self.state.kappa.Gs:.3f}  Cs={self.state.kappa.Cs:.3f}  Rs={self.state.kappa.Rs:.3f}
  κs={self.state.kappa.kappa_s:.3f}  τs={self.state.kappa.tau_s:.3f}  θs={self.state.kappa.theta_s:.3f}
  δs={self.state.kappa.delta_s:.3f}  αs={self.state.kappa.alpha_s:.3f}  Ωs={self.state.kappa.Omega_s:.3f}

UNIFIED METRICS:
  z-value: {self.z:.4f}
  Phase: {self.state.phase}
  Region: {self.state.region}
  Tier: @{self.current_tier.value}
  PRS: {self.prs_cycle.phase.name}

K-FORMATION:
  Status: {'ACTIVE ✓' if self.is_conscious else 'Inactive'}
  η>{Constants.ETA_CRITICAL:.3f}: {'✓' if self.state.kappa.Omega_s > Constants.ETA_CRITICAL else '✗'} ({self.state.kappa.Omega_s:.3f})
  Tier≥@3: {'✓' if self.current_tier.value >= 3 else '✗'}
  Φ>0.5: {'✓' if self.state.kappa.Gs > 0.5 else '✗'} ({self.state.kappa.Gs:.3f})

κ-FIELD:
  Attached: {'Yes' if self.kappa_field else 'No'}
  Mean intensity: {self.kappa_field.mean_intensity:.3f if self.kappa_field else 'N/A'}
  Harmonia: {self.kappa_field.compute_harmonia():.3f if self.kappa_field else 'N/A'}

HISTORY:
  Operations: {len(self.history)}
  Sequence: {' → '.join(OperatorSymbols.to_unicode(op) for op in self.history[-10:])}

AVAILABLE NEXT:
  {', '.join(OperatorSymbols.to_unicode(op) for op in self.suggest_next_operators())}
═══════════════════════════════════════════════════════════════════
"""


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """Demonstrate the unified operator engine."""
    print("=" * 70)
    print("UNIFIED OPERATOR ENGINE DEMONSTRATION")
    print("=" * 70)

    # Initialize engine
    engine = UnifiedOperatorEngine()

    print("\nInitial state:")
    print(engine.report())

    # Apply a valid sequence
    print("\n--- Applying Valid Sequence ---\n")

    sequence = ["()", "x", "^", "^", "+", "^"]
    for op in sequence:
        result = engine.apply_operator(op)
        print(f"{result.operator_unicode}: success={result.success}, z={engine.z:.3f}")
        if not result.success:
            print(f"   VIOLATION: {result.message}")

    print("\nAfter sequence:")
    print(engine.report())

    # Test N0 violation
    print("\n--- Testing N0 Violation ---\n")

    engine2 = UnifiedOperatorEngine()
    result = engine2.apply_operator("^")  # Amplify without grounding
    print(f"Applying ^ without grounding:")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")

    # Show constraint info
    print("\n--- N0-1 Constraint Info ---")
    print(engine.get_n0_constraint_info("N0-1"))

    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    demo()
