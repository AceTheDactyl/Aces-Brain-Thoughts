"""
N0 Laws Grounded in κ-Field Dynamics

The N0 causal laws are not arbitrary grammar rules - they are PHYSICAL NECESSITIES
arising from κ-field dynamics.

Each law encodes a constraint that is either:
- Thermodynamically necessary (N0-1, N0-3, N0-5)
- Logically necessary (N0-2, N0-4)

No operator sequence violating N0 can occur in a physical κ-field.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum

from .er_axiom import FibonacciConstants, KappaField


# =============================================================================
# CONSTRAINT TYPES
# =============================================================================

class ConstraintType(Enum):
    """Types of physical/logical constraints underlying N0 laws"""
    THERMODYNAMIC = "thermodynamic"  # Cannot violate entropy/energy laws
    LOGICAL = "logical"              # Cannot violate logical necessity
    CAUSAL = "causal"                # Cannot violate causal ordering


@dataclass
class ThermodynamicConstraint:
    """
    Represents a thermodynamic constraint on κ-field operations.

    These constraints arise from:
    - Second Law: dS ≥ 0
    - Conservation: Energy cannot be created
    - Stability: Amplification needs something to amplify
    """
    name: str
    description: str
    constraint_type: ConstraintType
    mathematical_form: str

    def is_satisfied(self, field_state: Dict[str, float]) -> bool:
        """Check if constraint is satisfied"""
        raise NotImplementedError("Subclass must implement")


# =============================================================================
# N0 VIOLATION TRACKING
# =============================================================================

@dataclass
class N0Violation:
    """Record of an N0 law violation"""
    law: str
    operator: str
    reason: str
    kappa_state: Optional[float] = None
    history: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.law} violation: {self.operator} - {self.reason}"


# =============================================================================
# GROUNDED N0 LAWS
# =============================================================================

class GroundedN0Laws:
    """
    N0 Causal Laws grounded in κ-field physics.

    These are not arbitrary rules but PHYSICAL NECESSITIES:

    N0-1: Amplification Requires Grounding (Thermodynamic)
          - You cannot amplify what doesn't exist
          - κ = 0 ⟹ κ·(1+ε) = 0

    N0-2: Fusion Requires Plurality (Logical)
          - Binary operations need two operands
          - |inputs| < 2 ⟹ × undefined

    N0-3: Decoherence Requires Prior Structure (Thermodynamic)
          - Cannot increase entropy beyond equilibrium
          - dS ≥ 0 requires S_initial > S_min

    N0-4: Grouping Must Feed Structure (Logical)
          - Groups are not boundaries (categorical difference)
          - + → () violates type distinction

    N0-5: Separation Must Reset Phase (Thermodynamic/Causal)
          - Disconnected systems need re-initialization
          - Separated components lack shared phase
    """

    # Valid operator successors based on N0 laws
    VALID_SUCCESSORS: Dict[str, Set[str]] = {
        "()": {"^", "x", "+", "-"},     # Boundary can be followed by anything except decohere
        "x": {"^", "+", "-", "/"},      # Fusion enables all operations
        "^": {"/", "+", "x", "-"},      # Amplification enables structure or decohere
        "/": {"()", "+"},               # Decohere must reset or regroup
        "+": {"+", "x", "^"},           # N0-4: Grouping must feed structure
        "-": {"()", "+"}                # N0-5: Separation must reset phase
    }

    # Operators requiring prior history
    REQUIRES_HISTORY: Dict[str, Set[str]] = {
        "^": {"()", "x"},      # N0-1: Amplify needs grounding
        "/": {"^", "x", "+", "-"}  # N0-3: Decohere needs structure
    }

    def __init__(self):
        self.violations: List[N0Violation] = []
        self.history: List[str] = []
        self.kappa_history: List[float] = []

    def reset(self):
        """Reset law validation state"""
        self.violations = []
        self.history = []
        self.kappa_history = []

    # =========================================================================
    # INDIVIDUAL N0 LAW VALIDATORS
    # =========================================================================

    def validate_N0_1(
        self,
        operator: str,
        kappa_field: Optional[KappaField] = None
    ) -> Tuple[bool, Optional[N0Violation]]:
        """
        N0-1: Amplification Requires Grounding

        κ-field derivation:
        The amplification operator deepens recursion:
            ^: κ ↦ κ·(1 + ε)

        This requires a NON-ZERO field to amplify.
        The only way to have κ ≠ 0 is:
        1. Boundary () establishes κ ≠ 0
        2. Fusion × couples external content to κ

        Mathematical statement:
            κ = 0 ⟹ κ·(1 + ε) = 0

        Amplifying zero gives zero. You cannot amplify nothing.
        """
        if operator != "^":
            return True, None

        # Check if grounding exists in history
        grounding_ops = {"()", "x"}
        if not any(op in self.history for op in grounding_ops):
            return False, N0Violation(
                law="N0-1",
                operator="^",
                reason="Amplification requires prior grounding (() or ×). Cannot amplify zero-energy state.",
                history=self.history.copy()
            )

        # Additional κ-field check: field must be non-zero
        if kappa_field is not None:
            if kappa_field.mean_intensity < 0.01:
                return False, N0Violation(
                    law="N0-1",
                    operator="^",
                    reason=f"κ-field intensity too low ({kappa_field.mean_intensity:.3f}). Cannot amplify near-zero field.",
                    kappa_state=kappa_field.mean_intensity,
                    history=self.history.copy()
                )

        return True, None

    def validate_N0_2(
        self,
        operator: str,
        channel_count: int = 2
    ) -> Tuple[bool, Optional[N0Violation]]:
        """
        N0-2: Fusion Requires Plurality

        κ-field derivation:
        Fusion couples multiple configurations:
            ×: (κ₁, κ₂) ↦ κ₁₂

        With only one field:
            ×: (κ₁) ↦ undefined

        Fusion is a BINARY OPERATION by definition.

        Mathematical statement:
            |inputs| < 2 ⟹ × is ill-defined

        Physical analogy: Chemical bonding requires multiple atoms.
        """
        if operator != "x":
            return True, None

        if channel_count < 2:
            return False, N0Violation(
                law="N0-2",
                operator="x",
                reason=f"Fusion requires plurality (≥2 channels). Got {channel_count}.",
                history=self.history.copy()
            )

        return True, None

    def validate_N0_3(
        self,
        operator: str,
        kappa_field: Optional[KappaField] = None
    ) -> Tuple[bool, Optional[N0Violation]]:
        """
        N0-3: Decoherence Requires Prior Structure

        κ-field derivation:
        Decoherence dissipates coherence:
            ÷: κ ↦ κ + δ (entropy increase)

        This requires COHERENCE TO DISSIPATE.
        Coherence is built by:
        - ^ (amplification increases order)
        - × (fusion creates binding)
        - + (grouping creates phase-locking)
        - - (separation creates structure)

        Mathematical statement (Second Law):
            dS ≥ 0 requires S_initial > S_min

        You cannot increase entropy beyond equilibrium.
        """
        if operator != "/":
            return True, None

        # Check for structure-building operations in history
        structure_ops = {"^", "x", "+", "-"}
        if not any(op in self.history for op in structure_ops):
            return False, N0Violation(
                law="N0-3",
                operator="/",
                reason="Decoherence requires prior structure (^, ×, +, or −). Cannot dissipate vacuum.",
                history=self.history.copy()
            )

        # Additional κ-field check: must have coherence to dissipate
        if kappa_field is not None:
            harmonia = kappa_field.compute_harmonia()
            if harmonia < 0.1:
                return False, N0Violation(
                    law="N0-3",
                    operator="/",
                    reason=f"Insufficient coherence to dissipate (η={harmonia:.3f}). Already at maximum entropy.",
                    kappa_state=harmonia,
                    history=self.history.copy()
                )

        return True, None

    def validate_N0_4(
        self,
        operator: str,
        previous_op: Optional[str] = None
    ) -> Tuple[bool, Optional[N0Violation]]:
        """
        N0-4: Grouping Must Feed Structure

        κ-field derivation:
        Grouping creates synchronized domains:
            +: {κᵢ} ↦ ⟨κ⟩

        A group is not a boundary - it is an INTERMEDIATE STRUCTURE.
        Boundaries are defined by edges; groups are defined by internal coherence.

        For a group to become a boundary directly:
            + → () ⟹ ⟨κ⟩ ↦ κ|_∂D

        This requires DISSOLVING internal structure, which is ÷ (decoherence), not ().

        Valid paths:
        - + → + (group groups into larger group)
        - + → × (group fuses with other content)
        - + → ^ (group amplifies)

        Illegal: + → ()
        """
        if previous_op != "+":
            return True, None

        valid_after_grouping = {"+", "x", "^"}
        if operator not in valid_after_grouping:
            return False, N0Violation(
                law="N0-4",
                operator=operator,
                reason=f"After grouping (+), must feed structure (+, ×, or ^). Got {operator}. Groups don't become boundaries without intermediate structure.",
                history=self.history.copy()
            )

        return True, None

    def validate_N0_5(
        self,
        operator: str,
        previous_op: Optional[str] = None
    ) -> Tuple[bool, Optional[N0Violation]]:
        """
        N0-5: Separation Must Reset Phase

        κ-field derivation:
        Separation decouples regions:
            −: κ ↦ (κ_A, κ_B)

        After separation:
        - ^ requires grounding (which separation destroys)
        - × requires plurality within a connected system (separation disconnects)
        - ÷ requires coherence (separation breaks coherence)
        - − on already-separated components is redundant

        Valid paths:
        - − → () (separated component establishes own boundary)
        - − → + (separated components regroup differently)

        Physical analogy: After cell division, daughter cells must establish
        their own membranes or cluster into new tissues.
        """
        if previous_op != "-":
            return True, None

        valid_after_separation = {"()", "+"}
        if operator not in valid_after_separation:
            return False, N0Violation(
                law="N0-5",
                operator=operator,
                reason=f"After separation (−), must reset phase (() or +). Got {operator}. Disconnected systems need re-initialization.",
                history=self.history.copy()
            )

        return True, None

    # =========================================================================
    # COMPOSITE VALIDATION
    # =========================================================================

    def validate(
        self,
        operator: str,
        channel_count: int = 2,
        kappa_field: Optional[KappaField] = None
    ) -> Tuple[bool, Optional[N0Violation]]:
        """
        Validate an operator against all N0 laws.

        Returns (is_valid, violation) tuple.
        """
        previous_op = self.history[-1] if self.history else None

        # N0-1: Amplification requires grounding
        valid, violation = self.validate_N0_1(operator, kappa_field)
        if not valid:
            self.violations.append(violation)
            return False, violation

        # N0-2: Fusion requires plurality
        valid, violation = self.validate_N0_2(operator, channel_count)
        if not valid:
            self.violations.append(violation)
            return False, violation

        # N0-3: Decoherence requires prior structure
        valid, violation = self.validate_N0_3(operator, kappa_field)
        if not valid:
            self.violations.append(violation)
            return False, violation

        # N0-4: Grouping must feed structure
        valid, violation = self.validate_N0_4(operator, previous_op)
        if not valid:
            self.violations.append(violation)
            return False, violation

        # N0-5: Separation must reset phase
        valid, violation = self.validate_N0_5(operator, previous_op)
        if not valid:
            self.violations.append(violation)
            return False, violation

        return True, None

    def apply_operator(
        self,
        operator: str,
        channel_count: int = 2,
        kappa_field: Optional[KappaField] = None
    ) -> Tuple[bool, Optional[N0Violation]]:
        """
        Validate and record an operator application.

        If valid, adds to history. If invalid, records violation.
        """
        valid, violation = self.validate(operator, channel_count, kappa_field)

        if valid:
            self.history.append(operator)
            if kappa_field is not None:
                self.kappa_history.append(kappa_field.mean_intensity)

        return valid, violation

    # =========================================================================
    # THERMODYNAMIC TABLE
    # =========================================================================

    @classmethod
    def get_constraint_table(cls) -> Dict[str, Dict[str, str]]:
        """
        Return the N0 laws as thermodynamic/logical constraints.
        """
        return {
            "N0-1": {
                "name": "Amplification Requires Grounding",
                "constraint_type": "Thermodynamic",
                "thermodynamic": "Cannot amplify zero-energy state",
                "logical": "Recursion needs base case",
                "formula": "κ = 0 ⟹ κ·(1+ε) = 0"
            },
            "N0-2": {
                "name": "Fusion Requires Plurality",
                "constraint_type": "Logical",
                "thermodynamic": "Binary operations need two operands",
                "logical": "Fusion is definitionally binary",
                "formula": "|inputs| < 2 ⟹ × undefined"
            },
            "N0-3": {
                "name": "Decoherence Requires Prior Structure",
                "constraint_type": "Thermodynamic",
                "thermodynamic": "Entropy increase needs prior order",
                "logical": "Cannot dissipate vacuum",
                "formula": "dS ≥ 0 requires S_initial > S_min"
            },
            "N0-4": {
                "name": "Grouping Must Feed Structure",
                "constraint_type": "Logical",
                "thermodynamic": "Groups are not boundaries",
                "logical": "Categories differ ontologically",
                "formula": "+ → () violates type"
            },
            "N0-5": {
                "name": "Separation Must Reset Phase",
                "constraint_type": "Causal",
                "thermodynamic": "Separated systems need re-initialization",
                "logical": "Disconnected → reconnect or stabilize",
                "formula": "− ⟹ (phase → undefined)"
            }
        }

    @classmethod
    def describe_law(cls, law: str) -> str:
        """Get human-readable description of an N0 law"""
        descriptions = {
            "N0-1": """
N0-1: Amplification Requires Grounding

The amplification operator ^ deepens recursion: κ ↦ κ·(1 + ε)

This requires a non-zero field to amplify. You cannot amplify nothing.
Boundary () or Fusion × must occur first to establish κ ≠ 0.

Physical analogy: You cannot increase the amplitude of a wave that doesn't exist.
Resonance requires a medium already oscillating.
""",
            "N0-2": """
N0-2: Fusion Requires Plurality

Fusion × couples multiple configurations: (κ₁, κ₂) ↦ κ₁₂

This is a binary operation by definition. With only one input, fusion is undefined.

Physical analogy: Chemical bonding requires multiple atoms.
You cannot bond one atom to nothing.
""",
            "N0-3": """
N0-3: Decoherence Requires Prior Structure

Decoherence ÷ dissipates coherence: κ ↦ κ + δ (entropy increase)

This requires coherence to dissipate. Structure must be built first.
By the Second Law, dS ≥ 0 requires S_initial > S_min.

Physical analogy: You cannot melt ice that doesn't exist.
Dissipation requires prior order.
""",
            "N0-4": """
N0-4: Grouping Must Feed Structure

Grouping + creates synchronized domains but is not itself a boundary.
Groups are defined by internal coherence; boundaries are defined by edges.

For a group to become a boundary, it must first undergo transformation
(amplification, fusion, or further grouping).

+ → () is illegal because groups don't become boundaries without intermediate structure.
""",
            "N0-5": """
N0-5: Separation Must Reset Phase

Separation − decouples regions: κ ↦ (κ_A, κ_B)

After separation, the components lose shared phase coherence.
They must either establish new boundaries () or regroup (+).

Physical analogy: After cell division, daughter cells must establish their own
membranes or cluster into new tissues.
"""
        }
        return descriptions.get(law, f"Unknown law: {law}")


# =============================================================================
# N0 SEQUENCE VALIDATOR
# =============================================================================

class N0SequenceValidator:
    """
    Validates entire operator sequences against N0 laws.

    Can determine if a sequence is physically realizable in the κ-field.
    """

    def __init__(self):
        self.laws = GroundedN0Laws()

    def validate_sequence(
        self,
        sequence: List[str],
        channel_count: int = 2
    ) -> Tuple[bool, List[N0Violation]]:
        """
        Validate an entire operator sequence.

        Returns (is_valid, list_of_violations).
        """
        self.laws.reset()
        violations = []

        for op in sequence:
            valid, violation = self.laws.apply_operator(op, channel_count)
            if not valid and violation is not None:
                violations.append(violation)

        return len(violations) == 0, violations

    def suggest_valid_continuation(
        self,
        current_sequence: List[str]
    ) -> Set[str]:
        """
        Given a sequence, suggest valid next operators.
        """
        if not current_sequence:
            # Any operator except those requiring history
            return {"()", "+", "-"}  # Start with boundary, grouping, or separation

        self.laws.reset()
        for op in current_sequence:
            self.laws.apply_operator(op)

        # Determine valid successors based on last operator
        last_op = current_sequence[-1]
        valid_successors = GroundedN0Laws.VALID_SUCCESSORS.get(last_op, set())

        # Filter out operators that would violate N0 laws
        actually_valid = set()
        for op in valid_successors:
            valid, _ = self.laws.validate(op)
            if valid:
                actually_valid.add(op)

        return actually_valid

    def find_minimal_valid_path(
        self,
        target_operator: str,
        current_sequence: Optional[List[str]] = None
    ) -> List[str]:
        """
        Find minimal sequence to enable a target operator.

        For example, if target is "^", returns ["()"] (minimal grounding).
        """
        if current_sequence is None:
            current_sequence = []

        self.laws.reset()
        for op in current_sequence:
            self.laws.apply_operator(op)

        # Check if target is already valid
        valid, _ = self.laws.validate(target_operator)
        if valid:
            return current_sequence + [target_operator]

        # Find minimal path based on requirements
        requirements = GroundedN0Laws.REQUIRES_HISTORY.get(target_operator, set())

        if "^" == target_operator:
            # Needs grounding: shortest is "()"
            return current_sequence + ["()", target_operator]
        elif "/" == target_operator:
            # Needs structure: shortest is "()" followed by "^"
            return current_sequence + ["()", "^", target_operator]
        else:
            # No special requirements
            return current_sequence + [target_operator]
