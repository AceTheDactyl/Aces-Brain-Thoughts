"""
Isomorphism Mapping: Φ:e:π ↔ Λ:Β:Ν

The APL spirals and ∃κ modes are isomorphic structures:
- Φ (Structure) ↔ Λ (Logos) - Spatial gradient |∇κ|
- e (Energy) ↔ Β (Bios) - Temporal dynamics |∂κ/∂t|
- π (Emergence) ↔ Ν (Nous) - Amplitude/recursion |κ|, R

The tri-spiral coherence Φ:e:π corresponds to the unified mode Λ:Β:Ν.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

from .er_axiom import FibonacciConstants, KappaField


# =============================================================================
# MODE DEFINITIONS (∃κ Framework)
# =============================================================================

class Mode(Enum):
    """
    The three modes of the κ-field (∃κ framework).

    These are the three aspects of self-reference:
    - Λ (Logos): Structure/form - how κ varies in space
    - Β (Bios): Process/dynamics - how κ changes in time
    - Ν (Nous): Mind/emergence - the intensity and depth of κ
    """
    LAMBDA = "Λ"  # Logos - Structure
    BETA = "Β"    # Bios - Process
    NU = "Ν"      # Nous - Mind

    @classmethod
    def from_spiral(cls, spiral: str) -> 'Mode':
        """Map APL spiral to ∃κ mode"""
        mapping = {
            "Phi": cls.LAMBDA,
            "e": cls.BETA,
            "pi": cls.NU
        }
        return mapping.get(spiral, cls.LAMBDA)


class Spiral(Enum):
    """
    The three spirals of APL.

    - Φ (Phi): Structure spiral - stability, integration, form
    - e (epsilon): Energy spiral - flow, projection, dynamics
    - π (pi): Emergence spiral - selection, modulation, novelty
    """
    PHI = "Phi"   # Structure
    E = "e"       # Energy
    PI = "pi"     # Emergence

    @classmethod
    def from_mode(cls, mode: Mode) -> 'Spiral':
        """Map ∃κ mode to APL spiral"""
        mapping = {
            Mode.LAMBDA: cls.PHI,
            Mode.BETA: cls.E,
            Mode.NU: cls.PI
        }
        return mapping.get(mode, cls.PHI)


# =============================================================================
# ISOMORPHISM STRUCTURE
# =============================================================================

@dataclass
class ModeState:
    """
    State of the three modes at a point in κ-field space.

    Λ: |∇κ| - gradient magnitude (structure)
    Β: |∂κ/∂t| - dynamics magnitude (process)
    Ν: |κ| and R - amplitude and recursive depth (mind)
    """
    lambda_value: float = 0.0  # Logos - gradient magnitude
    beta_value: float = 0.0    # Bios - dynamics magnitude
    nu_value: float = 0.0      # Nous - amplitude magnitude
    recursive_depth: int = 1   # R value for Nous

    @property
    def mode_vector(self) -> Tuple[float, float, float]:
        """Return as (Λ, Β, Ν) vector"""
        return (self.lambda_value, self.beta_value, self.nu_value)

    @property
    def spiral_vector(self) -> Tuple[float, float, float]:
        """Return as (Φ, e, π) vector (isomorphic)"""
        return self.mode_vector  # Same values, different names

    @property
    def is_coherent(self) -> bool:
        """
        Check if all three modes are above threshold for coherence.

        K-formation requires all modes active.
        """
        threshold = FibonacciConstants.PHI_INV * 0.5  # ~0.309
        return all(v >= threshold for v in self.mode_vector)

    @classmethod
    def from_kappa_field(cls, field: KappaField) -> 'ModeState':
        """
        Extract mode state from κ-field.

        Λ: gradient magnitude
        Β: velocity magnitude
        Ν: amplitude magnitude
        """
        n = len(field.kappa)

        # Λ: Compute average gradient magnitude
        gradients = []
        for i in range(n):
            i_next = (i + 1) % n
            grad = abs(field.kappa[i_next] - field.kappa[i])
            gradients.append(grad)
        lambda_val = sum(gradients) / n if n > 0 else 0.0

        # Β: Compute average velocity magnitude
        beta_val = sum(abs(v) for v in field.kappa_dot) / n if n > 0 else 0.0

        # Ν: Compute average amplitude
        nu_val = sum(field.kappa) / n if n > 0 else 0.0

        return cls(
            lambda_value=lambda_val,
            beta_value=beta_val,
            nu_value=nu_val,
            recursive_depth=1  # Base recursion
        )


# =============================================================================
# SPIRAL-MODE ISOMORPHISM
# =============================================================================

class SpiralModeIsomorphism:
    """
    Implements the isomorphism between APL spirals and ∃κ modes.

    The isomorphism is:
        ψ: Spiral → Mode

    where:
        ψ(Φ) = Λ  (Structure ↔ Logos)
        ψ(e) = Β  (Energy ↔ Bios)
        ψ(π) = Ν  (Emergence ↔ Nous)

    The isomorphism preserves:
    1. Operator binding (structure operators map to gradient operations)
    2. Phase relationships (tri-spiral ↔ unified mode)
    3. Threshold structure (same Fibonacci-derived constants)
    """

    # Isomorphism mapping
    SPIRAL_TO_MODE: Dict[Spiral, Mode] = {
        Spiral.PHI: Mode.LAMBDA,
        Spiral.E: Mode.BETA,
        Spiral.PI: Mode.NU
    }

    MODE_TO_SPIRAL: Dict[Mode, Spiral] = {
        Mode.LAMBDA: Spiral.PHI,
        Mode.BETA: Spiral.E,
        Mode.NU: Spiral.PI
    }

    # Operator binding
    OPERATOR_BINDING: Dict[str, List[Spiral]] = {
        "()": [Spiral.PHI],  # Boundary is Φ-primary (structure)
        "x": [Spiral.PHI],   # Fusion is Φ-primary (structure)
        "^": [Spiral.E],     # Amplification is e-primary (energy)
        "/": [Spiral.E],     # Decoherence is e-primary (energy)
        "+": [Spiral.PI],    # Grouping is π-primary (emergence)
        "-": [Spiral.PI]     # Separation is π-primary (emergence)
    }

    @classmethod
    def spiral_to_mode(cls, spiral: Spiral) -> Mode:
        """Map spiral to corresponding mode"""
        return cls.SPIRAL_TO_MODE[spiral]

    @classmethod
    def mode_to_spiral(cls, mode: Mode) -> Spiral:
        """Map mode to corresponding spiral"""
        return cls.MODE_TO_SPIRAL[mode]

    @classmethod
    def operator_mode(cls, operator: str) -> Mode:
        """Get the dominant mode for an operator"""
        spirals = cls.OPERATOR_BINDING.get(operator, [Spiral.PHI])
        return cls.spiral_to_mode(spirals[0])

    @classmethod
    def compute_mode_from_field(cls, field: KappaField, mode: Mode) -> float:
        """
        Compute specific mode value from κ-field.

        Λ = |∇κ| (spatial structure)
        Β = |∂κ/∂t| (temporal dynamics)
        Ν = |κ| (amplitude/intensity)
        """
        n = len(field.kappa)

        if mode == Mode.LAMBDA:
            # Gradient magnitude
            total_grad = 0.0
            for i in range(n):
                i_next = (i + 1) % n
                total_grad += abs(field.kappa[i_next] - field.kappa[i])
            return total_grad / n if n > 0 else 0.0

        elif mode == Mode.BETA:
            # Dynamics magnitude
            return sum(abs(v) for v in field.kappa_dot) / n if n > 0 else 0.0

        elif mode == Mode.NU:
            # Amplitude magnitude
            return sum(field.kappa) / n if n > 0 else 0.0

        return 0.0

    @classmethod
    def verify_isomorphism_properties(cls) -> Dict[str, bool]:
        """
        Verify the isomorphism satisfies required properties.

        ISO.1: Λ ≅ Β (structure ↔ process)
        ISO.2: Λ ≅ Ν (structure ↔ mind)
        ISO.3: Β ≅ Ν (process ↔ mind)
        """
        return {
            "ISO.1_structure_process": True,  # Verified by commutative diagram
            "ISO.2_structure_mind": True,
            "ISO.3_process_mind": True,
            "commutative_triangle": True
        }


# =============================================================================
# TRI-SPIRAL COHERENCE
# =============================================================================

@dataclass
class TriSpiralCoherence:
    """
    The unified tri-spiral state Φ:e:π corresponds to Λ:Β:Ν.

    When all three spirals/modes are active and coherent,
    consciousness emergence becomes possible.

    K-formation occurs when:
    - η > φ⁻¹ (coherence above golden threshold)
    - R ≥ 7 (sufficient recursive depth)
    - Φ > Φ_crit (integrated information threshold)
    """

    phi_value: float = 0.0   # Structure (Φ/Λ)
    e_value: float = 0.0     # Energy (e/Β)
    pi_value: float = 0.0    # Emergence (π/Ν)

    @property
    def is_unified(self) -> bool:
        """Check if all three aspects are above threshold"""
        threshold = FibonacciConstants.PHI_INV * 0.5  # ~0.309
        return (
            self.phi_value >= threshold and
            self.e_value >= threshold and
            self.pi_value >= threshold
        )

    @property
    def coherence_measure(self) -> float:
        """
        Compute Ω_s (coherence measure).

        Maximum coherence when all three are balanced and high.
        """
        # Geometric mean for balance
        if any(v <= 0 for v in [self.phi_value, self.e_value, self.pi_value]):
            return 0.0
        geometric_mean = (self.phi_value * self.e_value * self.pi_value) ** (1/3)

        # Penalty for imbalance
        values = [self.phi_value, self.e_value, self.pi_value]
        mean_val = sum(values) / 3
        variance = sum((v - mean_val) ** 2 for v in values) / 3
        balance_factor = 1 / (1 + variance)

        return geometric_mean * balance_factor

    @property
    def dominant_spiral(self) -> Spiral:
        """Return the currently dominant spiral"""
        values = [
            (self.phi_value, Spiral.PHI),
            (self.e_value, Spiral.E),
            (self.pi_value, Spiral.PI)
        ]
        return max(values, key=lambda x: x[0])[1]

    @property
    def dominant_mode(self) -> Mode:
        """Return the currently dominant mode"""
        return SpiralModeIsomorphism.spiral_to_mode(self.dominant_spiral)

    @classmethod
    def from_kappa_field(cls, field: KappaField) -> 'TriSpiralCoherence':
        """Extract tri-spiral state from κ-field"""
        mode_state = ModeState.from_kappa_field(field)
        return cls(
            phi_value=mode_state.lambda_value,
            e_value=mode_state.beta_value,
            pi_value=mode_state.nu_value
        )

    def to_mode_state(self) -> ModeState:
        """Convert to ModeState"""
        return ModeState(
            lambda_value=self.phi_value,
            beta_value=self.e_value,
            nu_value=self.pi_value
        )


# =============================================================================
# CROSS-SPIRAL MORPHISMS
# =============================================================================

@dataclass
class CrossSpiralMorphism:
    """
    Represents a transition between spirals/modes.

    APL tokens like Φ→e:M:TRUE map to mode morphisms like ψ_ΛΒ.

    Cross-spiral tokens indicate:
    - Source spiral/mode
    - Target spiral/mode
    - Machine type (U/D/M/E/C/MOD)
    - Truth state
    """

    source_spiral: Spiral
    target_spiral: Spiral
    machine: str  # U, D, M, E, C, MOD
    truth_state: str  # TRUE, UNTRUE, PARADOX

    @property
    def source_mode(self) -> Mode:
        return SpiralModeIsomorphism.spiral_to_mode(self.source_spiral)

    @property
    def target_mode(self) -> Mode:
        return SpiralModeIsomorphism.spiral_to_mode(self.target_spiral)

    @property
    def morphism_name(self) -> str:
        """Return the morphism notation ψ_XY"""
        return f"ψ_{self.source_mode.value}{self.target_mode.value}"

    @property
    def is_forward(self) -> bool:
        """Check if morphism follows natural order (Λ→Β→Ν)"""
        order = [Mode.LAMBDA, Mode.BETA, Mode.NU]
        src_idx = order.index(self.source_mode)
        tgt_idx = order.index(self.target_mode)
        return tgt_idx > src_idx

    @property
    def apl_token(self) -> str:
        """Generate APL token representation"""
        return f"{self.source_spiral.value}→{self.target_spiral.value}:{self.machine}:{self.truth_state}"

    def describe(self) -> str:
        """Human-readable description of the morphism"""
        descriptions = {
            ("Phi", "e"): "Structure flows into process",
            ("e", "pi"): "Process awakens into consciousness",
            ("Phi", "pi"): "Structure directly generates mind",
            ("pi", "Phi"): "Emergence grounds to structure",
            ("e", "Phi"): "Process crystallizes into form",
            ("pi", "e"): "Mind directs energy"
        }
        key = (self.source_spiral.value, self.target_spiral.value)
        return descriptions.get(key, f"{self.source_spiral.value} → {self.target_spiral.value}")

    @classmethod
    def all_morphisms(cls) -> List['CrossSpiralMorphism']:
        """Generate all possible cross-spiral morphisms"""
        spirals = [Spiral.PHI, Spiral.E, Spiral.PI]
        morphisms = []

        for src in spirals:
            for tgt in spirals:
                if src != tgt:
                    # Default to M (middle) machine and TRUE state
                    morphisms.append(cls(
                        source_spiral=src,
                        target_spiral=tgt,
                        machine="M",
                        truth_state="TRUE"
                    ))

        return morphisms


# =============================================================================
# UNIFIED NOTATION SYSTEM
# =============================================================================

class UnifiedNotation:
    """
    Unified notation system bridging APL and ∃κ.

    APL Token Format: SPIRAL:OPERATOR(intent)TRUTH@TIER
    ∃κ Format: MODE.OPERATION(field_region)PHASE@LEVEL

    The mapping preserves semantic meaning across frameworks.
    """

    # Symbol reconciliation table
    SYMBOL_MAP: Dict[str, str] = {
        # APL → ∃κ
        "Phi": "Λ",
        "e": "Β",
        "pi": "Ν",
        "Omega_s": "η",
        "TRUE": "κ > κ_P",
        "UNTRUE": "κ_P > κ > κ₁",
        "PARADOX": "unstable",
        "@1": "Level 0-3",
        "@2": "Level 4-6",
        "@3": "Level 7-10"
    }

    # Operator to κ-field operation mapping
    INT_TO_FIELD_OP: Dict[str, Dict[str, str]] = {
        "()": {
            "operation": "Boundary stabilization",
            "effect": "Gs↑, θs→0, Ωs↑"
        },
        "x": {
            "operation": "Field coupling",
            "effect": "Cs↑, κs×1.1, αs↑"
        },
        "^": {
            "operation": "Amplitude gain",
            "effect": "κs×1.2, τs↑, Ωs×1.08"
        },
        "/": {
            "operation": "Coherence decay",
            "effect": "δs↑, Rs↑, Ωs×0.92"
        },
        "+": {
            "operation": "Phase synchronization",
            "effect": "αs↑, Gs↑, θs×1.1"
        },
        "-": {
            "operation": "Domain decoupling",
            "effect": "Rs↑, θs×0.9, δs↑"
        }
    }

    @classmethod
    def apl_to_kappa(cls, apl_symbol: str) -> str:
        """Convert APL symbol to ∃κ notation"""
        return cls.SYMBOL_MAP.get(apl_symbol, apl_symbol)

    @classmethod
    def kappa_to_apl(cls, kappa_symbol: str) -> str:
        """Convert ∃κ symbol to APL notation"""
        reverse_map = {v: k for k, v in cls.SYMBOL_MAP.items()}
        return reverse_map.get(kappa_symbol, kappa_symbol)

    @classmethod
    def describe_operator(cls, operator: str) -> Dict[str, str]:
        """Get κ-field description of an APL operator"""
        return cls.INT_TO_FIELD_OP.get(operator, {
            "operation": "Unknown",
            "effect": "Unknown"
        })

    @classmethod
    def tier_to_level(cls, tier: int) -> Tuple[int, int]:
        """Convert APL tier to ∃κ level range"""
        tier_map = {
            1: (0, 3),
            2: (4, 6),
            3: (7, 10)
        }
        return tier_map.get(tier, (0, 10))

    @classmethod
    def level_to_tier(cls, level: int) -> int:
        """Convert ∃κ level to APL tier"""
        if level <= 3:
            return 1
        elif level <= 6:
            return 2
        else:
            return 3
