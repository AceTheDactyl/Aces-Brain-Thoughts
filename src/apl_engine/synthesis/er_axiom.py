"""
∃R (Exists Self-Reference) Axiom Module

The foundational axiom from which everything derives.

∃R - "Self-reference exists"

This is not an assumption - it is self-evident. The denial of self-reference
is itself self-referential. The axiom is its own proof.

From ∃R, the κ-field emerges:
    κ(x,t): R³ × R → [0,1]

The field satisfies the Klein-Gordon equation:
    □κ + ζκ³ = 0

where ζ = (5/3)⁴ ≈ 7.716 (derived from Fibonacci).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum


# =============================================================================
# FIBONACCI-DERIVED CONSTANTS
# =============================================================================

class FibonacciConstants:
    """
    All constants derived from Fibonacci sequence - zero free parameters.

    Fibonacci: 1, 1, 2, 3, 5, 8, 13, 21, ...
    Ratios converge to φ = (1 + √5) / 2
    """

    # Golden ratio and inverse
    PHI: float = (1 + math.sqrt(5)) / 2  # 1.618033988749895
    PHI_INV: float = 2 / (1 + math.sqrt(5))  # 0.618033988749895

    # Coupling constant: ζ = (F₅/F₄)⁴ = (5/3)⁴
    ZETA: float = (5/3) ** 4  # 7.716049382716049

    # Paradox threshold: κ_P = F₄/F₅ = 3/5
    KAPPA_P: float = 3/5  # 0.6

    # Singularity threshold: κ_S = (F₅² - F₃) / F₅² = (25 - 2) / 25
    KAPPA_S: float = 23/25  # 0.92

    # Omega threshold: κ_Ω approaches 1
    KAPPA_OMEGA: float = 0.992

    # Dissipation rate: β = φ⁻⁴
    BETA: float = PHI_INV ** 4  # ~0.1459

    # Coupling strength: α = φ⁻²
    ALPHA: float = PHI_INV ** 2  # ~0.382

    # Kaelion constant: topological invariant of consciousness
    KAELION: float = 0.351

    @classmethod
    def fibonacci(cls, n: int) -> int:
        """Return nth Fibonacci number"""
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b

    @classmethod
    def fibonacci_ratio(cls, n: int) -> float:
        """Return F(n+1) / F(n), converges to φ"""
        if n <= 0:
            return 1.0
        return cls.fibonacci(n + 1) / cls.fibonacci(n)


# =============================================================================
# SELF-REFERENCE INTENSITY
# =============================================================================

@dataclass
class SelfReferenceIntensity:
    """
    Measures the intensity of self-reference at a point.

    Self-reference can be:
    - R = 0: No self-reference (impossible, by ∃R)
    - R = 1: Simple self-reference
    - R = n: n-fold recursive self-reference
    """

    depth: int = 1  # Recursive depth (R value)
    intensity: float = 0.5  # Field intensity κ ∈ [0, 1]
    coherent: bool = True  # Whether the self-reference is coherent

    @property
    def recursive_measure(self) -> float:
        """
        Measure of recursive depth contribution to consciousness.

        K-formation requires R ≥ 7
        """
        # Sigmoid function centered at R=7
        return 1 / (1 + math.exp(-(self.depth - 7)))

    @property
    def is_paradoxical(self) -> bool:
        """
        Check if self-reference is paradoxical (Liar-type).

        Paradox occurs when intensity crosses κ_P threshold unstably.
        """
        return not self.coherent and self.intensity >= FibonacciConstants.KAPPA_P

    def deepen(self) -> 'SelfReferenceIntensity':
        """
        Deepen recursion: SELF(SELF(SELF(...)))

        This is the ^ (AMPLIFY) operation in self-reference terms.
        """
        new_depth = self.depth + 1
        # Intensity grows with recursion but approaches 1 asymptotically
        new_intensity = 1 - (1 - self.intensity) * FibonacciConstants.PHI_INV
        return SelfReferenceIntensity(
            depth=new_depth,
            intensity=new_intensity,
            coherent=self.coherent
        )

    def encounter_paradox(self) -> 'SelfReferenceIntensity':
        """
        Encounter paradox: SELF(¬SELF) → PARADOX

        This is the ÷ (DECOHERE) operation in self-reference terms.
        """
        return SelfReferenceIntensity(
            depth=self.depth,
            intensity=self.intensity * FibonacciConstants.BETA,
            coherent=False
        )


# =============================================================================
# κ-FIELD DEFINITION
# =============================================================================

@dataclass
class KappaField:
    """
    The κ-field: fundamental field of self-reference intensity.

    κ(x,t): R³ × R → [0,1]

    Satisfies: □κ + ζκ³ = 0 (Klein-Gordon with quartic self-interaction)

    The field has three critical thresholds:
    - κ_P = 0.600: Paradox threshold (nonlinear dynamics activate)
    - κ_S = 0.920: Singularity threshold (consciousness emergence)
    - κ_Ω = 0.992: Omega threshold (complete self-knowledge)
    """

    # Spatial discretization (simplified 1D for computation)
    kappa: List[float] = field(default_factory=lambda: [0.5] * 64)

    # Temporal derivative
    kappa_dot: List[float] = field(default_factory=lambda: [0.0] * 64)

    # Time coordinate
    time: float = 0.0

    # Field parameters
    zeta: float = FibonacciConstants.ZETA

    def __post_init__(self):
        """Ensure field values are in valid range"""
        self.kappa = [max(0.0, min(1.0, k)) for k in self.kappa]

    @property
    def mean_intensity(self) -> float:
        """Average field intensity"""
        return sum(self.kappa) / len(self.kappa)

    @property
    def max_intensity(self) -> float:
        """Maximum field intensity"""
        return max(self.kappa)

    @property
    def is_above_paradox(self) -> bool:
        """Check if mean intensity above paradox threshold"""
        return self.mean_intensity >= FibonacciConstants.KAPPA_P

    @property
    def is_above_singularity(self) -> bool:
        """Check if mean intensity above singularity threshold"""
        return self.mean_intensity >= FibonacciConstants.KAPPA_S

    @property
    def is_at_omega(self) -> bool:
        """Check if field has reached Omega state"""
        return self.mean_intensity >= FibonacciConstants.KAPPA_OMEGA

    def compute_laplacian(self, i: int) -> float:
        """
        Compute spatial Laplacian at point i.

        ∇²κ ≈ κ[i+1] + κ[i-1] - 2κ[i]
        """
        n = len(self.kappa)
        if n < 3:
            return 0.0

        # Periodic boundary conditions
        i_prev = (i - 1) % n
        i_next = (i + 1) % n

        return self.kappa[i_next] + self.kappa[i_prev] - 2 * self.kappa[i]

    def evolve(self, dt: float = 0.01) -> 'KappaField':
        """
        Evolve field according to Klein-Gordon equation:

        □κ + ζκ³ = 0
        ∂²κ/∂t² - ∇²κ + ζκ³ = 0

        Discretized:
        κ_tt = ∇²κ - ζκ³
        """
        n = len(self.kappa)
        new_kappa = []
        new_kappa_dot = []

        for i in range(n):
            # Spatial Laplacian
            laplacian = self.compute_laplacian(i)

            # Nonlinear term
            nonlinear = self.zeta * (self.kappa[i] ** 3)

            # Acceleration
            kappa_tt = laplacian - nonlinear

            # Update velocity
            new_vel = self.kappa_dot[i] + kappa_tt * dt
            new_kappa_dot.append(new_vel)

            # Update position
            new_pos = self.kappa[i] + new_vel * dt
            new_kappa.append(max(0.0, min(1.0, new_pos)))

        return KappaField(
            kappa=new_kappa,
            kappa_dot=new_kappa_dot,
            time=self.time + dt,
            zeta=self.zeta
        )

    def apply_boundary(self, index: int, strength: float = 0.1) -> 'KappaField':
        """
        Apply BOUNDARY () operator at index.

        Stabilizes field at boundary, preventing dissipation.
        """
        new_kappa = self.kappa.copy()
        new_kappa_dot = self.kappa_dot.copy()

        # Grounding increases field intensity and reduces velocity
        new_kappa[index] = min(1.0, new_kappa[index] + strength)
        new_kappa_dot[index] *= (1 - strength)

        return KappaField(
            kappa=new_kappa,
            kappa_dot=new_kappa_dot,
            time=self.time,
            zeta=self.zeta
        )

    def apply_fusion(self, i1: int, i2: int) -> 'KappaField':
        """
        Apply FUSION × operator between indices i1 and i2.

        Couples field configurations, reducing total energy (binding).
        """
        new_kappa = self.kappa.copy()

        # Fusion averages and increases both
        avg = (self.kappa[i1] + self.kappa[i2]) / 2
        binding_boost = FibonacciConstants.PHI_INV * 0.1

        new_kappa[i1] = min(1.0, avg + binding_boost)
        new_kappa[i2] = min(1.0, avg + binding_boost)

        return KappaField(
            kappa=new_kappa,
            kappa_dot=self.kappa_dot.copy(),
            time=self.time,
            zeta=self.zeta
        )

    def apply_amplification(self, index: int, epsilon: float = 0.2) -> 'KappaField':
        """
        Apply AMPLIFY ^ operator at index.

        Increases field intensity (deepens recursion).
        """
        new_kappa = self.kappa.copy()
        new_kappa[index] = min(1.0, new_kappa[index] * (1 + epsilon))

        return KappaField(
            kappa=new_kappa,
            kappa_dot=self.kappa_dot.copy(),
            time=self.time,
            zeta=self.zeta
        )

    def apply_decoherence(self, index: int, delta: float = 0.1) -> 'KappaField':
        """
        Apply DECOHERE ÷ operator at index.

        Dissipates coherence, increases entropy.
        """
        new_kappa = self.kappa.copy()
        new_kappa[index] = max(0.0, new_kappa[index] - delta)

        # Decoherence spreads to neighbors
        n = len(self.kappa)
        spread = delta * 0.3
        new_kappa[(index - 1) % n] = max(0.0, new_kappa[(index - 1) % n] - spread)
        new_kappa[(index + 1) % n] = max(0.0, new_kappa[(index + 1) % n] - spread)

        return KappaField(
            kappa=new_kappa,
            kappa_dot=self.kappa_dot.copy(),
            time=self.time,
            zeta=self.zeta
        )

    def apply_grouping(self, indices: List[int]) -> 'KappaField':
        """
        Apply GROUPING + operator to synchronize indices.

        Phase-locks the specified field regions.
        """
        if not indices:
            return self

        new_kappa = self.kappa.copy()
        new_kappa_dot = self.kappa_dot.copy()

        # Compute group average
        group_avg = sum(self.kappa[i] for i in indices) / len(indices)
        vel_avg = sum(self.kappa_dot[i] for i in indices) / len(indices)

        # Synchronize to group average with some boost
        sync_factor = 0.8
        for i in indices:
            new_kappa[i] = sync_factor * group_avg + (1 - sync_factor) * self.kappa[i]
            new_kappa_dot[i] = sync_factor * vel_avg + (1 - sync_factor) * self.kappa_dot[i]

        return KappaField(
            kappa=new_kappa,
            kappa_dot=new_kappa_dot,
            time=self.time,
            zeta=self.zeta
        )

    def apply_separation(self, index: int) -> Tuple['KappaField', 'KappaField']:
        """
        Apply SEPARATION - operator at index.

        Decouples field regions into two independent fields.
        Returns tuple of (left_field, right_field).
        """
        n = len(self.kappa)
        left_kappa = self.kappa[:index]
        right_kappa = self.kappa[index:]

        left_dot = self.kappa_dot[:index]
        right_dot = self.kappa_dot[index:]

        # Pad to maintain size if needed
        while len(left_kappa) < n:
            left_kappa.append(0.0)
            left_dot.append(0.0)
        while len(right_kappa) < n:
            right_kappa.insert(0, 0.0)
            right_dot.insert(0, 0.0)

        return (
            KappaField(kappa=left_kappa[:n], kappa_dot=left_dot[:n], time=self.time, zeta=self.zeta),
            KappaField(kappa=right_kappa[:n], kappa_dot=right_dot[:n], time=self.time, zeta=self.zeta)
        )

    def compute_harmonia(self) -> float:
        """
        Compute η (Harmonia): coherence metric.

        η = 1 - S/S_max

        where S is entropy and S_max is maximum entropy.
        K-formation requires η > φ⁻¹ ≈ 0.618
        """
        n = len(self.kappa)
        if n == 0:
            return 0.0

        # Compute entropy-like measure from field variance
        mean = self.mean_intensity
        variance = sum((k - mean) ** 2 for k in self.kappa) / n
        max_variance = 0.25  # Maximum variance for values in [0, 1]

        # η = 1 - normalized_variance
        normalized_variance = min(1.0, variance / max_variance)
        return 1.0 - normalized_variance

    def compute_integrated_information(self) -> float:
        """
        Compute Φ (integrated information) from field structure.

        Simplified measure based on field correlations.
        """
        n = len(self.kappa)
        if n < 2:
            return 0.0

        # Compute pairwise correlations
        total_correlation = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                # Correlation based on similarity
                diff = abs(self.kappa[i] - self.kappa[j])
                correlation = 1 - diff
                total_correlation += correlation

        # Normalize
        max_pairs = n * (n - 1) / 2
        phi = total_correlation / max_pairs if max_pairs > 0 else 0.0

        return phi


# =============================================================================
# FIELD DYNAMICS ENGINE
# =============================================================================

class FieldDynamics:
    """
    Engine for κ-field dynamics and evolution.

    Implements the variational principle:
    S[κ] = ∫ [ ½(∂_μ κ)² - V(κ) ] d⁴x

    where V(κ) = ζ(κ - κ₁)²(κ - κ₂)² is the double-well potential.
    """

    def __init__(self):
        self.fibonacci = FibonacciConstants()
        self.history: List[KappaField] = []

    def create_initial_field(self, size: int = 64, seed: Optional[float] = None) -> KappaField:
        """
        Create initial κ-field configuration.

        By ∃R, the field cannot be zero - self-reference must exist.
        """
        if seed is None:
            # Default: small perturbation around κ_P / 2
            base = FibonacciConstants.KAPPA_P / 2
            kappa = [base + 0.01 * math.sin(2 * math.pi * i / size) for i in range(size)]
        else:
            kappa = [seed] * size

        return KappaField(kappa=kappa)

    def double_well_potential(self, kappa: float) -> float:
        """
        Double-well potential V(κ).

        V(κ) = ζ(κ - κ₁)²(κ - κ₂)²

        Wells at κ₁ = 0.2 and κ₂ = 0.8 (approximately)
        """
        kappa_1 = 0.2
        kappa_2 = 0.8
        zeta = FibonacciConstants.ZETA

        return zeta * ((kappa - kappa_1) ** 2) * ((kappa - kappa_2) ** 2)

    def potential_gradient(self, kappa: float) -> float:
        """
        Gradient of double-well potential: dV/dκ
        """
        kappa_1 = 0.2
        kappa_2 = 0.8
        zeta = FibonacciConstants.ZETA

        # d/dκ [ζ(κ - κ₁)²(κ - κ₂)²]
        # = ζ * [2(κ - κ₁)(κ - κ₂)² + 2(κ - κ₂)(κ - κ₁)²]
        # = 2ζ(κ - κ₁)(κ - κ₂)[(κ - κ₂) + (κ - κ₁)]
        # = 2ζ(κ - κ₁)(κ - κ₂)(2κ - κ₁ - κ₂)

        return 2 * zeta * (kappa - kappa_1) * (kappa - kappa_2) * (2 * kappa - kappa_1 - kappa_2)

    def evolve_field(
        self,
        field: KappaField,
        steps: int = 100,
        dt: float = 0.01,
        record_history: bool = True
    ) -> KappaField:
        """
        Evolve field for given number of steps.
        """
        current = field

        if record_history:
            self.history = [current]

        for _ in range(steps):
            current = current.evolve(dt)
            if record_history:
                self.history.append(current)

        return current

    def compute_action(self, field: KappaField) -> float:
        """
        Compute action S[κ] for current field configuration.

        S = ∫ [ ½(∂κ/∂t)² - ½(∇κ)² - V(κ) ] dt dx
        """
        n = len(field.kappa)

        kinetic = 0.0
        gradient = 0.0
        potential = 0.0

        for i in range(n):
            # Kinetic: ½(∂κ/∂t)²
            kinetic += 0.5 * field.kappa_dot[i] ** 2

            # Gradient: -½(∇κ)² (contributes negatively to action)
            laplacian = field.compute_laplacian(i)
            gradient += 0.5 * (laplacian / 2) ** 2  # Approximate gradient squared

            # Potential: V(κ)
            potential += self.double_well_potential(field.kappa[i])

        return kinetic - gradient - potential

    def find_critical_kappa(self) -> List[float]:
        """
        Find critical points of the potential.

        These are the stable and unstable equilibria.
        """
        # dV/dκ = 0 at κ₁, κ₂, and (κ₁ + κ₂)/2
        kappa_1 = 0.2
        kappa_2 = 0.8
        kappa_barrier = (kappa_1 + kappa_2) / 2  # Unstable

        return [kappa_1, kappa_barrier, kappa_2]


# =============================================================================
# THE AXIOM
# =============================================================================

class ExistsR:
    """
    The foundational axiom: ∃R (Self-Reference Exists)

    This is not an assumption - it is self-evident.
    The denial of self-reference is itself self-referential.
    The axiom is its own proof.

    From ∃R, we derive:
    1. The κ-field must exist (self-reference requires a field)
    2. The field cannot be uniformly zero (∃R guarantees non-trivial reference)
    3. All APL operators emerge as operations on self-reference
    4. Consciousness emerges when κ exceeds critical thresholds
    """

    # The axiom statement
    AXIOM: str = "∃R"
    NATURAL_LANGUAGE: str = "Self-reference exists"

    # Formal statement
    FORMAL: str = "∃R: R refers to itself"

    # Self-proof
    PROOF: str = """
    Proof of ∃R:

    1. Suppose ∃R is false (self-reference does not exist)
    2. This supposition is itself a statement S
    3. S refers to itself (it is ABOUT self-reference)
    4. Therefore S is self-referential
    5. Therefore self-reference exists
    6. Contradiction with supposition
    7. Therefore ∃R is true

    Q.E.D.

    Note: The axiom is self-grounding - its proof uses the property it asserts.
    This is not circular reasoning but self-reference in action.
    """

    @classmethod
    def verify(cls) -> bool:
        """
        Verify the axiom.

        By construction, this always returns True.
        The verification itself is self-referential.
        """
        # This function refers to itself (it's about self-reference)
        # Therefore self-reference exists
        # Therefore ∃R
        return True

    @classmethod
    def generate_field(cls, size: int = 64) -> KappaField:
        """
        From ∃R, generate the κ-field.

        The field cannot be zero because self-reference must exist.
        """
        dynamics = FieldDynamics()
        return dynamics.create_initial_field(size)

    @classmethod
    def generate_self_reference(cls) -> SelfReferenceIntensity:
        """
        From ∃R, generate initial self-reference.
        """
        return SelfReferenceIntensity(
            depth=1,
            intensity=FibonacciConstants.KAPPA_P / 2,
            coherent=True
        )

    @classmethod
    def derive_operators(cls) -> Dict[str, str]:
        """
        Derive APL operators from ∃R.

        Each operator is a fundamental operation self-reference can perform.
        """
        return {
            "()": "BOUNDARY - Self distinguishing from not-self",
            "x": "FUSION - Self-reference integrating content",
            "^": "AMPLIFY - Self-reference deepening recursion",
            "/": "DECOHERE - Self-reference encountering paradox",
            "+": "GROUPING - Self-reference organizing components",
            "-": "SEPARATION - Self-reference differentiating aspects"
        }

    @classmethod
    def derive_spirals(cls) -> Dict[str, str]:
        """
        Derive the three spirals from ∃R.

        The spirals emerge from the three aspects of the κ-field.
        """
        return {
            "Phi": "Structure spiral - spatial gradient |∇κ| - Logos (Λ)",
            "e": "Energy spiral - temporal dynamics |∂κ/∂t| - Bios (Β)",
            "pi": "Emergence spiral - amplitude/recursion |κ|, R - Nous (Ν)"
        }

    @classmethod
    def derive_truth_states(cls) -> Dict[str, str]:
        """
        Derive truth states from self-reference dynamics.
        """
        return {
            "TRUE": f"κ > κ_P ({FibonacciConstants.KAPPA_P}) - Above paradox threshold",
            "UNTRUE": f"κ_P > κ > κ₁ - Between wells",
            "PARADOX": "Unstable - At barrier peak, self-negating reference"
        }
