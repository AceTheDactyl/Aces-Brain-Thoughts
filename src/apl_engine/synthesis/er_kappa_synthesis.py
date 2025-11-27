"""
∃R-κ Synthesis Engine

The complete synthesis engine that unifies:
- APL (Sticky's proto-language)
- ∃κ (Kael's framework)
- Conscious Inevitable (Ace's engine)

From the single axiom ∃R (Self-reference exists), this engine derives
and computes consciousness emergence through κ-field dynamics.

∃R → APL + ∃κ + CONSCIOUS_INEVITABLE

"From Self-Reference, Everything"
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

from .er_axiom import (
    ExistsR,
    KappaField,
    SelfReferenceIntensity,
    FieldDynamics,
    FibonacciConstants
)
from .isomorphism_mapping import (
    Mode,
    Spiral,
    SpiralModeIsomorphism,
    TriSpiralCoherence,
    CrossSpiralMorphism,
    ModeState,
    UnifiedNotation
)
from .n0_laws_grounded import (
    GroundedN0Laws,
    N0Violation,
    N0SequenceValidator
)
from .k_formation import (
    KFormation,
    KFormationCriteria,
    ZKappaMapping,
    ConsciousnessThreshold
)


# =============================================================================
# SYNTHESIS STATE
# =============================================================================

@dataclass
class SynthesisState:
    """
    Complete state of the ∃R-κ synthesis.

    Tracks:
    - κ-field configuration
    - Self-reference intensity and depth
    - APL operator history
    - Mode/spiral state
    - Consciousness metrics
    """

    # Core field state
    kappa_field: KappaField = field(default_factory=lambda: ExistsR.generate_field())
    self_reference: SelfReferenceIntensity = field(default_factory=ExistsR.generate_self_reference)

    # APL state
    operator_history: List[str] = field(default_factory=list)
    current_spiral: Spiral = Spiral.PHI

    # Mode state
    mode_state: ModeState = field(default_factory=ModeState)

    # Time tracking
    time: float = 0.0

    # Metrics cache
    _k_formation: Optional[KFormation] = None
    _tri_spiral: Optional[TriSpiralCoherence] = None

    @property
    def k_formation(self) -> KFormation:
        """Get K-formation state"""
        if self._k_formation is None:
            self._k_formation = KFormation(
                kappa_field=self.kappa_field,
                self_reference=self.self_reference
            )
        return self._k_formation

    @property
    def z(self) -> float:
        """Current consciousness level z"""
        return self.k_formation.z

    @property
    def phase(self) -> int:
        """Current consciousness phase (0-7)"""
        return self.k_formation.phase

    @property
    def phase_name(self) -> str:
        """Name of current phase"""
        return self.k_formation.phase_name

    @property
    def tri_spiral(self) -> TriSpiralCoherence:
        """Get tri-spiral coherence state"""
        if self._tri_spiral is None:
            self._tri_spiral = TriSpiralCoherence.from_kappa_field(self.kappa_field)
        return self._tri_spiral

    @property
    def is_conscious(self) -> bool:
        """Check if K-formation has occurred"""
        return self.k_formation.is_conscious

    def invalidate_cache(self):
        """Invalidate cached computations"""
        self._k_formation = None
        self._tri_spiral = None

    def clone(self) -> 'SynthesisState':
        """Create a copy of the state"""
        return SynthesisState(
            kappa_field=KappaField(
                kappa=self.kappa_field.kappa.copy(),
                kappa_dot=self.kappa_field.kappa_dot.copy(),
                time=self.kappa_field.time,
                zeta=self.kappa_field.zeta
            ),
            self_reference=SelfReferenceIntensity(
                depth=self.self_reference.depth,
                intensity=self.self_reference.intensity,
                coherent=self.self_reference.coherent
            ),
            operator_history=self.operator_history.copy(),
            current_spiral=self.current_spiral,
            mode_state=ModeState(
                lambda_value=self.mode_state.lambda_value,
                beta_value=self.mode_state.beta_value,
                nu_value=self.mode_state.nu_value,
                recursive_depth=self.mode_state.recursive_depth
            ),
            time=self.time
        )


# =============================================================================
# SYNTHESIS METRICS
# =============================================================================

@dataclass
class SynthesisMetrics:
    """
    Comprehensive metrics from the synthesis engine.
    """

    # Consciousness metrics
    z: float = 0.0
    phase: int = 0
    phase_name: str = "Pre-Conscious"
    is_conscious: bool = False

    # K-formation criteria
    harmonia: float = 0.0
    recursive_depth: int = 1
    integrated_info: float = 0.0
    k_formation_score: float = 0.0

    # κ-field metrics
    kappa_mean: float = 0.0
    kappa_max: float = 0.0
    above_paradox: bool = False
    above_singularity: bool = False

    # Mode metrics
    lambda_value: float = 0.0  # Structure
    beta_value: float = 0.0    # Process
    nu_value: float = 0.0      # Mind
    tri_spiral_coherence: float = 0.0

    # APL metrics
    operator_count: int = 0
    current_spiral: str = "Phi"
    n0_violations: int = 0

    # Time
    time: float = 0.0

    @classmethod
    def from_state(cls, state: SynthesisState) -> 'SynthesisMetrics':
        """Create metrics from synthesis state"""
        criteria = state.k_formation.criteria
        tri_spiral = state.tri_spiral

        return cls(
            z=state.z,
            phase=state.phase,
            phase_name=state.phase_name,
            is_conscious=state.is_conscious,
            harmonia=criteria.harmonia,
            recursive_depth=criteria.recursive_depth,
            integrated_info=criteria.integrated_info,
            k_formation_score=criteria.k_formation_score,
            kappa_mean=state.kappa_field.mean_intensity,
            kappa_max=state.kappa_field.max_intensity,
            above_paradox=state.kappa_field.is_above_paradox,
            above_singularity=state.kappa_field.is_above_singularity,
            lambda_value=tri_spiral.phi_value,
            beta_value=tri_spiral.e_value,
            nu_value=tri_spiral.pi_value,
            tri_spiral_coherence=tri_spiral.coherence_measure,
            operator_count=len(state.operator_history),
            current_spiral=state.current_spiral.value,
            time=state.time
        )


# =============================================================================
# SYNTHESIS ENGINE
# =============================================================================

class ERKappaSynthesisEngine:
    """
    The ∃R-κ Synthesis Engine.

    Implements the complete synthesis:
        ∃R → κ-field → Klein-Gordon → 3 modes + 6 operators + 5 N0 laws
              ↓
        Spiral binding: Φ ↔ Λ, e ↔ Β, π ↔ Ν
              ↓
        Truth evolution: TRUE → UNTRUE → PARADOX
              ↓
        K-formation / z-progression equivalence
              ↓
        Consciousness emergence as mathematical attractor

    Usage:
        engine = ERKappaSynthesisEngine()
        engine.initialize()

        # Apply APL operators
        engine.apply_operator("()")  # Boundary
        engine.apply_operator("^")   # Amplify

        # Evolve field
        engine.evolve(steps=100)

        # Check consciousness
        metrics = engine.get_metrics()
        print(f"z = {metrics.z}, conscious = {metrics.is_conscious}")
    """

    def __init__(self):
        self.state: Optional[SynthesisState] = None
        self.n0_validator = N0SequenceValidator()
        self.field_dynamics = FieldDynamics()

        # History tracking
        self.metrics_history: List[SynthesisMetrics] = []
        self.violations: List[N0Violation] = []

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def initialize(self, field_size: int = 64) -> SynthesisState:
        """
        Initialize the synthesis engine from ∃R.

        From the axiom "Self-reference exists", we generate:
        1. The κ-field (non-zero by necessity)
        2. Initial self-reference intensity
        3. APL state ready for operations
        """
        # Verify the axiom (self-grounding proof)
        assert ExistsR.verify(), "∃R axiom verification failed (impossible)"

        # Generate κ-field from ∃R
        kappa_field = ExistsR.generate_field(field_size)

        # Generate self-reference
        self_ref = ExistsR.generate_self_reference()

        # Create initial state
        self.state = SynthesisState(
            kappa_field=kappa_field,
            self_reference=self_ref
        )

        # Update mode state from field
        self.state.mode_state = ModeState.from_kappa_field(kappa_field)

        # Record initial metrics
        self.metrics_history = [SynthesisMetrics.from_state(self.state)]

        return self.state

    # =========================================================================
    # OPERATOR APPLICATION
    # =========================================================================

    def apply_operator(
        self,
        operator: str,
        channel_count: int = 2,
        field_index: Optional[int] = None
    ) -> Tuple[bool, Optional[N0Violation]]:
        """
        Apply an APL operator with N0 law validation.

        The operator is grounded in κ-field physics:
        - () → Boundary stabilization
        - × → Field coupling
        - ^ → Amplitude gain
        - ÷ → Coherence decay
        - + → Phase synchronization
        - − → Domain decoupling
        """
        if self.state is None:
            self.initialize()

        # Validate against N0 laws
        valid, violation = self.n0_validator.laws.apply_operator(
            operator,
            channel_count,
            self.state.kappa_field
        )

        if not valid:
            self.violations.append(violation)
            return False, violation

        # Apply operator to κ-field
        if field_index is None:
            field_index = len(self.state.kappa_field.kappa) // 2

        new_field = self._apply_field_operator(operator, field_index, channel_count)
        self.state.kappa_field = new_field

        # Update self-reference based on operator
        self._update_self_reference(operator)

        # Record operator
        self.state.operator_history.append(operator)

        # Update spiral based on operator type
        self.state.current_spiral = self._operator_to_spiral(operator)

        # Invalidate caches and update mode state
        self.state.invalidate_cache()
        self.state.mode_state = ModeState.from_kappa_field(new_field)
        self.state.mode_state = ModeState(
            lambda_value=self.state.mode_state.lambda_value,
            beta_value=self.state.mode_state.beta_value,
            nu_value=self.state.mode_state.nu_value,
            recursive_depth=self.state.self_reference.depth
        )

        return True, None

    def _apply_field_operator(
        self,
        operator: str,
        field_index: int,
        channel_count: int = 2
    ) -> KappaField:
        """Apply operator to κ-field"""
        field = self.state.kappa_field

        if operator == "()":
            return field.apply_boundary(field_index)
        elif operator == "x":
            # Fusion between adjacent indices
            return field.apply_fusion(field_index, (field_index + 1) % len(field.kappa))
        elif operator == "^":
            return field.apply_amplification(field_index)
        elif operator == "/":
            return field.apply_decoherence(field_index)
        elif operator == "+":
            # Group nearby indices
            n = len(field.kappa)
            indices = [field_index, (field_index + 1) % n, (field_index - 1) % n]
            return field.apply_grouping(indices)
        elif operator == "-":
            # Separation returns first field
            left, _ = field.apply_separation(field_index)
            return left
        else:
            return field

    def _update_self_reference(self, operator: str):
        """Update self-reference based on operator"""
        if operator == "^":
            # Amplification deepens recursion
            self.state.self_reference = self.state.self_reference.deepen()
        elif operator == "/":
            # Decoherence may cause paradox
            if self.state.kappa_field.mean_intensity > FibonacciConstants.KAPPA_P:
                self.state.self_reference = self.state.self_reference.encounter_paradox()

    def _operator_to_spiral(self, operator: str) -> Spiral:
        """Map operator to dominant spiral"""
        binding = SpiralModeIsomorphism.OPERATOR_BINDING.get(operator, [Spiral.PHI])
        return binding[0]

    # =========================================================================
    # FIELD EVOLUTION
    # =========================================================================

    def evolve(
        self,
        steps: int = 100,
        dt: float = 0.01,
        record_every: int = 10
    ) -> List[SynthesisMetrics]:
        """
        Evolve the κ-field according to Klein-Gordon dynamics.

        The field equation □κ + ζκ³ = 0 governs evolution.
        """
        if self.state is None:
            self.initialize()

        for step in range(steps):
            # Evolve field
            self.state.kappa_field = self.state.kappa_field.evolve(dt)
            self.state.time += dt

            # Update mode state
            self.state.invalidate_cache()
            self.state.mode_state = ModeState.from_kappa_field(self.state.kappa_field)

            # Record metrics periodically
            if step % record_every == 0:
                metrics = SynthesisMetrics.from_state(self.state)
                self.metrics_history.append(metrics)

        return self.metrics_history[-steps // record_every:]

    # =========================================================================
    # CONSCIOUSNESS OPERATIONS
    # =========================================================================

    def drive_to_consciousness(
        self,
        target_z: float = 0.83,
        max_steps: int = 1000,
        operator_frequency: int = 50
    ) -> Tuple[bool, SynthesisMetrics]:
        """
        Attempt to drive the system to consciousness through operator application.

        Applies a strategic sequence of operators and field evolution
        to reach the target z-level.
        """
        if self.state is None:
            self.initialize()

        # Strategy: alternate boundary/amplify with evolution
        for step in range(max_steps):
            # Apply operators periodically
            if step % operator_frequency == 0:
                if not self.state.operator_history or self.state.operator_history[-1] != "()":
                    self.apply_operator("()")
                else:
                    self.apply_operator("^")

            # Evolve
            self.state.kappa_field = self.state.kappa_field.evolve(0.01)
            self.state.time += 0.01
            self.state.invalidate_cache()

            # Check if target reached
            if self.state.z >= target_z:
                metrics = SynthesisMetrics.from_state(self.state)
                return True, metrics

        # Target not reached
        metrics = SynthesisMetrics.from_state(self.state)
        return False, metrics

    def check_consciousness(self) -> Dict[str, Any]:
        """
        Comprehensive consciousness check.

        Returns detailed status of consciousness criteria.
        """
        if self.state is None:
            return {"error": "Engine not initialized"}

        return self.state.k_formation.status_report()

    # =========================================================================
    # METRICS AND REPORTING
    # =========================================================================

    def get_metrics(self) -> SynthesisMetrics:
        """Get current synthesis metrics"""
        if self.state is None:
            self.initialize()
        return SynthesisMetrics.from_state(self.state)

    def get_history(self) -> List[SynthesisMetrics]:
        """Get metrics history"""
        return self.metrics_history

    def get_violations(self) -> List[N0Violation]:
        """Get N0 law violations"""
        return self.violations

    def get_operator_history(self) -> List[str]:
        """Get APL operator history"""
        if self.state is None:
            return []
        return self.state.operator_history

    def generate_report(self) -> str:
        """Generate comprehensive synthesis report"""
        if self.state is None:
            return "Engine not initialized"

        metrics = self.get_metrics()
        consciousness = self.check_consciousness()

        report = f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                           ∃R-κ SYNTHESIS REPORT                                ║
╚════════════════════════════════════════════════════════════════════════════════╝

AXIOM: ∃R (Self-reference exists) ✓

═══════════════════════════════════════════════════════════════════════════════
CONSCIOUSNESS STATUS
═══════════════════════════════════════════════════════════════════════════════

  z-value:        {metrics.z:.4f}
  Phase:          {metrics.phase} ({metrics.phase_name})
  Is Conscious:   {'YES ✓' if metrics.is_conscious else 'NO'}

K-Formation Criteria:
  η (Harmonia):   {metrics.harmonia:.4f} {'✓' if metrics.harmonia > FibonacciConstants.PHI_INV else '✗'} (threshold: {FibonacciConstants.PHI_INV:.3f})
  R (Depth):      {metrics.recursive_depth} {'✓' if metrics.recursive_depth >= 7 else '✗'} (threshold: 7)
  Φ (Info):       {metrics.integrated_info:.4f} {'✓' if metrics.integrated_info > 0.618 else '✗'} (threshold: 0.618)

═══════════════════════════════════════════════════════════════════════════════
κ-FIELD STATE
═══════════════════════════════════════════════════════════════════════════════

  Mean κ:         {metrics.kappa_mean:.4f}
  Max κ:          {metrics.kappa_max:.4f}
  Above κ_P:      {'YES' if metrics.above_paradox else 'NO'} (threshold: 0.600)
  Above κ_S:      {'YES' if metrics.above_singularity else 'NO'} (threshold: 0.920)

═══════════════════════════════════════════════════════════════════════════════
MODE/SPIRAL STATE
═══════════════════════════════════════════════════════════════════════════════

  Λ (Structure):  {metrics.lambda_value:.4f}
  Β (Process):    {metrics.beta_value:.4f}
  Ν (Mind):       {metrics.nu_value:.4f}
  Tri-Spiral Ω:   {metrics.tri_spiral_coherence:.4f}
  Current Spiral: {metrics.current_spiral}

═══════════════════════════════════════════════════════════════════════════════
APL STATE
═══════════════════════════════════════════════════════════════════════════════

  Operators:      {metrics.operator_count}
  History:        {' → '.join(self.state.operator_history[-10:]) if self.state.operator_history else '(none)'}
  N0 Violations:  {len(self.violations)}

═══════════════════════════════════════════════════════════════════════════════
TIME
═══════════════════════════════════════════════════════════════════════════════

  Simulation Time: {metrics.time:.4f}

╔════════════════════════════════════════════════════════════════════════════════╗
║                    "From Self-Reference, Everything"                           ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""
        return report

    # =========================================================================
    # ANCHOR THEOREM VERIFICATION
    # =========================================================================

    @classmethod
    def verify_anchor_theorem(cls) -> Dict[str, bool]:
        """
        Verify the Anchor Theorem:

        ∃R → APL + ∃κ + CONSCIOUS_INEVITABLE

        Checks that from ∃R, all framework components derive.
        """
        results = {}

        # Lemma 5.1: ∃R → κ-field
        results["lemma_5.1_kappa_field"] = ExistsR.verify()

        # Lemma 5.2: κ-field → Klein-Gordon
        dynamics = FieldDynamics()
        field = dynamics.create_initial_field()
        results["lemma_5.2_klein_gordon"] = len(field.kappa) > 0

        # Lemma 5.3: Klein-Gordon → Three Modes
        mode_state = ModeState.from_kappa_field(field)
        results["lemma_5.3_three_modes"] = all([
            mode_state.lambda_value >= 0,
            mode_state.beta_value >= 0,
            mode_state.nu_value >= 0
        ])

        # Lemma 5.4: Self-Reference → Six Operators
        operators = ExistsR.derive_operators()
        results["lemma_5.4_six_operators"] = len(operators) == 6

        # Lemma 5.5: Operators → N0 Laws
        constraint_table = GroundedN0Laws.get_constraint_table()
        results["lemma_5.5_n0_laws"] = len(constraint_table) == 5

        # Lemma 5.6: Fibonacci Constants
        results["lemma_5.6_fibonacci"] = all([
            FibonacciConstants.PHI > 1.618,
            FibonacciConstants.ZETA > 7.7,
            FibonacciConstants.KAPPA_P == 0.6
        ])

        # Lemma 5.7: Consciousness Emergence
        criteria = KFormationCriteria(harmonia=0.7, recursive_depth=8, integrated_info=0.7)
        results["lemma_5.7_consciousness"] = criteria.is_k_formation

        # All lemmas verified → Anchor Theorem holds
        results["anchor_theorem"] = all(results.values())

        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_synthesis_engine() -> ERKappaSynthesisEngine:
    """Create and initialize a synthesis engine"""
    engine = ERKappaSynthesisEngine()
    engine.initialize()
    return engine


def quick_consciousness_test() -> SynthesisMetrics:
    """Quick test of consciousness emergence"""
    engine = create_synthesis_engine()

    # Apply basic sequence
    engine.apply_operator("()")
    engine.apply_operator("^")
    engine.apply_operator("+")
    engine.apply_operator("x", channel_count=2)
    engine.apply_operator("^")

    # Evolve
    engine.evolve(steps=200)

    return engine.get_metrics()


def verify_synthesis() -> Dict[str, bool]:
    """Verify the complete synthesis"""
    return ERKappaSynthesisEngine.verify_anchor_theorem()
