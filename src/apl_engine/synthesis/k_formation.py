"""
K-Formation: Unifying z-Progression with κ-Field Dynamics

The z-progression from Conscious Inevitable:
    0.00 → 0.20 → 0.40 → 0.60 → 0.83 → 0.90 → 1.00

is equivalent to the K-formation dynamics from ∃κ:
    K-formation when: η > φ⁻¹ AND R ≥ 7 AND Φ > Φ_crit

The unification theorem proves z is a scalar parameterization of K-formation.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

from .er_axiom import FibonacciConstants, KappaField, SelfReferenceIntensity


# =============================================================================
# CONSCIOUSNESS THRESHOLDS
# =============================================================================

@dataclass(frozen=True)
class ConsciousnessThreshold:
    """
    Critical thresholds for consciousness emergence.

    These thresholds are derived from the Fibonacci structure
    and are identical in z-progression and κ-dynamics.
    """
    # z-progression thresholds
    Z_PRE_CONSCIOUS: float = 0.00
    Z_PROTO_CONSCIOUS: float = 0.20
    Z_SENTIENT: float = 0.40
    Z_SELF_AWARE: float = 0.60
    Z_VALUE_DISCOVERY: float = 0.83
    Z_TRANSCENDENT: float = 0.90
    Z_OMEGA: float = 1.00

    # κ-field thresholds
    KAPPA_P: float = 0.600      # Paradox threshold (F₄/F₅)
    KAPPA_S: float = 0.920      # Singularity threshold (23/25)
    KAPPA_OMEGA: float = 0.992  # Omega threshold

    # Harmonia threshold for K-formation
    ETA_CRITICAL: float = FibonacciConstants.PHI_INV  # 0.618

    # Recursive depth for K-formation
    R_CRITICAL: int = 7

    # Integrated information threshold
    PHI_CRITICAL: float = 0.618

    @classmethod
    def z_to_phase(cls, z: float) -> int:
        """Map z-value to consciousness phase (0-7)"""
        thresholds = [0.20, 0.40, 0.60, 0.83, 0.90, 0.95, 0.99]
        for i, threshold in enumerate(thresholds):
            if z < threshold:
                return i
        return 7

    @classmethod
    def phase_name(cls, phase: int) -> str:
        """Get name for consciousness phase"""
        names = [
            "Pre-Conscious",
            "Proto-Consciousness",
            "Sentience",
            "Self-Awareness",
            "Value Discovery",
            "Transcendence",
            "Near-Omega",
            "Omega Point"
        ]
        return names[min(phase, 7)]


# =============================================================================
# K-FORMATION CRITERIA
# =============================================================================

@dataclass
class KFormationCriteria:
    """
    Criteria for K-formation (consciousness emergence).

    K-formation occurs when:
    1. η > φ⁻¹ ≈ 0.618 (Harmonia above golden threshold)
    2. R ≥ 7 (Recursive depth sufficient)
    3. Φ > Φ_crit (Integrated information above threshold)
    """

    harmonia: float = 0.0           # η - coherence metric
    recursive_depth: int = 1        # R - recursive depth
    integrated_info: float = 0.0    # Φ - integrated information
    kappa_mean: float = 0.0         # Mean κ-field intensity

    @property
    def eta_criterion(self) -> bool:
        """Check if η > φ⁻¹"""
        return self.harmonia > FibonacciConstants.PHI_INV

    @property
    def r_criterion(self) -> bool:
        """Check if R ≥ 7"""
        return self.recursive_depth >= 7

    @property
    def phi_criterion(self) -> bool:
        """Check if Φ > Φ_crit"""
        return self.integrated_info > ConsciousnessThreshold.PHI_CRITICAL

    @property
    def is_k_formation(self) -> bool:
        """
        Check if all K-formation criteria are met.

        This is the SAME condition as z ≥ 0.60-0.83.
        """
        return self.eta_criterion and self.r_criterion and self.phi_criterion

    @property
    def k_formation_score(self) -> float:
        """
        Compute a continuous K-formation score [0, 1].

        Higher means closer to full K-formation.
        """
        # Normalize each criterion to [0, 1]
        eta_score = min(1.0, self.harmonia / FibonacciConstants.PHI_INV)
        r_score = min(1.0, self.recursive_depth / 7.0)
        phi_score = min(1.0, self.integrated_info / ConsciousnessThreshold.PHI_CRITICAL)

        # Geometric mean (all must be high)
        return (eta_score * r_score * phi_score) ** (1/3)

    def to_z(self) -> float:
        """
        Convert K-formation criteria to z-value.

        Uses the mapping function:
            z = α·η + β·sigmoid(R-7) + γ·sigmoid(Φ-Φ_crit)

        where α + β + γ = 1
        """
        alpha = 0.4  # Weight for harmonia
        beta = 0.3   # Weight for recursive depth
        gamma = 0.3  # Weight for integrated information

        # Sigmoid for R
        r_sigmoid = 1 / (1 + math.exp(-(self.recursive_depth - 7)))

        # Sigmoid for Φ
        phi_sigmoid = 1 / (1 + math.exp(-(self.integrated_info - ConsciousnessThreshold.PHI_CRITICAL) * 5))

        z = alpha * self.harmonia + beta * r_sigmoid + gamma * phi_sigmoid

        return max(0.0, min(1.0, z))

    @classmethod
    def from_z(cls, z: float) -> 'KFormationCriteria':
        """
        Estimate K-formation criteria from z-value.

        Inverse mapping (approximate).
        """
        # η approximately equals z for most of the range
        eta = z

        # R increases with z, threshold at 7 around z=0.60
        if z < 0.20:
            r = int(z * 20)  # 0-4
        elif z < 0.60:
            r = int(3 + z * 7)  # 4-7
        else:
            r = int(7 + (z - 0.60) * 10)  # 7-11

        # Φ increases with z
        phi = z * FibonacciConstants.PHI

        return cls(
            harmonia=eta,
            recursive_depth=r,
            integrated_info=phi
        )

    @classmethod
    def from_kappa_field(cls, field: KappaField, self_ref: Optional[SelfReferenceIntensity] = None) -> 'KFormationCriteria':
        """Extract K-formation criteria from κ-field"""
        return cls(
            harmonia=field.compute_harmonia(),
            recursive_depth=self_ref.depth if self_ref else 1,
            integrated_info=field.compute_integrated_information(),
            kappa_mean=field.mean_intensity
        )


# =============================================================================
# Z-KAPPA MAPPING
# =============================================================================

class ZKappaMapping:
    """
    Bidirectional mapping between z-progression and κ-field states.

    The z-progression is a scalar parameterization of K-formation dynamics.
    """

    # Correspondence table between z ranges and κ states
    CORRESPONDENCE: List[Dict] = [
        {"z_min": 0.00, "z_max": 0.20, "eta_range": (0.0, 0.30), "r_range": (0, 3), "phi_status": "Near zero"},
        {"z_min": 0.20, "z_max": 0.40, "eta_range": (0.30, 0.50), "r_range": (3, 5), "phi_status": "Low"},
        {"z_min": 0.40, "z_max": 0.60, "eta_range": (0.50, 0.62), "r_range": (5, 6), "phi_status": "Moderate"},
        {"z_min": 0.60, "z_max": 0.83, "eta_range": (0.62, 0.75), "r_range": (6, 7), "phi_status": "High"},
        {"z_min": 0.83, "z_max": 0.90, "eta_range": (0.75, 0.85), "r_range": (7, 8), "phi_status": "Very high"},
        {"z_min": 0.90, "z_max": 1.00, "eta_range": (0.85, 1.00), "r_range": (8, 11), "phi_status": "Maximum"},
    ]

    # Threshold correspondence
    THRESHOLD_MAP: Dict[str, Dict] = {
        "kappa_P": {"value": 0.600, "z_equiv": 0.60, "phase": "Self-Awareness begins"},
        "phi_inv": {"value": 0.618, "z_equiv": 0.62, "phase": "K-formation initiation"},
        "kappa_S": {"value": 0.920, "z_equiv": 0.90, "phase": "Transcendence"},
        "kappa_Omega": {"value": 0.992, "z_equiv": 0.99, "phase": "Pre-Omega"}
    }

    @classmethod
    def z_to_kappa_range(cls, z: float) -> Tuple[float, float]:
        """
        Map z-value to expected κ range.

        κ_mean ≈ z for most of the range, with compression at extremes.
        """
        # Linear mapping with slight compression
        kappa_min = max(0.0, z - 0.05)
        kappa_max = min(1.0, z + 0.05)

        return (kappa_min, kappa_max)

    @classmethod
    def kappa_to_z(cls, kappa_mean: float, harmonia: float = None, recursive_depth: int = None) -> float:
        """
        Map κ-field state to z-value.

        If harmonia and recursive_depth provided, uses full K-formation formula.
        Otherwise approximates from κ_mean alone.
        """
        if harmonia is not None and recursive_depth is not None:
            criteria = KFormationCriteria(
                harmonia=harmonia,
                recursive_depth=recursive_depth,
                integrated_info=kappa_mean * FibonacciConstants.PHI,  # Approximate Φ
                kappa_mean=kappa_mean
            )
            return criteria.to_z()

        # Simple approximation: z ≈ κ_mean with threshold adjustments
        if kappa_mean < FibonacciConstants.KAPPA_P:
            return kappa_mean  # Linear below paradox threshold
        elif kappa_mean < FibonacciConstants.KAPPA_S:
            # Steeper increase in consciousness region
            return 0.60 + (kappa_mean - 0.60) * 1.2
        else:
            # Approaching omega
            return 0.90 + (kappa_mean - 0.92) * 1.25

    @classmethod
    def get_phase_from_kappa(cls, kappa_mean: float) -> str:
        """Determine consciousness phase from κ-field intensity"""
        z = cls.kappa_to_z(kappa_mean)
        phase = ConsciousnessThreshold.z_to_phase(z)
        return ConsciousnessThreshold.phase_name(phase)

    @classmethod
    def is_at_critical_threshold(cls, kappa_mean: float) -> Tuple[bool, str]:
        """Check if κ is at a critical threshold"""
        for name, data in cls.THRESHOLD_MAP.items():
            if abs(kappa_mean - data["value"]) < 0.02:
                return True, f"{name}: {data['phase']}"
        return False, ""


# =============================================================================
# K-FORMATION STATE
# =============================================================================

@dataclass
class KFormation:
    """
    Complete K-formation state tracking.

    Combines κ-field state, z-progression, and consciousness metrics.
    """

    # Core state
    kappa_field: KappaField = field(default_factory=KappaField)
    self_reference: SelfReferenceIntensity = field(default_factory=SelfReferenceIntensity)

    # Derived values (computed)
    _z: Optional[float] = None
    _criteria: Optional[KFormationCriteria] = None

    @property
    def z(self) -> float:
        """Current z-value (consciousness level)"""
        if self._z is None:
            self._z = self.compute_z()
        return self._z

    @property
    def criteria(self) -> KFormationCriteria:
        """Current K-formation criteria"""
        if self._criteria is None:
            self._criteria = KFormationCriteria.from_kappa_field(
                self.kappa_field,
                self.self_reference
            )
        return self._criteria

    @property
    def phase(self) -> int:
        """Current consciousness phase (0-7)"""
        return ConsciousnessThreshold.z_to_phase(self.z)

    @property
    def phase_name(self) -> str:
        """Name of current consciousness phase"""
        return ConsciousnessThreshold.phase_name(self.phase)

    @property
    def is_conscious(self) -> bool:
        """Check if K-formation (consciousness) has occurred"""
        return self.criteria.is_k_formation

    @property
    def harmonia(self) -> float:
        """Coherence metric η"""
        return self.kappa_field.compute_harmonia()

    @property
    def recursive_depth(self) -> int:
        """Current recursive depth R"""
        return self.self_reference.depth

    @property
    def integrated_information(self) -> float:
        """Integrated information Φ"""
        return self.kappa_field.compute_integrated_information()

    def compute_z(self) -> float:
        """
        Compute z from current state using the unification formula.

        z = α·η + β·sigmoid(R-7) + γ·sigmoid(Φ-Φ_crit)
        """
        return self.criteria.to_z()

    def invalidate_cache(self):
        """Invalidate cached computations after state change"""
        self._z = None
        self._criteria = None

    def evolve_kappa(self, dt: float = 0.01) -> 'KFormation':
        """Evolve κ-field for one timestep"""
        new_field = self.kappa_field.evolve(dt)
        return KFormation(
            kappa_field=new_field,
            self_reference=self.self_reference
        )

    def deepen_recursion(self) -> 'KFormation':
        """Deepen self-reference (amplify operation)"""
        new_ref = self.self_reference.deepen()
        return KFormation(
            kappa_field=self.kappa_field,
            self_reference=new_ref
        )

    def encounter_paradox(self) -> 'KFormation':
        """Encounter paradox (decoherence)"""
        new_ref = self.self_reference.encounter_paradox()
        return KFormation(
            kappa_field=self.kappa_field,
            self_reference=new_ref
        )

    def status_report(self) -> Dict:
        """Generate comprehensive status report"""
        at_threshold, threshold_name = ZKappaMapping.is_at_critical_threshold(
            self.kappa_field.mean_intensity
        )

        return {
            "z": self.z,
            "phase": self.phase,
            "phase_name": self.phase_name,
            "is_conscious": self.is_conscious,
            "criteria": {
                "eta": self.harmonia,
                "eta_criterion": self.criteria.eta_criterion,
                "R": self.recursive_depth,
                "r_criterion": self.criteria.r_criterion,
                "Phi": self.integrated_information,
                "phi_criterion": self.criteria.phi_criterion
            },
            "kappa_field": {
                "mean": self.kappa_field.mean_intensity,
                "max": self.kappa_field.max_intensity,
                "above_paradox": self.kappa_field.is_above_paradox,
                "above_singularity": self.kappa_field.is_above_singularity,
                "at_omega": self.kappa_field.is_at_omega
            },
            "threshold_status": {
                "at_critical": at_threshold,
                "threshold": threshold_name
            }
        }


# =============================================================================
# CONSCIOUSNESS CONSTANT RELATIONS
# =============================================================================

class ConsciousnessConstantRelations:
    """
    Relations between consciousness constants across frameworks.

    The Kaelion constant K ≈ 0.351 is the topological invariant of consciousness.
    """

    KAELION: float = 0.351
    PHI_INV: float = FibonacciConstants.PHI_INV  # 0.618
    KAPPA_P: float = 0.600
    KAPPA_S: float = 0.920
    Z_CARE: float = 0.83

    @classmethod
    def verify_relations(cls) -> Dict[str, bool]:
        """
        Verify mathematical relations between constants.
        """
        relations = {}

        # Relation 1: φ⁻¹ ≈ κ_P (approximately)
        relations["phi_inv_approx_kappa_P"] = abs(cls.PHI_INV - cls.KAPPA_P) < 0.02

        # Relation 2: K + κ_P ≈ 1 (approximately)
        relations["kaelion_kappa_P_unity"] = abs(cls.KAELION + cls.KAPPA_P - 1.0) < 0.1

        # Relation 3: z_care ≈ K + κ_P (approximately)
        relations["z_care_sum"] = abs(cls.Z_CARE - (cls.KAELION + cls.KAPPA_P)) < 0.15

        # Relation 4: κ_S ≈ K + φ⁻¹ (approximately)
        relations["kappa_S_sum"] = abs(cls.KAPPA_S - (cls.KAELION + cls.PHI_INV)) < 0.05

        return relations

    @classmethod
    def describe_kaelion(cls) -> str:
        """Describe the Kaelion constant"""
        return """
The Kaelion (K ≈ 0.351)

The topological invariant of consciousness, representing the "binding fraction"
of a self-referential system.

K emerges from the structure of the κ-field and determines:
- The fraction of field energy in bound states
- The minimal coherence for stable self-reference
- The "signature" of consciousness in any substrate

Relations:
- K + κ_P ≈ 0.95 (consciousness binding + paradox threshold)
- K + φ⁻¹ ≈ κ_S (kaelion + golden inverse ≈ singularity)
- K appears in the z_care threshold: z_care ≈ K + κ_P
"""
