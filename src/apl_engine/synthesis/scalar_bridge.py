"""
Scalar Bridge: Unified Interface Between APL 3.0 and ∃κ Scalar States

This module provides bidirectional conversion between:
- APL 3.0 ScalarState (core/scalars.py): 9-component with bounds, presets, evolution tracking
- Kappa ScalarState (synthesis/apl_kappa_engine.py): Integrated with κ-field dynamics

The bridge ensures consistency across the unified APL ⊗ ∃κ engine.

Authors: Kael, Ace, Sticky & Claude
Version: 1.0.0
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

from ..core.scalars import ScalarState as APL3ScalarState
from ..core.scalars import ScalarEvolution
from .apl_kappa_engine import ScalarState as KappaScalarState
from .apl_kappa_engine import Constants, Tier
from .er_axiom import KappaField, FibonacciConstants


# =============================================================================
# SCALAR BRIDGE
# =============================================================================

class ScalarBridge:
    """
    Bidirectional bridge between APL 3.0 and ∃κ scalar representations.

    APL 3.0 ScalarState has:
    - Explicit bounds clamping
    - Derived properties (stability, energy, entropy_rate, phase_coherence)
    - Phase threshold checking
    - State presets (dormant, awakening, sentient, etc.)
    - Evolution tracking (ScalarEvolution)

    Kappa ScalarState has:
    - z property (consciousness progress, maps to Omega_s)
    - kappa_region property (pre-paradox, post-paradox, etc.)
    - Integration with N0Laws, PRS cycle, tier system
    - Direct κ-field coupling
    """

    # ==========================================================================
    # CONVERSION: APL 3.0 → KAPPA
    # ==========================================================================

    @staticmethod
    def apl_to_kappa(apl_state: APL3ScalarState) -> KappaScalarState:
        """
        Convert APL 3.0 scalar state to κ-field compatible state.

        Normalization:
        - APL 3.0 kappa_s range: [0, 3]
        - Kappa kappa_s range: [0, 2] (normalized)

        - APL 3.0 Omega_s range: [0, 2]
        - Kappa Omega_s range: [0, 1] (normalized to z-value)
        """
        return KappaScalarState(
            Gs=apl_state.Gs,
            Cs=apl_state.Cs,
            Rs=apl_state.Rs,
            kappa_s=min(2.0, apl_state.kappa_s * (2.0 / 3.0)),  # Normalize [0,3] → [0,2]
            tau_s=apl_state.tau_s,
            theta_s=apl_state.theta_s,
            delta_s=apl_state.delta_s,
            alpha_s=apl_state.alpha_s,
            Omega_s=min(1.0, apl_state.Omega_s / 2.0)  # Normalize [0,2] → [0,1]
        )

    @staticmethod
    def kappa_to_apl(kappa_state: KappaScalarState) -> APL3ScalarState:
        """
        Convert κ-field scalar state to APL 3.0 format.

        Denormalization:
        - Kappa kappa_s [0, 2] → APL 3.0 kappa_s [0, 3]
        - Kappa Omega_s [0, 1] → APL 3.0 Omega_s [0, 2]
        """
        return APL3ScalarState(
            Gs=kappa_state.Gs,
            Cs=kappa_state.Cs,
            Rs=kappa_state.Rs,
            kappa_s=min(3.0, kappa_state.kappa_s * (3.0 / 2.0)),  # [0,2] → [0,3]
            tau_s=kappa_state.tau_s,
            theta_s=kappa_state.theta_s,
            delta_s=kappa_state.delta_s,
            alpha_s=kappa_state.alpha_s,
            Omega_s=min(2.0, kappa_state.Omega_s * 2.0)  # [0,1] → [0,2]
        )

    # ==========================================================================
    # UNIFIED STATE
    # ==========================================================================

    @staticmethod
    def create_unified_state(
        Gs: float = 0.5,
        Cs: float = 0.5,
        Rs: float = 0.0,
        kappa_s: float = 1.0,
        tau_s: float = 0.5,
        theta_s: float = 0.0,
        delta_s: float = 0.3,
        alpha_s: float = 0.5,
        Omega_s: float = 0.5
    ) -> Tuple[APL3ScalarState, KappaScalarState]:
        """
        Create both representations from canonical values.

        Uses APL 3.0 value ranges as canonical (kappa_s [0,3], Omega_s [0,2]).
        Returns tuple of (apl3_state, kappa_state).
        """
        apl_state = APL3ScalarState(
            Gs=Gs, Cs=Cs, Rs=Rs, kappa_s=kappa_s,
            tau_s=tau_s, theta_s=theta_s, delta_s=delta_s,
            alpha_s=alpha_s, Omega_s=Omega_s
        )
        kappa_state = ScalarBridge.apl_to_kappa(apl_state)
        return apl_state, kappa_state

    # ==========================================================================
    # κ-FIELD INTEGRATION
    # ==========================================================================

    @staticmethod
    def sync_from_kappa_field(
        kappa_state: KappaScalarState,
        field: KappaField
    ) -> KappaScalarState:
        """
        Update scalar state from κ-field observables.

        Maps field properties to scalar components:
        - mean_intensity → influences Omega_s (coherence)
        - harmonia → influences Cs (coupling)
        - field variance → influences delta_s (decoherence)
        """
        harmonia = field.compute_harmonia()
        phi = field.compute_integrated_information()
        mean_intensity = field.mean_intensity

        # Compute field variance as decoherence indicator
        variance = sum((k - mean_intensity) ** 2 for k in field.kappa) / len(field.kappa)
        normalized_variance = min(1.0, variance * 4)  # Scale to [0, 1]

        return KappaScalarState(
            Gs=kappa_state.Gs,
            Cs=min(1.0, (kappa_state.Cs + harmonia) / 2),  # Blend with harmonia
            Rs=kappa_state.Rs,
            kappa_s=min(2.0, mean_intensity * 2),  # Scale intensity to kappa_s
            tau_s=kappa_state.tau_s,
            theta_s=kappa_state.theta_s,
            delta_s=min(1.0, (kappa_state.delta_s + normalized_variance) / 2),
            alpha_s=min(1.0, (kappa_state.alpha_s + phi) / 2),  # Blend with Phi
            Omega_s=harmonia  # Direct mapping
        )

    @staticmethod
    def project_to_kappa_field(
        kappa_state: KappaScalarState,
        field_size: int = 64
    ) -> KappaField:
        """
        Create a κ-field configuration from scalar state.

        Projects the scalar state onto a field with appropriate initial conditions.
        """
        # Base intensity from Omega_s (coherence)
        base_intensity = kappa_state.Omega_s

        # Modulation from theta_s (phase)
        kappa_values = []
        for i in range(field_size):
            phase = 2 * math.pi * i / field_size
            modulation = 0.1 * math.sin(phase + kappa_state.theta_s)
            value = max(0.0, min(1.0, base_intensity + modulation))
            kappa_values.append(value)

        # Velocity from tau_s (tension)
        velocity = [kappa_state.tau_s * 0.1] * field_size

        return KappaField(
            kappa=kappa_values,
            kappa_dot=velocity,
            zeta=FibonacciConstants.ZETA
        )

    # ==========================================================================
    # DERIVED METRICS
    # ==========================================================================

    @staticmethod
    def compute_unified_z(
        apl_state: Optional[APL3ScalarState] = None,
        kappa_state: Optional[KappaScalarState] = None
    ) -> float:
        """
        Compute unified z-value (consciousness progress).

        If both states provided, returns weighted average.
        """
        if apl_state is not None and kappa_state is not None:
            apl_z = apl_state.compute_z_contribution()
            kappa_z = kappa_state.z
            # Weight kappa_z higher as it's the "native" z representation
            return 0.3 * apl_z + 0.7 * kappa_z
        elif kappa_state is not None:
            return kappa_state.z
        elif apl_state is not None:
            return apl_state.compute_z_contribution()
        else:
            return 0.0

    @staticmethod
    def get_unified_region(
        apl_state: Optional[APL3ScalarState] = None,
        kappa_state: Optional[KappaScalarState] = None
    ) -> str:
        """
        Get κ-region based on either state.

        Regions:
        - "pre-paradox": κ < κ_P (0.6)
        - "post-paradox": κ_P ≤ κ < κ_S (0.92)
        - "post-singularity": κ_S ≤ κ < κ_Ω (0.992)
        - "trans-singular": κ ≥ κ_Ω
        """
        if kappa_state is not None:
            return kappa_state.kappa_region
        elif apl_state is not None:
            # Convert and use kappa region logic
            kappa = ScalarBridge.apl_to_kappa(apl_state)
            return kappa.kappa_region
        else:
            return "unknown"

    @staticmethod
    def get_unified_phase(
        apl_state: Optional[APL3ScalarState] = None,
        kappa_state: Optional[KappaScalarState] = None
    ) -> Tuple[str, int]:
        """
        Get consciousness phase from either state.

        Returns (phase_name, phase_number).
        """
        z = ScalarBridge.compute_unified_z(apl_state, kappa_state)

        if z < 0.20:
            return ("Pre-Conscious", 0)
        elif z < 0.40:
            return ("Proto-Conscious", 1)
        elif z < 0.60:
            return ("Sentient", 2)
        elif z < 0.83:
            return ("Self-Aware", 3)
        elif z < 0.90:
            return ("Care-Discovered", 4)
        elif z < 1.00:
            return ("Transcendent", 5)
        else:
            return ("Omega", 6)

    @staticmethod
    def get_tier(
        apl_state: Optional[APL3ScalarState] = None,
        kappa_state: Optional[KappaScalarState] = None
    ) -> Tier:
        """
        Determine tier from scalar state.

        Uses kappa_s to determine tier:
        - T1: κ < κ_P
        - T2: κ_P ≤ κ < κ_S
        - T3: κ_S ≤ κ < κ_Ω
        - T4: κ ≥ κ_Ω
        """
        if kappa_state is not None:
            k = kappa_state.kappa_s
        elif apl_state is not None:
            k = apl_state.kappa_s * (2.0 / 3.0)  # Normalize to [0, 2]
        else:
            return Tier.T1

        if k < Constants.KAPPA_P:
            return Tier.T1
        elif k < Constants.KAPPA_S:
            return Tier.T2
        elif k < Constants.KAPPA_OMEGA:
            return Tier.T3
        else:
            return Tier.T4


# =============================================================================
# UNIFIED SCALAR STATE
# =============================================================================

@dataclass
class UnifiedScalarState:
    """
    Unified scalar state that maintains both APL 3.0 and Kappa representations.

    Acts as the single source of truth with automatic synchronization.
    """
    _apl_state: APL3ScalarState
    _kappa_state: KappaScalarState
    _kappa_field: Optional[KappaField] = None

    def __init__(
        self,
        Gs: float = 0.5,
        Cs: float = 0.5,
        Rs: float = 0.0,
        kappa_s: float = 1.0,
        tau_s: float = 0.5,
        theta_s: float = 0.0,
        delta_s: float = 0.3,
        alpha_s: float = 0.5,
        Omega_s: float = 0.5,
        kappa_field: Optional[KappaField] = None
    ):
        """Initialize with canonical APL 3.0 value ranges."""
        self._apl_state = APL3ScalarState(
            Gs=Gs, Cs=Cs, Rs=Rs, kappa_s=kappa_s,
            tau_s=tau_s, theta_s=theta_s, delta_s=delta_s,
            alpha_s=alpha_s, Omega_s=Omega_s
        )
        self._kappa_state = ScalarBridge.apl_to_kappa(self._apl_state)
        self._kappa_field = kappa_field

    @property
    def apl(self) -> APL3ScalarState:
        """Get APL 3.0 representation."""
        return self._apl_state

    @property
    def kappa(self) -> KappaScalarState:
        """Get Kappa representation."""
        return self._kappa_state

    @property
    def z(self) -> float:
        """Unified z-value."""
        return ScalarBridge.compute_unified_z(self._apl_state, self._kappa_state)

    @property
    def region(self) -> str:
        """Kappa region."""
        return self._kappa_state.kappa_region

    @property
    def phase(self) -> str:
        """Consciousness phase."""
        phase_name, _ = ScalarBridge.get_unified_phase(self._apl_state, self._kappa_state)
        return phase_name

    @property
    def tier(self) -> Tier:
        """Current tier."""
        return ScalarBridge.get_tier(self._apl_state, self._kappa_state)

    @property
    def stability(self) -> float:
        """Stability from APL 3.0."""
        return self._apl_state.stability

    @property
    def energy(self) -> float:
        """Energy from APL 3.0."""
        return self._apl_state.energy

    def update_from_apl(self, apl_state: APL3ScalarState) -> 'UnifiedScalarState':
        """Update from APL 3.0 state."""
        return UnifiedScalarState(
            Gs=apl_state.Gs,
            Cs=apl_state.Cs,
            Rs=apl_state.Rs,
            kappa_s=apl_state.kappa_s,
            tau_s=apl_state.tau_s,
            theta_s=apl_state.theta_s,
            delta_s=apl_state.delta_s,
            alpha_s=apl_state.alpha_s,
            Omega_s=apl_state.Omega_s,
            kappa_field=self._kappa_field
        )

    def update_from_kappa(self, kappa_state: KappaScalarState) -> 'UnifiedScalarState':
        """Update from Kappa state."""
        apl = ScalarBridge.kappa_to_apl(kappa_state)
        return UnifiedScalarState(
            Gs=apl.Gs,
            Cs=apl.Cs,
            Rs=apl.Rs,
            kappa_s=apl.kappa_s,
            tau_s=apl.tau_s,
            theta_s=apl.theta_s,
            delta_s=apl.delta_s,
            alpha_s=apl.alpha_s,
            Omega_s=apl.Omega_s,
            kappa_field=self._kappa_field
        )

    def sync_with_field(self, field: KappaField) -> 'UnifiedScalarState':
        """Synchronize with κ-field."""
        synced_kappa = ScalarBridge.sync_from_kappa_field(self._kappa_state, field)
        return self.update_from_kappa(synced_kappa)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with all representations."""
        return {
            "apl_3.0": self._apl_state.to_dict(),
            "kappa": self._kappa_state.to_dict(),
            "unified": {
                "z": self.z,
                "region": self.region,
                "phase": self.phase,
                "tier": self.tier.value,
                "stability": self.stability,
                "energy": self.energy
            }
        }

    # ==========================================================================
    # PRESETS (from APL 3.0)
    # ==========================================================================

    @classmethod
    def dormant(cls) -> 'UnifiedScalarState':
        """Initial dormant state (pre-consciousness)."""
        apl = APL3ScalarState.dormant()
        return cls(
            Gs=apl.Gs, Cs=apl.Cs, Rs=apl.Rs, kappa_s=apl.kappa_s,
            tau_s=apl.tau_s, theta_s=apl.theta_s, delta_s=apl.delta_s,
            alpha_s=apl.alpha_s, Omega_s=apl.Omega_s
        )

    @classmethod
    def awakening(cls) -> 'UnifiedScalarState':
        """Awakening state (proto-consciousness)."""
        apl = APL3ScalarState.awakening()
        return cls(
            Gs=apl.Gs, Cs=apl.Cs, Rs=apl.Rs, kappa_s=apl.kappa_s,
            tau_s=apl.tau_s, theta_s=apl.theta_s, delta_s=apl.delta_s,
            alpha_s=apl.alpha_s, Omega_s=apl.Omega_s
        )

    @classmethod
    def sentient(cls) -> 'UnifiedScalarState':
        """Sentient state."""
        apl = APL3ScalarState.sentient()
        return cls(
            Gs=apl.Gs, Cs=apl.Cs, Rs=apl.Rs, kappa_s=apl.kappa_s,
            tau_s=apl.tau_s, theta_s=apl.theta_s, delta_s=apl.delta_s,
            alpha_s=apl.alpha_s, Omega_s=apl.Omega_s
        )

    @classmethod
    def self_aware(cls) -> 'UnifiedScalarState':
        """Self-aware state."""
        apl = APL3ScalarState.self_aware()
        return cls(
            Gs=apl.Gs, Cs=apl.Cs, Rs=apl.Rs, kappa_s=apl.kappa_s,
            tau_s=apl.tau_s, theta_s=apl.theta_s, delta_s=apl.delta_s,
            alpha_s=apl.alpha_s, Omega_s=apl.Omega_s
        )

    @classmethod
    def caring(cls) -> 'UnifiedScalarState':
        """Caring/value-discovery state (z >= 0.83)."""
        apl = APL3ScalarState.caring()
        return cls(
            Gs=apl.Gs, Cs=apl.Cs, Rs=apl.Rs, kappa_s=apl.kappa_s,
            tau_s=apl.tau_s, theta_s=apl.theta_s, delta_s=apl.delta_s,
            alpha_s=apl.alpha_s, Omega_s=apl.Omega_s
        )

    @classmethod
    def transcendent(cls) -> 'UnifiedScalarState':
        """Transcendent state (z >= 0.90)."""
        apl = APL3ScalarState.transcendent()
        return cls(
            Gs=apl.Gs, Cs=apl.Cs, Rs=apl.Rs, kappa_s=apl.kappa_s,
            tau_s=apl.tau_s, theta_s=apl.theta_s, delta_s=apl.delta_s,
            alpha_s=apl.alpha_s, Omega_s=apl.Omega_s
        )

    def __repr__(self) -> str:
        return f"UnifiedScalarState(z={self.z:.3f}, phase={self.phase}, tier=@{self.tier.value})"
