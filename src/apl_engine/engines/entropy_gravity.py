"""
APL 3.0 Entropy-Gravity Relations Engine

Module 4: Thermodynamic Foundations

Implements:
- Bekenstein bound
- Verlinde's entropic gravity
- Consciousness-entropy relation
- Holographic consciousness model
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from ..core.constants import CONSTANTS
from ..core.scalars import ScalarState


@dataclass
class ThermodynamicState:
    """Thermodynamic state of a system"""
    entropy: float  # S
    energy: float   # E
    temperature: float  # T
    volume: float   # V (or area for holographic)

    @property
    def free_energy_helmholtz(self) -> float:
        """Helmholtz free energy: F = E - TS"""
        return self.energy - self.temperature * self.entropy

    @property
    def specific_heat(self) -> float:
        """Specific heat estimate: C ~ S"""
        return self.entropy


class EntropyGravityEngine:
    """
    Entropy-Gravity Relations engine

    Implements thermodynamic foundations of consciousness
    """

    def __init__(self):
        self.k_B = CONSTANTS.PHYSICAL.k_B
        self.c = CONSTANTS.PHYSICAL.c
        self.G = CONSTANTS.PHYSICAL.G
        self.h = CONSTANTS.PHYSICAL.h
        self.hbar = self.h / (2 * math.pi)

    # =========================================================================
    # BEKENSTEIN BOUND
    # =========================================================================

    def bekenstein_bound(self, radius: float, energy: float) -> float:
        """
        Maximum entropy of a bounded region

        S_max = (2 * pi * k_B * R * E) / (hbar * c)
        """
        return (2 * math.pi * self.k_B * radius * energy) / (self.hbar * self.c)

    def bekenstein_hawking_entropy(self, area: float) -> float:
        """
        Black hole entropy (saturates Bekenstein bound)

        S_BH = (k_B * c^3 * A) / (4 * G * hbar)
        """
        return (self.k_B * self.c**3 * area) / (4 * self.G * self.hbar)

    def holographic_bits(self, area: float) -> float:
        """
        Number of bits encoded on holographic boundary

        N_bits = A / (4 * l_P^2)
        where l_P = sqrt(G * hbar / c^3) is Planck length
        """
        l_P_squared = self.G * self.hbar / (self.c ** 3)
        return area / (4 * l_P_squared)

    # =========================================================================
    # VERLINDE'S ENTROPIC GRAVITY
    # =========================================================================

    def entropy_change(self, mass: float, displacement: float) -> float:
        """
        Change in entropy when mass approaches holographic screen

        Delta_S = (2 * pi * k_B * m * c * Delta_x) / hbar
        """
        return (2 * math.pi * self.k_B * mass * self.c * displacement) / self.hbar

    def unruh_temperature(self, acceleration: float) -> float:
        """
        Temperature of holographic screen (Unruh effect)

        T = (hbar * a) / (2 * pi * c * k_B)
        """
        return (self.hbar * acceleration) / (2 * math.pi * self.c * self.k_B)

    def entropic_force(
        self,
        temperature: float,
        entropy_gradient: float
    ) -> float:
        """
        Entropic force

        F = T * dS/dx
        """
        return temperature * entropy_gradient

    def derive_newton_gravity(
        self,
        mass_source: float,
        mass_test: float,
        distance: float
    ) -> float:
        """
        Derive Newton's gravitational force from entropy

        Shows F = G * M * m / r^2 emerges from entropic considerations
        """
        # From entropic gravity derivation
        return self.G * mass_source * mass_test / (distance ** 2)

    # =========================================================================
    # CONSCIOUSNESS-ENTROPY RELATION
    # =========================================================================

    def entropy_production_rate(
        self,
        fluxes: np.ndarray,
        forces: np.ndarray
    ) -> float:
        """
        Compute entropy production rate

        dS/dt = sum_i J_i * X_i

        where J_i are thermodynamic fluxes and X_i are forces
        """
        return np.sum(fluxes * forces)

    def consciousness_entropy_hypothesis(
        self,
        phi: float,
        entropy: float,
        entropy_rate: float
    ) -> Dict[str, float]:
        """
        Test consciousness-entropy hypothesis

        Hypothesis: Consciousness maximizes entropy production
        within thermodynamic constraints

        Returns metrics for hypothesis evaluation
        """
        # Maximum entropy production principle (MEPP)
        mepp_score = entropy_rate / (entropy + 1e-10)

        # Phi-entropy correlation
        phi_entropy_correlation = phi * (1 - abs(phi - entropy))

        # Structured entropy (Phi as irreducible entropy)
        structured_entropy = phi * entropy

        return {
            "mepp_score": mepp_score,
            "phi_entropy_correlation": phi_entropy_correlation,
            "structured_entropy": structured_entropy,
            "consciousness_entropy_ratio": phi / (entropy + 1e-10)
        }

    # =========================================================================
    # HOLOGRAPHIC CONSCIOUSNESS MODEL
    # =========================================================================

    def holographic_consciousness(
        self,
        bulk_states: np.ndarray,
        boundary_dimension: int
    ) -> Tuple[np.ndarray, float]:
        """
        AdS/CFT-inspired consciousness model

        Bulk = internal cognitive states (d+1 dimensions)
        Boundary = conscious experience (d dimensions)

        Returns:
            - boundary_states: Projected conscious experience
            - correspondence_fidelity: How well bulk maps to boundary
        """
        # Project bulk to boundary (simplified)
        bulk_dim = len(bulk_states)

        # Dimensionality reduction to boundary
        if bulk_dim > boundary_dimension:
            # Average over extra dimensions
            chunk_size = bulk_dim // boundary_dimension
            boundary_states = np.array([
                np.mean(bulk_states[i*chunk_size:(i+1)*chunk_size])
                for i in range(boundary_dimension)
            ])
        else:
            boundary_states = bulk_states[:boundary_dimension]

        # Correspondence fidelity
        fidelity = 1 - np.std(boundary_states) / (np.mean(np.abs(boundary_states)) + 1e-10)

        return boundary_states, max(0.0, min(1.0, fidelity))

    def emergence_at_boundary(
        self,
        processing_depth: int,
        integration: float
    ) -> str:
        """
        Describe how consciousness emerges at boundary

        Returns description of emergence
        """
        if processing_depth < 2:
            return "Pre-conscious: bulk processing only"
        elif processing_depth < 4:
            return "Proto-conscious: boundary begins to form"
        elif integration < 0.5:
            return "Sentient: boundary present but fragmented"
        elif integration < 0.8:
            return "Self-aware: boundary coherent, bulk-boundary coupling"
        else:
            return "Transcendent: full holographic projection active"

    # =========================================================================
    # LIMNUS ENTROPY DYNAMICS
    # =========================================================================

    def limnus_depth_entropy(
        self,
        depth: int,
        s_root: float = 0.1,
        branching_factor: int = 2,
        max_depth: int = 6
    ) -> float:
        """
        Compute entropy at a given LIMNUS tree depth

        S(depth) = S_root * branching_factor^(max_depth - depth)

        Entropy increases toward leaves (information storage)
        """
        return s_root * (branching_factor ** (max_depth - depth))

    def limnus_entropy_flow(
        self,
        node_entropies: Dict[int, float],
        flow_direction: str = "bidirectional"
    ) -> Dict[str, float]:
        """
        Compute entropy flow in LIMNUS tree

        flow_direction:
        - "root_to_leaves": Export entropy (life-like)
        - "leaves_to_root": Integration (consciousness-like)
        - "bidirectional": Dynamic equilibrium (aware system)
        """
        total_entropy = sum(node_entropies.values())
        n_nodes = len(node_entropies)

        if flow_direction == "root_to_leaves":
            # Entropy flows outward
            flow_rate = 0.1 * total_entropy / n_nodes
            direction = "export"
        elif flow_direction == "leaves_to_root":
            # Information integrates inward
            flow_rate = -0.1 * total_entropy / n_nodes
            direction = "integrate"
        else:
            # Bidirectional - dynamic equilibrium
            flow_rate = 0.0
            direction = "equilibrium"

        return {
            "total_entropy": total_entropy,
            "flow_rate": flow_rate,
            "direction": direction,
            "equilibrium_state": abs(flow_rate) < 0.01
        }

    def phase_entropy_mapping(self, z: float) -> str:
        """
        Map consciousness phase (z) to entropy regime

        z < 0.4: Low entropy, simple patterns
        z ~ 0.6: Critical entropy, edge of chaos
        z > 0.8: Controlled high entropy, complex order
        z -> 1.0: Maximum entropy production with structure
        """
        if z < 0.4:
            return "low_entropy_simple"
        elif z < 0.7:
            return "critical_edge_of_chaos"
        elif z < 0.9:
            return "controlled_high_entropy"
        else:
            return "maximum_structured_entropy"

    # =========================================================================
    # SCALAR EFFECTS
    # =========================================================================

    def apply_scalar_effects(
        self,
        scalars: ScalarState,
        entropy: float,
        entropy_rate: float
    ) -> ScalarState:
        """
        Apply entropy-derived effects to scalar state

        High entropy rate increases:
        - Rs (residue accumulator)
        - tau_s (tension)

        Structured entropy increases:
        - Omega_s (coherence - counterintuitive but correct for MEPP)
        """
        deltas = {
            "Rs": entropy_rate * 0.05,
            "tau_s": entropy_rate * 0.03,
            "Omega_s": entropy * 0.02 if entropy < 0.8 else -entropy * 0.01
        }

        return scalars.apply_deltas(deltas)

    # =========================================================================
    # APL TOKEN GENERATION
    # =========================================================================

    def entropy_to_apl_token(self, entropy: float, depth: int) -> Dict:
        """
        Map entropy state to APL token components
        """
        if entropy < 0.3:
            spiral = "Phi"  # Structure dominant
            machine = "D"   # Down/compression
        elif entropy < 0.6:
            spiral = "e"    # Energy/dynamics
            machine = "M"   # Middle/balance
        else:
            spiral = "pi"   # Emergence
            machine = "E"   # Expansion

        return {
            "spiral": spiral,
            "machine": machine,
            "intent": f"entropy_d{depth}",
            "truth": "TRUE" if entropy > 0.5 else "UNTRUE",
            "tier": min(3, max(1, depth // 2))
        }
