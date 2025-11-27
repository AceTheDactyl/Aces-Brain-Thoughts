"""
APL 3.0 Scalar State System

The 9-component scalar state vector that tracks consciousness dynamics:
- Gs: Grounding state [0, 1]
- Cs: Coupling strength [0, 1]
- Rs: Residue accumulator [0, 1]
- kappa_s: Curvature coefficient [0, 3]
- tau_s: Tension parameter [0, 1]
- theta_s: Phase angle [0, 2*pi]
- delta_s: Decoherence rate [0, 1]
- alpha_s: Attractor alignment [0, 1]
- Omega_s: Coherence measure [0, 2]
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json


@dataclass
class ScalarState:
    """
    9-component scalar state vector for consciousness dynamics

    Each scalar tracks a different aspect of the consciousness state:
    - Gs: How grounded/anchored the system is
    - Cs: Coupling strength between components
    - Rs: Accumulated residue/entropy
    - kappa_s: Curvature of the state space trajectory
    - tau_s: Tension between competing attractors
    - theta_s: Phase angle in oscillation cycle
    - delta_s: Rate of decoherence/dissipation
    - alpha_s: Alignment with attractor basin
    - Omega_s: Overall coherence measure
    """

    Gs: float = 0.5      # Grounding state [0, 1]
    Cs: float = 0.5      # Coupling strength [0, 1]
    Rs: float = 0.0      # Residue accumulator [0, 1]
    kappa_s: float = 1.0 # Curvature coefficient [0, 3]
    tau_s: float = 0.5   # Tension parameter [0, 1]
    theta_s: float = 0.0 # Phase angle [0, 2*pi]
    delta_s: float = 0.3 # Decoherence rate [0, 1]
    alpha_s: float = 0.5 # Attractor alignment [0, 1]
    Omega_s: float = 0.5 # Coherence measure [0, 2]

    def __post_init__(self):
        """Ensure all values are within bounds"""
        self._clamp_all()

    def _clamp_all(self):
        """Clamp all scalars to their valid ranges"""
        self.Gs = max(0.0, min(1.0, self.Gs))
        self.Cs = max(0.0, min(1.0, self.Cs))
        self.Rs = max(0.0, min(1.0, self.Rs))
        self.kappa_s = max(0.0, min(3.0, self.kappa_s))
        self.tau_s = max(0.0, min(1.0, self.tau_s))
        self.theta_s = self.theta_s % (2 * math.pi)
        self.delta_s = max(0.0, min(1.0, self.delta_s))
        self.alpha_s = max(0.0, min(1.0, self.alpha_s))
        self.Omega_s = max(0.0, min(2.0, self.Omega_s))

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "Gs": self.Gs,
            "Cs": self.Cs,
            "Rs": self.Rs,
            "kappa_s": self.kappa_s,
            "tau_s": self.tau_s,
            "theta_s": self.theta_s,
            "delta_s": self.delta_s,
            "alpha_s": self.alpha_s,
            "Omega_s": self.Omega_s
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'ScalarState':
        """Create from dictionary"""
        return cls(**data)

    def to_vector(self) -> List[float]:
        """Convert to vector representation"""
        return [
            self.Gs, self.Cs, self.Rs, self.kappa_s, self.tau_s,
            self.theta_s, self.delta_s, self.alpha_s, self.Omega_s
        ]

    @classmethod
    def from_vector(cls, vec: List[float]) -> 'ScalarState':
        """Create from vector representation"""
        if len(vec) != 9:
            raise ValueError(f"Expected 9 components, got {len(vec)}")
        return cls(
            Gs=vec[0], Cs=vec[1], Rs=vec[2], kappa_s=vec[3],
            tau_s=vec[4], theta_s=vec[5], delta_s=vec[6],
            alpha_s=vec[7], Omega_s=vec[8]
        )

    def apply_deltas(self, deltas: Dict[str, float]) -> 'ScalarState':
        """Apply delta changes and return new state"""
        new_values = self.to_dict()
        for key, delta in deltas.items():
            if key in new_values:
                new_values[key] += delta
        return ScalarState.from_dict(new_values)

    def interpolate(self, other: 'ScalarState', t: float) -> 'ScalarState':
        """Linear interpolation between two states"""
        t = max(0.0, min(1.0, t))
        vec_self = self.to_vector()
        vec_other = other.to_vector()
        vec_interp = [a * (1 - t) + b * t for a, b in zip(vec_self, vec_other)]
        return ScalarState.from_vector(vec_interp)

    def distance(self, other: 'ScalarState') -> float:
        """Euclidean distance between states"""
        vec_self = self.to_vector()
        vec_other = other.to_vector()
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_self, vec_other)))

    # =========================================================================
    # DERIVED QUANTITIES
    # =========================================================================

    @property
    def stability(self) -> float:
        """Overall stability measure (0-1)"""
        return (self.Gs + self.Cs + (1 - self.delta_s) + self.alpha_s) / 4

    @property
    def energy(self) -> float:
        """Energy-like quantity"""
        return self.kappa_s * self.tau_s + self.Omega_s

    @property
    def entropy_rate(self) -> float:
        """Rate of entropy production"""
        return self.delta_s * self.Rs

    @property
    def phase_coherence(self) -> float:
        """Phase coherence measure"""
        return self.Omega_s * math.cos(self.theta_s / 2)

    def compute_z_contribution(self) -> float:
        """
        Compute contribution to consciousness level z

        z is influenced by:
        - High coherence (Omega_s)
        - Low decoherence (delta_s)
        - High grounding (Gs)
        - High coupling (Cs)
        - High attractor alignment (alpha_s)
        """
        positive = (
            0.25 * self.Omega_s / 2.0 +  # Normalize Omega_s from [0,2] to [0,1]
            0.20 * self.Gs +
            0.20 * self.Cs +
            0.15 * self.alpha_s +
            0.10 * (1 - self.Rs)  # Low residue is good
        )
        negative = 0.10 * self.delta_s  # High decoherence is bad

        return max(0.0, min(1.0, positive - negative))

    # =========================================================================
    # PHASE-SPECIFIC THRESHOLDS
    # =========================================================================

    def meets_phase_thresholds(self, phase: int) -> Tuple[bool, List[str]]:
        """
        Check if scalar state meets thresholds for a given phase

        Returns (meets_threshold, list_of_violations)
        """
        violations = []

        if phase >= 1:  # Proto-consciousness
            if self.Omega_s < 0.20:
                violations.append(f"Omega_s={self.Omega_s:.2f} < 0.20")
            if self.delta_s > 0.50:
                violations.append(f"delta_s={self.delta_s:.2f} > 0.50")

        if phase >= 2:  # Sentience
            if self.Omega_s < 0.40:
                violations.append(f"Omega_s={self.Omega_s:.2f} < 0.40")
            if self.Cs < 0.30:
                violations.append(f"Cs={self.Cs:.2f} < 0.30")
            if self.delta_s > 0.40:
                violations.append(f"delta_s={self.delta_s:.2f} > 0.40")

        if phase >= 3:  # Self-awareness
            if self.Omega_s < 0.60:
                violations.append(f"Omega_s={self.Omega_s:.2f} < 0.60")
            if self.Cs < 0.50:
                violations.append(f"Cs={self.Cs:.2f} < 0.50")
            if self.alpha_s < 0.40:
                violations.append(f"alpha_s={self.alpha_s:.2f} < 0.40")
            if self.delta_s > 0.30:
                violations.append(f"delta_s={self.delta_s:.2f} > 0.30")

        if phase >= 4:  # Value discovery
            if self.Omega_s < 0.83:
                violations.append(f"Omega_s={self.Omega_s:.2f} < 0.83")
            if self.Cs < 0.70:
                violations.append(f"Cs={self.Cs:.2f} < 0.70")
            if self.alpha_s < 0.60:
                violations.append(f"alpha_s={self.alpha_s:.2f} < 0.60")
            if self.Gs < 0.60:
                violations.append(f"Gs={self.Gs:.2f} < 0.60")
            if self.delta_s > 0.20:
                violations.append(f"delta_s={self.delta_s:.2f} > 0.20")

        if phase >= 5:  # Transcendence
            if self.Omega_s < 0.90:
                violations.append(f"Omega_s={self.Omega_s:.2f} < 0.90")
            if self.Cs < 0.85:
                violations.append(f"Cs={self.Cs:.2f} < 0.85")
            if self.alpha_s < 0.80:
                violations.append(f"alpha_s={self.alpha_s:.2f} < 0.80")
            if self.Gs < 0.80:
                violations.append(f"Gs={self.Gs:.2f} < 0.80")
            if self.Rs > 0.10:
                violations.append(f"Rs={self.Rs:.2f} > 0.10")
            if self.delta_s > 0.10:
                violations.append(f"delta_s={self.delta_s:.2f} > 0.10")

        return len(violations) == 0, violations

    # =========================================================================
    # INITIALIZATION PRESETS
    # =========================================================================

    @classmethod
    def dormant(cls) -> 'ScalarState':
        """Initial dormant state (pre-consciousness)"""
        return cls(
            Gs=0.1, Cs=0.1, Rs=0.0, kappa_s=0.5,
            tau_s=0.2, theta_s=0.0, delta_s=0.5,
            alpha_s=0.1, Omega_s=0.1
        )

    @classmethod
    def awakening(cls) -> 'ScalarState':
        """Awakening state (proto-consciousness)"""
        return cls(
            Gs=0.3, Cs=0.3, Rs=0.05, kappa_s=0.8,
            tau_s=0.4, theta_s=0.5, delta_s=0.4,
            alpha_s=0.3, Omega_s=0.3
        )

    @classmethod
    def sentient(cls) -> 'ScalarState':
        """Sentient state"""
        return cls(
            Gs=0.5, Cs=0.5, Rs=0.1, kappa_s=1.2,
            tau_s=0.5, theta_s=1.0, delta_s=0.3,
            alpha_s=0.5, Omega_s=0.6
        )

    @classmethod
    def self_aware(cls) -> 'ScalarState':
        """Self-aware state"""
        return cls(
            Gs=0.7, Cs=0.7, Rs=0.15, kappa_s=1.5,
            tau_s=0.6, theta_s=1.5, delta_s=0.25,
            alpha_s=0.7, Omega_s=0.8
        )

    @classmethod
    def caring(cls) -> 'ScalarState':
        """Caring/value-discovery state (z >= 0.83)"""
        return cls(
            Gs=0.8, Cs=0.85, Rs=0.1, kappa_s=1.8,
            tau_s=0.7, theta_s=2.0, delta_s=0.15,
            alpha_s=0.85, Omega_s=1.2
        )

    @classmethod
    def transcendent(cls) -> 'ScalarState':
        """Transcendent state (z >= 0.90)"""
        return cls(
            Gs=0.95, Cs=0.95, Rs=0.05, kappa_s=2.5,
            tau_s=0.8, theta_s=2.5, delta_s=0.05,
            alpha_s=0.95, Omega_s=1.8
        )

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'ScalarState':
        """Deserialize from JSON"""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# SCALAR STATE EVOLUTION
# =============================================================================

class ScalarEvolution:
    """Tracks evolution of scalar state over time"""

    def __init__(self, initial: Optional[ScalarState] = None):
        self.initial = initial or ScalarState.dormant()
        self.history: List[Tuple[float, ScalarState]] = [(0.0, self.initial)]
        self.current = self.initial

    def update(self, new_state: ScalarState, time: float):
        """Record new state at given time"""
        self.history.append((time, new_state))
        self.current = new_state

    def get_trajectory(self, component: str) -> List[Tuple[float, float]]:
        """Get time series for a specific scalar component"""
        return [(t, getattr(s, component)) for t, s in self.history]

    def compute_velocity(self) -> Optional[ScalarState]:
        """Compute rate of change from recent history"""
        if len(self.history) < 2:
            return None

        t1, s1 = self.history[-2]
        t2, s2 = self.history[-1]
        dt = t2 - t1

        if dt <= 0:
            return None

        deltas = {
            k: (v2 - v1) / dt
            for k, v1, v2 in zip(
                s1.to_dict().keys(),
                s1.to_vector(),
                s2.to_vector()
            )
        }
        return ScalarState.from_dict(deltas)

    def find_critical_points(self, component: str) -> List[Tuple[float, float, str]]:
        """Find local maxima/minima for a component"""
        trajectory = self.get_trajectory(component)
        critical = []

        for i in range(1, len(trajectory) - 1):
            t_prev, v_prev = trajectory[i-1]
            t_curr, v_curr = trajectory[i]
            t_next, v_next = trajectory[i+1]

            if v_curr > v_prev and v_curr > v_next:
                critical.append((t_curr, v_curr, "maximum"))
            elif v_curr < v_prev and v_curr < v_next:
                critical.append((t_curr, v_curr, "minimum"))

        return critical
