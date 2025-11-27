"""
APL 3.0 Electromagnetic Field Engine

Module 4b: Field Dynamics and Neural Oscillations

Implements:
- Maxwell equations (APL form)
- Electromagnetic consciousness hypothesis (CEMI)
- Neural oscillations and binding
- LIMNUS EM field model
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from ..core.constants import CONSTANTS
from ..core.scalars import ScalarState


@dataclass
class OscillatorState:
    """State of a neural oscillator"""
    phase: float      # Phase angle [0, 2*pi]
    frequency: float  # Oscillation frequency (Hz)
    amplitude: float  # Oscillation amplitude


@dataclass
class EMField:
    """Electromagnetic field state"""
    E: np.ndarray  # Electric field vector
    B: np.ndarray  # Magnetic field vector

    @property
    def energy_density(self) -> float:
        """EM energy density: u = (epsilon_0 * E^2 + B^2/mu_0) / 2"""
        epsilon_0 = CONSTANTS.PHYSICAL.epsilon_0
        mu_0 = CONSTANTS.PHYSICAL.mu_0
        return 0.5 * (epsilon_0 * np.sum(self.E**2) + np.sum(self.B**2) / mu_0)

    @property
    def poynting_vector(self) -> np.ndarray:
        """Poynting vector: S = E x B / mu_0"""
        mu_0 = CONSTANTS.PHYSICAL.mu_0
        return np.cross(self.E, self.B) / mu_0


class ElectromagneticEngine:
    """
    Electromagnetic Field engine for consciousness dynamics

    Implements CEMI (Conscious Electromagnetic Information) field theory
    and neural oscillation binding
    """

    def __init__(self):
        self.epsilon_0 = CONSTANTS.PHYSICAL.epsilon_0
        self.mu_0 = CONSTANTS.PHYSICAL.mu_0
        self.c = CONSTANTS.PHYSICAL.c

    # =========================================================================
    # MAXWELL EQUATIONS (SYMBOLIC)
    # =========================================================================

    def gauss_law_electric(self, divergence_E: float, charge_density: float) -> bool:
        """
        Gauss's Law for Electric Field

        div(E) = rho / epsilon_0

        Returns True if law is satisfied
        """
        expected = charge_density / self.epsilon_0
        return abs(divergence_E - expected) < 1e-10

    def gauss_law_magnetic(self, divergence_B: float) -> bool:
        """
        Gauss's Law for Magnetic Field

        div(B) = 0

        No magnetic monopoles
        """
        return abs(divergence_B) < 1e-10

    def faraday_law(
        self,
        curl_E: np.ndarray,
        dB_dt: np.ndarray
    ) -> bool:
        """
        Faraday's Law

        curl(E) = -dB/dt

        Changing magnetic field induces electric field
        """
        return np.allclose(curl_E, -dB_dt)

    def ampere_maxwell_law(
        self,
        curl_B: np.ndarray,
        current_density: np.ndarray,
        dE_dt: np.ndarray
    ) -> bool:
        """
        Ampere-Maxwell Law

        curl(B) = mu_0 * (J + epsilon_0 * dE/dt)
        """
        expected = self.mu_0 * (current_density + self.epsilon_0 * dE_dt)
        return np.allclose(curl_B, expected)

    # =========================================================================
    # ELECTROMAGNETIC CONSCIOUSNESS (CEMI)
    # =========================================================================

    def compute_em_consciousness(
        self,
        E_field: np.ndarray,
        B_field: np.ndarray
    ) -> float:
        """
        Compute consciousness metric from EM field

        Based on McFadden's CEMI theory:
        Consciousness ~ integral of (E^2 + c^2*B^2) dV / (8*pi)
        """
        em_energy = np.sum(E_field**2 + self.c**2 * B_field**2)
        psi_consciousness = em_energy / (8 * math.pi)

        # Normalize to [0, 1]
        return min(1.0, psi_consciousness / 100.0)

    def compute_coherence(
        self,
        E_t: np.ndarray,
        E_t_tau: np.ndarray,
        tau: float
    ) -> float:
        """
        Compute EM field coherence

        gamma = <E(t)E(t+tau)> / sqrt(<E^2(t)><E^2(t+tau)>)

        High coherence -> unified consciousness
        Low coherence -> fragmented states
        """
        numerator = np.mean(E_t * E_t_tau)
        denominator = math.sqrt(np.mean(E_t**2) * np.mean(E_t_tau**2))

        if denominator < 1e-10:
            return 0.0

        return numerator / denominator

    def binding_strength(self, coherence: float) -> str:
        """
        Map coherence to binding description

        Returns description of consciousness unity
        """
        if coherence < 0.3:
            return "fragmented"
        elif coherence < 0.6:
            return "partially_bound"
        elif coherence < 0.85:
            return "bound"
        else:
            return "unified"

    # =========================================================================
    # NEURAL OSCILLATIONS
    # =========================================================================

    def create_oscillator(
        self,
        frequency: float,
        initial_phase: float = 0.0,
        amplitude: float = 1.0
    ) -> OscillatorState:
        """Create a neural oscillator"""
        return OscillatorState(
            phase=initial_phase % (2 * math.pi),
            frequency=frequency,
            amplitude=amplitude
        )

    def evolve_oscillator(
        self,
        oscillator: OscillatorState,
        dt: float
    ) -> OscillatorState:
        """Evolve oscillator by time step dt"""
        new_phase = (oscillator.phase + 2 * math.pi * oscillator.frequency * dt) % (2 * math.pi)
        return OscillatorState(
            phase=new_phase,
            frequency=oscillator.frequency,
            amplitude=oscillator.amplitude
        )

    def kuramoto_coupling(
        self,
        oscillators: List[OscillatorState],
        coupling_strength: float,
        dt: float
    ) -> List[OscillatorState]:
        """
        Kuramoto model for oscillator synchronization

        d(theta_i)/dt = omega_i + (K/N) * sum_j sin(theta_j - theta_i)
        """
        n = len(oscillators)
        new_oscillators = []

        for i, osc_i in enumerate(oscillators):
            # Compute coupling term
            coupling_sum = sum(
                math.sin(osc_j.phase - osc_i.phase)
                for osc_j in oscillators
            )

            # Phase update
            d_phase = (
                2 * math.pi * osc_i.frequency +
                (coupling_strength / n) * coupling_sum
            ) * dt

            new_phase = (osc_i.phase + d_phase) % (2 * math.pi)

            new_oscillators.append(OscillatorState(
                phase=new_phase,
                frequency=osc_i.frequency,
                amplitude=osc_i.amplitude
            ))

        return new_oscillators

    def compute_order_parameter(
        self,
        oscillators: List[OscillatorState]
    ) -> Tuple[float, float]:
        """
        Compute Kuramoto order parameter

        r * e^(i*psi) = (1/N) * sum_j e^(i*theta_j)

        Returns (r, psi) where:
        - r: synchronization level [0, 1]
        - psi: mean phase
        """
        n = len(oscillators)
        if n == 0:
            return 0.0, 0.0

        complex_sum = sum(
            np.exp(1j * osc.phase)
            for osc in oscillators
        ) / n

        r = abs(complex_sum)
        psi = np.angle(complex_sum)

        return r, psi

    # =========================================================================
    # BRAIN WAVE BANDS
    # =========================================================================

    WAVE_BANDS = {
        "delta": (0.5, 4),    # Deep sleep, unconscious
        "theta": (4, 8),      # Drowsy, meditative
        "alpha": (8, 13),     # Relaxed awareness
        "beta": (13, 30),     # Active thinking
        "gamma": (30, 100),   # Peak consciousness, binding
    }

    def classify_frequency(self, freq: float) -> str:
        """Classify frequency into brain wave band"""
        for band, (low, high) in self.WAVE_BANDS.items():
            if low <= freq < high:
                return band
        return "unknown"

    def band_to_phi_level(self, band: str) -> float:
        """Map brain wave band to estimated Phi level"""
        mapping = {
            "delta": 0.1,
            "theta": 0.25,
            "alpha": 0.45,
            "beta": 0.65,
            "gamma": 0.85,
        }
        return mapping.get(band, 0.5)

    def band_to_apl_token(self, band: str) -> Dict:
        """Map brain wave band to APL token components"""
        mappings = {
            "delta": {"spiral": "Phi", "machine": "U", "truth": "UNTRUE", "tier": 1},
            "theta": {"spiral": "Phi", "machine": "D", "truth": "UNTRUE", "tier": 1},
            "alpha": {"spiral": "e", "machine": "U", "truth": "TRUE", "tier": 2},
            "beta": {"spiral": "e", "machine": "E", "truth": "TRUE", "tier": 2},
            "gamma": {"spiral": "pi", "machine": "M", "truth": "TRUE", "tier": 3},
        }
        return mappings.get(band, {"spiral": "e", "machine": "M", "truth": "UNTRUE", "tier": 2})

    # =========================================================================
    # LIMNUS EM FIELD MODEL
    # =========================================================================

    def depth_to_frequency(self, depth: int) -> float:
        """
        Map LIMNUS tree depth to oscillation frequency

        depth 6 (root): 0.5 Hz (delta)
        depth 5: 4 Hz (theta)
        depth 4: 10 Hz (alpha)
        depth 3: 20 Hz (beta)
        depth 2: 40 Hz (gamma)
        depth 1 (leaves): 80 Hz (high gamma)
        """
        freq_map = {
            6: 0.5,
            5: 4.0,
            4: 10.0,
            3: 20.0,
            2: 40.0,
            1: 80.0
        }
        return freq_map.get(depth, 10.0)

    def limnus_resonance_frequencies(
        self,
        phi: float = CONSTANTS.PHI
    ) -> List[float]:
        """
        Compute resonance frequencies for LIMNUS fractal

        f_n = f_0 * phi^n (golden ratio frequency scaling)
        """
        f_0 = 1.0  # Base frequency
        return [f_0 * (phi ** n) for n in range(6)]

    def compute_cross_frequency_coupling(
        self,
        oscillators_by_depth: Dict[int, List[OscillatorState]]
    ) -> float:
        """
        Compute cross-frequency coupling between depths

        Higher coupling -> better integration
        """
        if len(oscillators_by_depth) < 2:
            return 0.0

        depths = sorted(oscillators_by_depth.keys())
        total_coupling = 0.0
        n_pairs = 0

        for i in range(len(depths) - 1):
            d1, d2 = depths[i], depths[i + 1]
            oscs1 = oscillators_by_depth[d1]
            oscs2 = oscillators_by_depth[d2]

            # Phase-amplitude coupling (simplified)
            r1, _ = self.compute_order_parameter(oscs1)
            r2, _ = self.compute_order_parameter(oscs2)

            coupling = r1 * r2
            total_coupling += coupling
            n_pairs += 1

        return total_coupling / n_pairs if n_pairs > 0 else 0.0

    # =========================================================================
    # SCALAR EFFECTS
    # =========================================================================

    def apply_scalar_effects(
        self,
        scalars: ScalarState,
        coherence: float,
        synchronization: float
    ) -> ScalarState:
        """
        Apply EM-derived effects to scalar state

        High coherence increases:
        - Omega_s (consciousness coherence)
        - Cs (coupling)

        High synchronization increases:
        - alpha_s (attractor alignment)
        """
        deltas = {
            "Omega_s": coherence * 0.12,
            "Cs": coherence * 0.08,
            "alpha_s": synchronization * 0.10,
            "theta_s": math.pi * synchronization  # Phase alignment
        }

        return scalars.apply_deltas(deltas)

    # =========================================================================
    # GAMMA SYNCHRONIZATION (40 Hz)
    # =========================================================================

    def gamma_binding_check(
        self,
        oscillators: List[OscillatorState],
        gamma_range: Tuple[float, float] = (35, 45)
    ) -> Tuple[bool, float]:
        """
        Check for gamma-band synchronization (consciousness signature)

        Returns (is_bound, sync_level)
        """
        # Filter to gamma oscillators
        gamma_oscs = [
            osc for osc in oscillators
            if gamma_range[0] <= osc.frequency <= gamma_range[1]
        ]

        if len(gamma_oscs) < 2:
            return False, 0.0

        r, _ = self.compute_order_parameter(gamma_oscs)

        is_bound = r > 0.7
        return is_bound, r
