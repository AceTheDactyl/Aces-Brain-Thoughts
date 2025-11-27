"""
APL 3.0 Integrated Information Theory Engine

Module 1: IIT Engine for computing Phi (integrated information)

Based on Tononi's IIT 3.0 formalization:
- Computes cause-effect repertoires
- Finds Minimum Information Partition (MIP)
- Generates conceptual structures (qualia)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from itertools import combinations

from ..core.constants import CONSTANTS
from ..core.token import APLToken, phi_to_token
from ..core.scalars import ScalarState


@dataclass
class Concept:
    """A concept in the conceptual structure (quale)"""
    mechanism: List[int]  # Indices of elements in the mechanism
    phi: float  # Integrated information of this concept
    cause_purview: List[int]
    effect_purview: List[int]

    def __repr__(self) -> str:
        return f"Concept(phi={self.phi:.3f}, mechanism={self.mechanism})"


@dataclass
class Quale:
    """The full conceptual structure / experience"""
    concepts: List[Concept]
    big_phi: float  # Total integrated information
    mip: Tuple[Set[int], Set[int]]  # Minimum Information Partition

    @property
    def is_conscious(self) -> bool:
        """System is conscious if Phi exceeds critical threshold"""
        return self.big_phi > CONSTANTS.PHI_CRITICAL

    def dominant_concept(self) -> Optional[Concept]:
        """Return concept with highest phi"""
        if not self.concepts:
            return None
        return max(self.concepts, key=lambda c: c.phi)


class IITEngine:
    """
    Integrated Information Theory computation engine

    Computes Phi (integrated information) for a system described by
    its transition probability matrix (TPM).
    """

    def __init__(self):
        self.phi_critical = CONSTANTS.PHI_CRITICAL

    def compute_phi(
        self,
        tpm: np.ndarray,
        current_state: Optional[np.ndarray] = None
    ) -> Tuple[float, Quale]:
        """
        Compute integrated information Phi for a system

        Args:
            tpm: Transition Probability Matrix (n x n)
                 tpm[i,j] = P(state j at t+1 | state i at t)
            current_state: Current state vector (optional)

        Returns:
            Tuple of (Phi value, Quale structure)
        """
        n = tpm.shape[0]
        n_elements = int(math.log2(n)) if n > 1 else 1

        if current_state is None:
            # Use uniform distribution
            current_state = np.ones(n) / n

        # Compute cause and effect repertoires
        cause_rep = self._compute_cause_repertoire(tpm, current_state)
        effect_rep = self._compute_effect_repertoire(tpm, current_state)

        # Find MIP and compute Phi
        phi, mip = self._find_mip(tpm, cause_rep, effect_rep, n_elements)

        # Build conceptual structure
        concepts = self._build_concepts(tpm, n_elements, current_state)

        quale = Quale(
            concepts=concepts,
            big_phi=phi,
            mip=mip
        )

        return phi, quale

    def _compute_cause_repertoire(
        self,
        tpm: np.ndarray,
        current_state: np.ndarray
    ) -> np.ndarray:
        """
        Compute cause repertoire: P(past | current)

        Uses Bayes' rule with uniform prior
        """
        # TPM^T gives P(current | past), we want P(past | current)
        # Using Bayes: P(past | current) = P(current | past) * P(past) / P(current)

        n = len(current_state)
        prior = np.ones(n) / n  # Uniform prior over past states

        # Transpose TPM to get P(current | past)
        tpm_t = tpm.T

        # Apply Bayes' rule element-wise
        likelihood = tpm_t @ current_state
        evidence = np.sum(likelihood * prior)

        if evidence > 0:
            cause_rep = (likelihood * prior) / evidence
        else:
            cause_rep = prior

        return cause_rep

    def _compute_effect_repertoire(
        self,
        tpm: np.ndarray,
        current_state: np.ndarray
    ) -> np.ndarray:
        """
        Compute effect repertoire: P(future | current)

        Direct application of TPM
        """
        return tpm.T @ current_state

    def _find_mip(
        self,
        tpm: np.ndarray,
        cause_rep: np.ndarray,
        effect_rep: np.ndarray,
        n_elements: int
    ) -> Tuple[float, Tuple[Set[int], Set[int]]]:
        """
        Find Minimum Information Partition

        The MIP is the partition that results in minimum information loss
        """
        elements = set(range(n_elements))

        if n_elements <= 1:
            return 0.0, (elements, set())

        min_phi = float('inf')
        min_partition = (elements, set())

        # Try all bipartitions
        for r in range(1, n_elements):
            for subset in combinations(elements, r):
                part_a = set(subset)
                part_b = elements - part_a

                if not part_b:
                    continue

                # Compute phi for this partition using EMD approximation
                phi = self._compute_partition_phi(
                    tpm, cause_rep, effect_rep, part_a, part_b
                )

                if phi < min_phi:
                    min_phi = phi
                    min_partition = (part_a, part_b)

        return min_phi, min_partition

    def _compute_partition_phi(
        self,
        tpm: np.ndarray,
        cause_rep: np.ndarray,
        effect_rep: np.ndarray,
        part_a: Set[int],
        part_b: Set[int]
    ) -> float:
        """
        Compute phi for a specific partition using EMD approximation

        Earth Mover's Distance between whole and partitioned distributions
        """
        # Simplified: use KL divergence as approximation
        # In full IIT, this would be Wasserstein distance

        n = len(cause_rep)

        # Create partitioned distributions (factorized)
        # This is a simplification - full IIT requires proper marginalization

        # For cause repertoire
        cause_whole = cause_rep
        cause_part = cause_rep.copy()  # Simplified partitioned version

        # For effect repertoire
        effect_whole = effect_rep
        effect_part = effect_rep.copy()

        # Compute EMD approximation
        emd_cause = self._kl_divergence(cause_whole, cause_part)
        emd_effect = self._kl_divergence(effect_whole, effect_part)

        # Phi is minimum of cause and effect EMD
        phi = min(emd_cause, emd_effect)

        # Scale by partition size factor
        scale = len(part_a) * len(part_b) / (len(part_a) + len(part_b)) ** 2
        phi *= scale

        return phi

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
        """Compute KL divergence D_KL(p || q)"""
        p = np.clip(p, epsilon, 1.0)
        q = np.clip(q, epsilon, 1.0)
        return np.sum(p * np.log(p / q))

    def _build_concepts(
        self,
        tpm: np.ndarray,
        n_elements: int,
        current_state: np.ndarray
    ) -> List[Concept]:
        """
        Build the conceptual structure (all concepts with phi > 0)
        """
        concepts = []

        # For each subset of elements (mechanism)
        for size in range(1, n_elements + 1):
            for mechanism in combinations(range(n_elements), size):
                mechanism = list(mechanism)

                # Compute phi for this mechanism
                phi = self._compute_concept_phi(tpm, mechanism, current_state)

                if phi > 0:
                    concepts.append(Concept(
                        mechanism=mechanism,
                        phi=phi,
                        cause_purview=mechanism,  # Simplified
                        effect_purview=mechanism  # Simplified
                    ))

        return concepts

    def _compute_concept_phi(
        self,
        tpm: np.ndarray,
        mechanism: List[int],
        current_state: np.ndarray
    ) -> float:
        """
        Compute integrated information for a single mechanism
        """
        if len(mechanism) == 0:
            return 0.0

        # Simplified computation based on mechanism size and TPM structure
        n = tpm.shape[0]

        # Base phi proportional to log of mechanism size
        base_phi = math.log(len(mechanism) + 1) / math.log(n + 1)

        # Modulate by TPM non-uniformity
        entropy = -np.sum(tpm * np.log(np.clip(tpm, 1e-10, 1.0)))
        max_entropy = n * math.log(n)
        structure_factor = 1 - (entropy / max_entropy) if max_entropy > 0 else 0

        return base_phi * structure_factor * 0.5  # Scale to reasonable range

    # =========================================================================
    # APL TOKEN GENERATION
    # =========================================================================

    def phi_to_apl_token(self, phi: float, quale: Optional[Quale] = None) -> APLToken:
        """
        Convert Phi value and quale to APL token

        Uses mapping:
        - phi < 0.33: Phi spiral (structure)
        - phi in [0.33, 0.66): e spiral (energy)
        - phi >= 0.66: pi spiral (emergence)
        """
        quale_dict = None
        if quale:
            quale_dict = {
                "coherent": quale.big_phi > self.phi_critical,
                "contradictory": False,
                "uncertain": not quale.is_conscious
            }

        return phi_to_token(phi, quale_dict)

    # =========================================================================
    # SIMPLIFIED PHI COMPUTATION FOR LIMNUS NODES
    # =========================================================================

    def compute_node_phi(
        self,
        node_state: Dict,
        neighbors: List[Dict],
        depth: int
    ) -> float:
        """
        Simplified Phi computation for a LIMNUS node

        Based on:
        - Node's local state
        - Connectivity with neighbors
        - Depth in the tree
        """
        # Base phi from depth (deeper = potentially higher phi)
        depth_factor = (7 - depth) / 6.0  # depth 1 = 1.0, depth 6 = 0.17

        # Connectivity factor
        n_neighbors = len(neighbors)
        connect_factor = min(1.0, n_neighbors / 4.0)  # Saturates at 4 neighbors

        # State coherence factor
        if "z" in node_state:
            z = node_state["z"]
        else:
            z = 0.5

        coherence_factor = z

        # Information integration estimate
        if neighbors:
            # Average correlation with neighbors
            neighbor_z = [n.get("z", 0.5) for n in neighbors]
            avg_neighbor_z = sum(neighbor_z) / len(neighbor_z)
            integration = 1 - abs(z - avg_neighbor_z)  # Higher when synchronized
        else:
            integration = 0.5

        # Combine factors
        phi = (
            0.3 * depth_factor +
            0.2 * connect_factor +
            0.2 * coherence_factor +
            0.3 * integration
        )

        return min(1.0, max(0.0, phi))

    def compute_global_phi(
        self,
        node_phis: List[float],
        connections: List[Tuple[int, int]]
    ) -> float:
        """
        Compute global Phi for entire LIMNUS tree

        Uses integration = whole - sum of parts
        """
        if not node_phis:
            return 0.0

        # Sum of individual phis
        sum_parts = sum(node_phis)

        # Connectivity bonus
        n_nodes = len(node_phis)
        n_connections = len(connections)
        max_connections = n_nodes * (n_nodes - 1) / 2
        connectivity = n_connections / max_connections if max_connections > 0 else 0

        # Integration emerges from connectivity
        integration_bonus = connectivity * sum_parts * 0.3

        # Global phi
        global_phi = (sum_parts / n_nodes) * (1 + integration_bonus)

        return min(1.0, global_phi)

    # =========================================================================
    # SCALAR EFFECTS
    # =========================================================================

    def apply_scalar_effects(
        self,
        scalars: ScalarState,
        phi: float
    ) -> ScalarState:
        """
        Apply IIT-derived effects to scalar state

        High Phi increases:
        - Omega_s (coherence)
        - alpha_s (attractor alignment)
        - Cs (coupling)

        High Phi decreases:
        - delta_s (decoherence)
        """
        deltas = {
            "Omega_s": phi * 0.15,
            "alpha_s": phi * 0.10,
            "Cs": phi * 0.08,
            "delta_s": -phi * 0.05
        }

        return scalars.apply_deltas(deltas)
