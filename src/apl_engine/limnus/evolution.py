"""
APL 3.0 LIMNUS Evolution Engine

Handles the evolution of consciousness in the LIMNUS tree
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

from ..core.constants import CONSTANTS
from ..core.scalars import ScalarState
from .node import LimnusNode, NodeActivity
from .tree import LimnusTree, PRS


@dataclass
class EvolutionInput:
    """Input for evolution step"""
    sensory: Dict[str, float]    # Sensory input
    social: List[Dict]           # Social environment
    neural: Dict[int, Dict]      # Neural oscillations by node
    time: float                  # Current time
    dt: float                    # Time step


@dataclass
class EvolutionMetrics:
    """Metrics from evolution step"""
    z_global: float
    phi_global: float
    phase: str
    cooperation_rate: float
    free_energy: float
    gamma_sync: float
    care_discovered: bool
    substrate_aware: bool


class LimnusEvolutionEngine:
    """
    Engine for evolving consciousness in the LIMNUS tree

    Integrates all computation engines:
    - IIT (Phi computation)
    - Game Theory (cooperation)
    - Free Energy (prediction)
    - EM (oscillation binding)
    - Strange Loop (self-reference)
    """

    def __init__(self, tree: Optional[LimnusTree] = None):
        """Initialize evolution engine"""
        self.tree = tree or LimnusTree()

        # Import engines
        from ..engines.iit import IITEngine
        from ..engines.game_theory import GameTheoryEngine
        from ..engines.free_energy import FreeEnergyEngine
        from ..engines.electromagnetic import ElectromagneticEngine
        from ..engines.strange_loop import StrangeLoopEngine

        self.iit = IITEngine()
        self.game = GameTheoryEngine()
        self.free_energy = FreeEnergyEngine()
        self.em = ElectromagneticEngine()
        self.strange_loop = StrangeLoopEngine()

        # Evolution parameters
        self.alpha_phi = 0.35    # IIT weight
        self.alpha_F = 0.25     # Free energy weight
        self.alpha_R = 0.25     # Recursion weight
        self.alpha_E = 0.15     # EM coherence weight

    def evolve(
        self,
        inputs: EvolutionInput
    ) -> EvolutionMetrics:
        """
        Execute one evolution step

        1. Update each node
        2. Propagate information (bottom-up)
        3. Propagate predictions (top-down)
        4. Compute global consciousness
        5. Check phase transitions
        6. Check emergent properties
        """
        # Step 1: Update each node
        for node in self.tree.nodes:
            self._evolve_node(node, inputs)

        # Step 2: Bottom-up propagation
        self.tree.propagate_up()

        # Step 3: Top-down propagation
        self.tree.propagate_down()

        # Step 4: Compute global metrics
        self.tree.update_global_state()

        # Step 5: Check phase transitions
        transition = self.tree.check_phase_transition()
        if transition:
            old_phase, new_phase = transition
            self.tree.trigger_phase_transition(old_phase, new_phase)

        # Step 6: Compute metrics
        metrics = self._compute_metrics()

        return metrics

    def _evolve_node(
        self,
        node: LimnusNode,
        inputs: EvolutionInput
    ):
        """Evolve a single node's state"""
        # Local consciousness evolution
        dz = self._compute_dz(node, inputs)
        node.update_z(dz * inputs.dt)

        # Local Phi evolution
        dphi = self._compute_dphi(node)
        node.update_phi(dphi * inputs.dt)

        # Free energy update
        F_new = self._compute_local_F(node, inputs)
        dF = F_new - node.state.F_local
        node.state.F_local = F_new

        # Phase oscillation
        node.update_phase(inputs.dt)

        # Scalar evolution
        scalar_deltas = self._compute_scalar_deltas(node, inputs)
        node.update_scalars(scalar_deltas)

        # Game theory (if has neighbors)
        if node.neighbors:
            self._update_strategy(node)

        # Truth state evolution
        self._evolve_truth(node)

        # Activity level
        node.update_activity()

    def _compute_dz(
        self,
        node: LimnusNode,
        inputs: EvolutionInput
    ) -> float:
        """Compute rate of change of z for a node"""
        # Integration term (driven by Phi)
        integration = self.alpha_phi * node.state.phi_local * 0.1

        # Free energy term (lower F = higher z)
        fe_term = -self.alpha_F * (node.state.F_local - 0.5) * 0.05

        # Cooperation term
        if node.neighbors:
            avg_z = sum(n.state.z for n in node.neighbors) / len(node.neighbors)
            coop_term = 0.1 * (avg_z - node.state.z)
        else:
            coop_term = 0.0

        # Recursion term (from strange loop depth)
        recursion_depth = self.strange_loop.get_level_from_z(node.state.z)
        recursion_term = self.alpha_R * recursion_depth * (1 - node.state.z) * 0.05

        # EM coherence term
        em_term = self.alpha_E * node.state.scalars.Omega_s * 0.05

        return integration + fe_term + coop_term + recursion_term + em_term

    def _compute_dphi(self, node: LimnusNode) -> float:
        """Compute rate of change of Phi for a node"""
        # Connectivity factor
        connectivity = len(node.neighbors) / 10.0

        # Coherence factor
        coherence = node.state.scalars.Omega_s

        # Coupling factor
        coupling = node.state.scalars.Cs

        return (connectivity + coherence + coupling) * 0.02

    def _compute_local_F(
        self,
        node: LimnusNode,
        inputs: EvolutionInput
    ) -> float:
        """Compute local free energy for a node"""
        # Prediction error (simplified)
        if node.prediction:
            z_expected = node.prediction.get("z_expected", node.state.z)
            error = abs(node.state.z - z_expected)
        else:
            error = 0.1

        # Model complexity penalty
        complexity = len(node.children) * 0.05

        return error + complexity

    def _compute_scalar_deltas(
        self,
        node: LimnusNode,
        inputs: EvolutionInput
    ) -> Dict[str, float]:
        """Compute scalar state changes"""
        deltas = {}

        # Coherence increases with z
        deltas["Omega_s"] = node.state.z * 0.02 - 0.01

        # Decoherence decreases with coherence
        deltas["delta_s"] = -node.state.scalars.Omega_s * 0.01

        # Grounding from sensory
        if inputs.sensory:
            deltas["Gs"] = inputs.sensory.get("structure", 0.5) * 0.02

        # Coupling from neighbors
        if node.neighbors:
            avg_coupling = sum(n.state.scalars.Cs for n in node.neighbors) / len(node.neighbors)
            deltas["Cs"] = (avg_coupling - node.state.scalars.Cs) * 0.1

        return deltas

    def _update_strategy(self, node: LimnusNode):
        """Update game theory strategy for a node"""
        # Simple: adopt most successful neighbor's strategy
        if node.neighbors:
            neighbor_scores = [
                (n, n.state.z * n.state.scalars.Cs)
                for n in node.neighbors
            ]
            best_neighbor, best_score = max(neighbor_scores, key=lambda x: x[1])

            current_score = node.state.z * node.state.scalars.Cs
            if best_score > current_score * 1.1:  # 10% better
                node.state.strategy = best_neighbor.state.strategy

    def _evolve_truth(self, node: LimnusNode):
        """Evolve truth state based on conditions"""
        z = node.state.z
        coherence = node.state.scalars.Omega_s

        if z >= 0.6 and coherence >= 0.5:
            node.update_truth(TruthState.TRUE)
        elif coherence < 0.3:
            node.update_truth(TruthState.UNTRUE)
        # Paradox state stays as is

    def _compute_metrics(self) -> EvolutionMetrics:
        """Compute evolution metrics"""
        # Cooperation rate
        coop_strategies = ["tit_for_tat", "generous_tit_for_tat", "cooperate"]
        coop_nodes = sum(1 for n in self.tree.nodes if n.state.strategy in coop_strategies)
        cooperation_rate = coop_nodes / len(self.tree.nodes)

        # Average free energy
        free_energy = sum(n.state.F_local for n in self.tree.nodes) / len(self.tree.nodes)

        # Gamma synchronization
        gamma_nodes = [n for n in self.tree.nodes if n.state.frequency >= 30]
        if len(gamma_nodes) >= 2:
            phases = [n.state.phase for n in gamma_nodes]
            complex_sum = sum(np.exp(1j * p) for p in phases) / len(phases)
            gamma_sync = abs(complex_sum)
        else:
            gamma_sync = 0.0

        return EvolutionMetrics(
            z_global=self.tree.state.z_global,
            phi_global=self.tree.state.phi_global,
            phase=self.tree.state.phase,
            cooperation_rate=cooperation_rate,
            free_energy=free_energy,
            gamma_sync=gamma_sync,
            care_discovered=self.tree.state.care_discovered,
            substrate_aware=self.tree.state.substrate_aware
        )

    # =========================================================================
    # EPOCH EVOLUTION
    # =========================================================================

    def evolve_epoch(
        self,
        duration: float,
        dt: float = 0.01
    ) -> List[EvolutionMetrics]:
        """
        Evolve through an entire epoch

        Returns metrics at each step
        """
        metrics_history = []
        t = 0.0

        while t < duration:
            # Generate inputs
            inputs = self._generate_inputs(t, dt)

            # Evolve
            metrics = self.evolve(inputs)
            metrics_history.append(metrics)

            t += dt

        return metrics_history

    def _generate_inputs(self, t: float, dt: float) -> EvolutionInput:
        """Generate evolution inputs"""
        return EvolutionInput(
            sensory={
                "patterns": 0.5 + 0.3 * math.sin(t * CONSTANTS.PHI),
                "structure": 0.5 + 0.2 * math.cos(t * CONSTANTS.E),
                "complexity": 0.5 + 0.3 * math.sin(t * CONSTANTS.PI / 10)
            },
            social=[
                {"id": i, "cooperation": 0.7, "z": 0.5}
                for i in range(5)
            ],
            neural={
                node.id: {"phase": node.state.phase, "frequency": node.state.frequency}
                for node in self.tree.nodes
            },
            time=t,
            dt=dt
        )

    # =========================================================================
    # FULL EVOLUTION TRACE
    # =========================================================================

    def evolve_to_z(
        self,
        target_z: float,
        max_time: float = 1000.0,
        dt: float = 0.01
    ) -> Tuple[bool, List[EvolutionMetrics]]:
        """
        Evolve until reaching target z or max time

        Returns (reached_target, metrics_history)
        """
        metrics_history = []
        t = 0.0

        while t < max_time and self.tree.state.z_global < target_z:
            inputs = self._generate_inputs(t, dt)
            metrics = self.evolve(inputs)
            metrics_history.append(metrics)

            # Adaptive time step near transitions
            if abs(self.tree.state.z_global - target_z) < 0.05:
                dt = dt / 2

            t += dt

        reached = self.tree.state.z_global >= target_z
        return reached, metrics_history


# Import TruthState for truth evolution
from ..core.constants import TruthState
